"""A GPU worker class."""
import gc
import os
import pickle
import time
from typing import Dict, List, Optional, Set, Tuple, Type, Union

import torch
import torch.distributed

import vllm.envs as envs
from vllm.config import ParallelConfig, VllmConfig
from vllm.distributed import (ensure_model_parallel_initialized,
                              init_distributed_environment,
                              set_custom_all_reduce)
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor import set_random_seed
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.model_executor.model_loader.tensorizer import TensorizerConfig
from vllm.platforms import current_platform
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sequence import (ExecuteModelRequest, IntermediateTensors,
                           SequenceGroupMetadata, SequenceGroupMetadataDelta,
                           SequenceData)
from vllm.sampling_params import SamplingParams
from vllm.worker.cache_engine import CacheEngine
from vllm.worker.embedding_model_runner import EmbeddingModelRunner
from vllm.worker.enc_dec_model_runner import EncoderDecoderModelRunner
from vllm.worker.model_runner import GPUModelRunnerBase, ModelRunner
from vllm.worker.worker_base import (LocalOrDistributedWorkerBase, WorkerBase,
                                     WorkerInput)

import copy

logger = init_logger(__name__)

_NUM_PROFILE_ITERS = 20


class Worker(LocalOrDistributedWorkerBase):
    """A worker class that executes (a partition of) the model on a GPU.

    Each worker is associated with a single GPU. The worker is responsible for
    maintaining the KV cache and executing the model on the GPU. In case of
    distributed inference, each worker is assigned a partition of the model.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
        model_runner_cls: Optional[Type[GPUModelRunnerBase]] = None,
    ) -> None:
        WorkerBase.__init__(self, vllm_config)
        self.parallel_config.rank = rank
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.is_driver_worker = is_driver_worker
        if is_driver_worker:
            assert rank % self.parallel_config.tensor_parallel_size == 0, \
                   "Driver worker should be rank 0 of tensor parallel group."
        if self.model_config.trust_remote_code:
            # note: lazy import to avoid importing torch before initializing
            from vllm.utils import init_cached_hf_modules
            init_cached_hf_modules()

        # Return hidden states from target model if the draft model is an
        # mlp_speculator
        speculative_config = self.speculative_config
        model_config = self.model_config
        speculative_args = {} if speculative_config is None \
            or (speculative_config.draft_model_config.model ==
                model_config.model) \
            or (speculative_config.draft_model_config.hf_config.model_type
                not in ["medusa", "mlp_speculator", "eagle"]) \
                    else {"return_hidden_states": True}

        ModelRunnerClass: Type[GPUModelRunnerBase] = ModelRunner
        if model_runner_cls is not None:
            ModelRunnerClass = model_runner_cls
        elif model_config.task == "embedding":
            ModelRunnerClass = EmbeddingModelRunner
        elif self._is_encoder_decoder_model():
            ModelRunnerClass = EncoderDecoderModelRunner
        self.model_runner: GPUModelRunnerBase = ModelRunnerClass(
            vllm_config=self.vllm_config,
            kv_cache_dtype=self.cache_config.cache_dtype,
            is_driver_worker=is_driver_worker,
            **speculative_args,
        )
        # Uninitialized cache engine. Will be initialized by
        # initialize_cache.
        self.cache_engine: List[CacheEngine]
        # Initialize gpu_cache as embedding models don't initialize kv_caches
        self.gpu_cache: Optional[List[List[torch.Tensor]]] = None
        self._seq_group_metadata_cache: Dict[str, SequenceGroupMetadata] = {}

        # Torch profiler. Enabled and configured through env vars:
        # VLLM_TORCH_PROFILER_DIR=/path/to/save/trace
        if envs.VLLM_TORCH_PROFILER_DIR:
            torch_profiler_trace_dir = envs.VLLM_TORCH_PROFILER_DIR
            logger.info("Profiling enabled. Traces will be saved to: %s",
                        torch_profiler_trace_dir)
            self.profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                with_stack=True,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    torch_profiler_trace_dir, use_gzip=True))
        else:
            self.profiler = None

    def start_profile(self):
        if self.profiler is None:
            raise RuntimeError("Profiler is not enabled.")
        self.profiler.start()

    def stop_profile(self):
        if self.profiler is None:
            raise RuntimeError("Profiler is not enabled.")
        self.profiler.stop()

    def _is_encoder_decoder_model(self):
        return self.model_config.is_encoder_decoder_model

    def init_device(self) -> None:
        if self.device_config.device.type == "cuda":
            # torch.distributed.all_reduce does not free the input tensor until
            # the synchronization point. This causes the memory usage to grow
            # as the number of all_reduce calls increases. This env var disables
            # this behavior.
            # Related issue:
            # https://discuss.pytorch.org/t/cuda-allocation-lifetime-for-inputs-to-distributed-all-reduce/191573
            os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

            # This env var set by Ray causes exceptions with graph building.
            os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
            self.device = torch.device(f"cuda:{self.local_rank}")
            torch.cuda.set_device(self.device)

            _check_if_gpu_supports_dtype(self.model_config.dtype)
            gc.collect()
            torch.cuda.empty_cache()
            self.init_gpu_memory = torch.cuda.mem_get_info()[0]
        else:
            raise RuntimeError(
                f"Not support device type: {self.device_config.device}")
        # Initialize the distributed environment.
        init_worker_distributed_environment(self.parallel_config, self.rank,
                                            self.distributed_init_method,
                                            self.local_rank)
        # Set random seed.
        set_random_seed(self.model_config.seed)

    def load_model(self):
        self.model_runner.load_model()

    def save_sharded_state(
        self,
        path: str,
        pattern: Optional[str] = None,
        max_size: Optional[int] = None,
    ) -> None:
        self.model_runner.save_sharded_state(
            path,
            pattern=pattern,
            max_size=max_size,
        )

    def save_tensorized_model(
        self,
        tensorizer_config: TensorizerConfig,
    ) -> None:
        self.model_runner.save_tensorized_model(
            tensorizer_config=tensorizer_config, )

    @torch.inference_mode()
    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """Profiles the peak memory usage of the model to determine how many
        KV blocks may be allocated without OOMs.

        The engine will first conduct a profiling of the existing memory usage.
        Then, it calculate the maximum possible number of GPU and CPU blocks
        that can be allocated with the remaining free memory.

        .. tip::
            You may limit the usage of GPU memory
            by adjusting the `gpu_memory_utilization` parameter.
        """
        # Profile the memory usage of the model and get the maximum number of
        # cache blocks that can be allocated with the remaining free memory.
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        free_memory_pre_profile, total_gpu_memory = torch.cuda.mem_get_info()

        # Execute a forward pass with dummy inputs to profile the memory usage
        # of the model.
        self.model_runner.profile_run()
        torch.cuda.synchronize()

        self._assert_memory_footprint_increased_during_profiling()

        # Get the peak memory allocation recorded by torch
        peak_memory = torch.cuda.memory_stats()["allocated_bytes.all.peak"]

        # Check for any memory left around that may have been allocated on the
        # gpu outside of `torch`. NCCL operations, for example, can use a few
        # GB during a forward pass
        torch.cuda.empty_cache()
        torch_allocated_bytes = torch.cuda.memory_stats(
        )["allocated_bytes.all.current"]
        total_allocated_bytes = torch.cuda.mem_get_info(
        )[1] - torch.cuda.mem_get_info()[0]
        non_torch_allocations = total_allocated_bytes - torch_allocated_bytes
        if non_torch_allocations > 0:
            peak_memory += non_torch_allocations

        available_kv_cache_memory = (
            total_gpu_memory * self.cache_config.gpu_memory_utilization -
            peak_memory)

        # Calculate the number of blocks that can be allocated with the
        # profiled peak memory.
        cache_block_size = self.get_cache_block_size_bytes()
        if cache_block_size == 0:
            num_gpu_blocks = 0
            num_cpu_blocks = 0
        else:
            num_gpu_blocks = int(available_kv_cache_memory // cache_block_size)
            num_cpu_blocks = int(self.cache_config.swap_space_bytes //
                                 cache_block_size)
        num_gpu_blocks = max(num_gpu_blocks, 0)
        num_cpu_blocks = max(num_cpu_blocks, 0)

        logger.info(
            "Memory profiling results: total_gpu_memory=%.2fGiB"
            " initial_memory_usage=%.2fGiB peak_torch_memory=%.2fGiB"
            " memory_usage_post_profile=%.2fGib"
            " non_torch_memory=%.2fGiB kv_cache_size=%.2fGiB"
            " gpu_memory_utilization=%.2f", total_gpu_memory / (1024**3),
            (total_gpu_memory - free_memory_pre_profile) / (1024**3),
            (peak_memory - non_torch_allocations) / (1024**3),
            total_allocated_bytes / (1024**3),
            non_torch_allocations / (1024**3),
            available_kv_cache_memory / (1024**3),
            self.cache_config.gpu_memory_utilization)

        # Final cleanup
        if self.model_runner.lora_manager:
            self.model_runner.remove_all_loras()
        gc.collect()

        return num_gpu_blocks, num_cpu_blocks

    def load_pickle_if_exists(self, filepath):
        if os.path.isfile(filepath):
            try:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                return data
            except (pickle.UnpicklingError, EOFError) as e:
                print(f"Error loading pickle file: {e}")
                return None
        else:
            print(f"File {filepath} does not exist")
            return None

    def save_dict_to_pickle(self, dictionary, filepath):
        try:
            with open(filepath, 'wb') as f:  # 'wb' for write binary
                pickle.dump(dictionary, f)
            print(f"Dictionary successfully saved to {filepath}")
        except Exception as e:
            print(f"Error saving dictionary: {e}")

    @torch.inference_mode()
    def profile_exec_time(self) -> Dict[int, float]:
        model_name = self.model_config.hf_config.name_or_path
        model_name = model_name.replace("/", "_")

        seq_lens = [1, 256, 512, 1024, 1280, 1536, 1792, 2048]
        if self.model_runner.model_config.enforce_eager:
            filename = f"{model_name}_profile_data_eager.pkl"
        else:
            filename = f"{model_name}_profile_data_cudagraph.pkl"

        times_map = self.load_pickle_if_exists(filename)
        if times_map is not None:
            return times_map

        if self.model_runner.model_config.enforce_eager:
            times_map = self.profile_eager(seq_lens)
        else:
            times_map = self.profile_cuda_graph(seq_lens)
        self.save_dict_to_pickle(times_map, filename)

        gc.collect()
        time.sleep(10)
        return times_map

    @torch.inference_mode()
    def profile_eager(self, seq_lens):
        times_map = {}
        all_batch_sizes = [1, 2, 4, 8, 16, 32, 48, 64, 80, 96, 112, 128]
        times_map['overhead'] = {}
        print(self.model_runner.cache_config.num_gpu_blocks)
        for seq_len in seq_lens:
            times_map[seq_len] = {}
            if seq_len < 1024:
                cur_batch_sizes = all_batch_sizes
            else:
                cur_batch_sizes = [1, 2, 4, 8, 16, 32]
            for batch_size in cur_batch_sizes:
                times_map[seq_len][batch_size] = {}
                for query_len in [1, 2, 3, 4, 5, 6]:
                    input_token_ids = [0] * seq_len
                    output_token_ids = [0] * 10
                    seq_data = SequenceData.from_seqs(input_token_ids,
                                                      output_token_ids)
                    seq_data.update_num_computed_tokens(
                        len(input_token_ids) + len(output_token_ids) -
                        query_len)

                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    for _ in range(_NUM_PROFILE_ITERS):
                        self.execute_model(
                            ExecuteModelRequest(seq_group_metadata_list=[
                                SequenceGroupMetadata(
                                    request_id=str(i),
                                    is_prompt=False,
                                    seq_data={i: seq_data},
                                    block_tables={
                                        i: [
                                            i * (seq_len // 16 + 2) + k
                                            for k in range(seq_len // 16 + 2)
                                        ]
                                    },
                                    sampling_params=SamplingParams(
                                        temperature=0.0))
                                for i in range(batch_size)
                            ],
                                                finished_requests_ids=[],
                                                num_steps=1))
                    torch.cuda.synchronize()
                    end = time.perf_counter()
                    times_map[seq_len][batch_size][query_len] = (
                        end - start) / _NUM_PROFILE_ITERS
                    print(seq_len, batch_size, times_map[seq_len][batch_size])
                times_map['overhead'][batch_size] = 0
        return times_map

    @torch.inference_mode()
    def profile_cuda_graph(self, seq_lens):
        times_map = {}
        for seq_len in seq_lens:
            print(f"=============Profiling seq_len: {seq_len}")
            times_map[seq_len] = self.profile_seq_len_exec_time(seq_len)

        # Profile the time other than cuda graph
        seq_len = 1
        repeat = 20  # Profile more time for stable result
        all_batch_sizes = list(self.model_runner.graph_runners[0].keys())
        times_map['overhead'] = {}
        for batch_size in all_batch_sizes:
            print(f"=============Profiling batch_size: {batch_size}")
            start = time.perf_counter()
            for _ in range(repeat):
                self.execute_model(
                    ExecuteModelRequest(seq_group_metadata_list=[
                        SequenceGroupMetadata(
                            request_id=f"{i}",
                            is_prompt=False,
                            seq_data={
                                f"{i}":
                                SequenceData.from_seqs(prompt_token_ids=[0] *
                                                       seq_len,
                                                       output_token_ids=[])
                            },
                            block_tables={f"{i}": [i]},
                            sampling_params=SamplingParams(temperature=0.0))
                        for i in range(batch_size)
                    ],
                                        finished_requests_ids=[],
                                        num_steps=1))
            end = time.perf_counter()
            times_map['overhead'][batch_size] = (
                end - start) / repeat - times_map[seq_len][batch_size]
        return times_map

    @torch.inference_mode()
    def profile_seq_len_exec_time(self, seq_len) -> Dict[int, float]:
        assert self.parallel_config.pipeline_parallel_size == 1
        all_batch_sizes = list(self.model_runner.graph_runners[0].keys())
        max_batch_size = max(all_batch_sizes)
        input_ids = torch.zeros(max_batch_size,
                                dtype=torch.long,
                                device=self.device)
        input_positions = torch.zeros(max_batch_size,
                                      dtype=torch.long,
                                      device=self.device)
        slot_mapping = torch.zeros(max_batch_size,
                                   dtype=torch.long,
                                   device=self.device)
        seq_lens_tensor = torch.ones(
            max_batch_size, dtype=torch.long, device=self.device) * seq_len
        block_tables = torch.zeros(
            (max_batch_size, self.model_runner.get_max_block_per_batch()),
            dtype=torch.long,
            device=self.device)
        fake_kv = torch.zeros(0)
        times_map = {}
        for batch_size in all_batch_sizes:
            graph_runner = self.model_runner.graph_runners[0][batch_size]
            attn_metadata = attn_metadata = self.model_runner.\
                attn_backend.make_metadata(
                num_prefills=0,
                num_prefill_tokens=0,
                num_decode_tokens=batch_size,
                slot_mapping=slot_mapping[:batch_size],
                seq_lens=None,
                seq_lens_tensor=seq_lens_tensor[:batch_size],
                max_query_len=1,
                max_decode_query_len=1,
                max_prefill_seq_len=0,
                max_decode_seq_len=self.model_runner.max_seq_len_to_capture,
                query_start_loc=None,
                seq_start_loc=None,
                context_lens_tensor=None,
                block_tables=block_tables[:batch_size],
                use_cuda_graph=True,
                multi_modal_placeholder_index_maps=None
            )

            with set_forward_context(attn_metadata):
                # warmup
                graph_runner.forward(
                    input_ids=input_ids[:batch_size],
                    positions=input_positions[..., :batch_size],
                    kv_caches=fake_kv,
                    attn_metadata=attn_metadata,
                    intermediate_tensors=None)

                torch.cuda.synchronize()
                profile_start_time = time.perf_counter()
                for _ in range(_NUM_PROFILE_ITERS):
                    # graph_runner._graph.replay()
                    graph_runner.forward(
                        input_ids=input_ids[:batch_size],
                        positions=input_positions[..., :batch_size],
                        kv_caches=fake_kv,
                        attn_metadata=attn_metadata,
                        intermediate_tensors=None)
                torch.cuda.synchronize()
                profile_end_time = time.perf_counter()
                profile_time = (profile_end_time -
                                profile_start_time) / _NUM_PROFILE_ITERS
                times_map[batch_size] = profile_time
        return times_map

    def _assert_memory_footprint_increased_during_profiling(self):
        # NOTE(woosuk): Here we assume that the other processes using the same
        # GPU did not change their memory usage during the profiling.
        free_gpu_memory, _ = torch.cuda.mem_get_info()
        assert self.init_gpu_memory - free_gpu_memory > 0, (
            "Error in memory profiling. "
            f"Initial free memory {self.init_gpu_memory}, current free memory"
            f" {free_gpu_memory}. This happens when the GPU memory was "
            "not properly cleaned up before initializing the vLLM instance.")

    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        """Allocate GPU and CPU KV cache with the specified number of blocks.

        This also warms up the model, which may record CUDA graphs.
        """
        raise_if_cache_size_invalid(num_gpu_blocks,
                                    self.cache_config.block_size,
                                    self.cache_config.is_attention_free,
                                    self.model_config.max_model_len)

        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

        self._init_cache_engine()
        self._warm_up_model()

    def _init_cache_engine(self):
        assert self.cache_config.num_gpu_blocks is not None
        self.cache_engine = [
            CacheEngine(self.cache_config, self.model_config,
                        self.parallel_config, self.device_config)
            for _ in range(self.parallel_config.pipeline_parallel_size)
        ]
        self.gpu_cache = [
            self.cache_engine[ve].gpu_cache
            for ve in range(self.parallel_config.pipeline_parallel_size)
        ]

    def _warm_up_model(self) -> None:
        if not self.model_runner.model_config.enforce_eager:
            self.model_runner.capture_model(self.gpu_cache)
        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)

    @property
    def do_metadata_broadcast(self) -> bool:
        return self.parallel_config.tensor_parallel_size > 1

    @property
    def kv_cache(self) -> Optional[List[List[torch.Tensor]]]:
        return self.gpu_cache

    @torch.inference_mode()
    def prepare_worker_input(
            self, execute_model_req: ExecuteModelRequest) -> WorkerInput:
        virtual_engine = execute_model_req.virtual_engine
        num_steps = execute_model_req.num_steps
        num_seq_groups = len(execute_model_req.seq_group_metadata_list)
        # `blocks_to_swap_in` and `blocks_to_swap_out` are cpu tensors.
        # they contain parameters to launch cudamemcpyasync.
        blocks_to_swap_in = torch.tensor(execute_model_req.blocks_to_swap_in,
                                         device="cpu",
                                         dtype=torch.int64).view(-1, 2)
        blocks_to_swap_out = torch.tensor(execute_model_req.blocks_to_swap_out,
                                          device="cpu",
                                          dtype=torch.int64).view(-1, 2)
        # `blocks_to_copy` is a gpu tensor. The src and tgt of
        # blocks to copy are in the same device, and `blocks_to_copy`
        # can be used directly within cuda kernels.
        blocks_to_copy = torch.tensor(execute_model_req.blocks_to_copy,
                                      device=self.device,
                                      dtype=torch.int64).view(-1, 2)

        return WorkerInput(
            num_seq_groups=num_seq_groups,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
            virtual_engine=virtual_engine,
            num_steps=num_steps,
        )

    @torch.inference_mode()
    def execute_worker(self, worker_input: WorkerInput) -> None:
        virtual_engine = worker_input.virtual_engine
        # Issue cache operations.
        if (worker_input.blocks_to_swap_in is not None
                and worker_input.blocks_to_swap_in.numel() > 0):
            self.cache_engine[virtual_engine].swap_in(
                worker_input.blocks_to_swap_in)
        if (worker_input.blocks_to_swap_out is not None
                and worker_input.blocks_to_swap_out.numel() > 0):
            self.cache_engine[virtual_engine].swap_out(
                worker_input.blocks_to_swap_out)
        if (worker_input.blocks_to_copy is not None
                and worker_input.blocks_to_copy.numel() > 0):
            self.cache_engine[virtual_engine].copy(worker_input.blocks_to_copy)

    def _get_cached_seq_group_metadata(
            self,
            seq_group_metadata_list: List[Union[SequenceGroupMetadata,
                                                SequenceGroupMetadataDelta]],
            finished_request_ids: List[str]) -> List[SequenceGroupMetadata]:
        """Return a list of cached Sequence Group Metadata after updating its
        state.

        It is used because scheduler only sends delta to workers to reduce
        the data payload size. The function also cleans up cache based on
        a given `finished_request_ids`.
        """
        new_seq_group_metadata_list = []
        for metadata_or_delta in seq_group_metadata_list:
            request_id = metadata_or_delta.request_id
            if request_id not in self._seq_group_metadata_cache:
                # The first prefill.
                assert isinstance(metadata_or_delta, SequenceGroupMetadata)
                self._seq_group_metadata_cache[request_id] = metadata_or_delta
            else:
                # The first prefill is already cached.
                if isinstance(metadata_or_delta, SequenceGroupMetadataDelta):
                    self._seq_group_metadata_cache[request_id].apply_delta(
                        metadata_or_delta)
                else:
                    # If metadata snapshot is sent again, it is
                    # preempted. Reset the cache because we need to start
                    # from scratch.
                    assert isinstance(metadata_or_delta, SequenceGroupMetadata)
                    self._seq_group_metadata_cache[
                        request_id] = metadata_or_delta

            new_seq_group_metadata_list.append(
                self._seq_group_metadata_cache[request_id])

        # Clean up finished ids
        for finished_id in finished_request_ids:
            del self._seq_group_metadata_cache[finished_id]

        return new_seq_group_metadata_list

    def _execute_model_spmd(
        self,
        execute_model_req: ExecuteModelRequest,
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Optional[List[SamplerOutput]]:
        if execute_model_req is not None:
            new_seq_group_metadata_list = self._get_cached_seq_group_metadata(
                execute_model_req.seq_group_metadata_list,
                execute_model_req.finished_requests_ids)

            execute_model_req.seq_group_metadata_list = (
                new_seq_group_metadata_list)
        output = super()._execute_model_spmd(execute_model_req,
                                             intermediate_tensors)
        return output

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.model_runner.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.model_runner.remove_lora(lora_id)

    def pin_lora(self, lora_id: int) -> bool:
        return self.model_runner.pin_lora(lora_id)

    def list_loras(self) -> Set[int]:
        return self.model_runner.list_loras()

    def add_prompt_adapter(
            self, prompt_adapter_request: PromptAdapterRequest) -> bool:
        return self.model_runner.add_prompt_adapter(prompt_adapter_request)

    def remove_prompt_adapter(self, prompt_adapter_id: int) -> bool:
        return self.model_runner.remove_lora(prompt_adapter_id)

    def pin_prompt_adapter(self, prompt_adapter_id: int) -> bool:
        return self.model_runner.pin_prompt_adapter(prompt_adapter_id)

    def list_prompt_adapters(self) -> Set[int]:
        return self.model_runner.list_prompt_adapters()

    @property
    def max_model_len(self) -> int:
        return self.model_config.max_model_len

    @property
    def vocab_size(self) -> int:
        return self.model_runner.vocab_size

    def get_cache_block_size_bytes(self) -> int:
        """Get the size of the KV cache block size in bytes.
        """
        return CacheEngine.get_cache_block_size(self.cache_config,
                                                self.model_config,
                                                self.parallel_config)


def init_worker_distributed_environment(
    parallel_config: ParallelConfig,
    rank: int,
    distributed_init_method: Optional[str] = None,
    local_rank: int = -1,
) -> None:
    """Initialize the distributed environment."""
    set_custom_all_reduce(not parallel_config.disable_custom_all_reduce)

    init_distributed_environment(parallel_config.world_size, rank,
                                 distributed_init_method, local_rank)

    ensure_model_parallel_initialized(parallel_config.tensor_parallel_size,
                                      parallel_config.pipeline_parallel_size)


def _check_if_gpu_supports_dtype(torch_dtype: torch.dtype):
    # Check if the GPU supports the dtype.
    if torch_dtype == torch.bfloat16:  # noqa: SIM102
        if not current_platform.has_device_capability(80):
            capability = current_platform.get_device_capability()
            gpu_name = current_platform.get_device_name()

            if capability is None:
                compute_str = "does not have a compute capability"
            else:
                version_str = capability.as_version_str()
                compute_str = f"has compute capability {version_str}"

            raise ValueError(
                "Bfloat16 is only supported on GPUs with compute capability "
                f"of at least 8.0. Your {gpu_name} GPU {compute_str}. "
                "You can use float16 instead by explicitly setting the"
                "`dtype` flag in CLI, for example: --dtype=half.")


def raise_if_cache_size_invalid(num_gpu_blocks, block_size, is_attention_free,
                                max_model_len) -> None:
    if is_attention_free and num_gpu_blocks != 0:
        raise ValueError("No memory should be allocated for the cache blocks "
                         f"for an attention-free model, but {num_gpu_blocks}"
                         "blocks are allocated.")
    if not is_attention_free and num_gpu_blocks <= 0:
        raise ValueError("No available memory for the cache blocks. "
                         "Try increasing `gpu_memory_utilization` when "
                         "initializing the engine.")
    max_seq_len = block_size * num_gpu_blocks
    if not is_attention_free and max_model_len > max_seq_len:
        raise ValueError(
            f"The model's max seq len ({max_model_len}) "
            "is larger than the maximum number of tokens that can be "
            f"stored in KV cache ({max_seq_len}). Try increasing "
            "`gpu_memory_utilization` or decreasing `max_model_len` when "
            "initializing the engine.")
