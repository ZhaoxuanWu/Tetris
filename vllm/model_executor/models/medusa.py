from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.transformers_utils.configs.medusa import MedusaConfig


class ResidualBlock(nn.Module):

    def __init__(self, hidden_size: int, num_layers: int, num_of_heads: int) -> None:
        super().__init__()
        self.num_of_heads = num_of_heads
        self.layer = nn.Linear(hidden_size, hidden_size * num_of_heads, bias=False)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x =  torch.cat([x for _ in range(self.num_of_heads)], dim=1) + self.act(self.layer(x))
        return x


class Medusa(nn.Module):
    """This class implements the Medusa draft model from the paper: https://arxiv.org/abs/2401.10774
    Reference implementation: https://github.com/FasterDecoding/Medusa
    
    Differences from reference implementation:
    1. Currently this only supports generating proposals from top-1 tokens.
    2. We have an optional token_map which reduces draft vocab to most 
       frequently used tokens to give some additional speed-up by reducing 
       sampling overhead. This is disabled unless the checkpoint file has 
       explicit token_map tensor and config has an optional attribute 
       truncated_vocab_size < vocab_size. To use this technique, one has to find
       the top-k most frequent tokens in target dataset and add that as a tensor
       in the draft checkpoint (using key token_map). Also, the draft config
       needs to have truncated_vocab_size (=k) as an attribute."""

    def __init__(self, config: MedusaConfig, **_) -> None:
        super().__init__()
        self.config = config
        # self.blocks = nn.ModuleList([
        #     ResidualBlock(hidden_size=self.config.hidden_size,
        #                   num_layers=self.config.num_hidden_layers,)
        #     for _ in range(self.config.num_heads)
        # ])
        self.block = ResidualBlock(hidden_size=self.config.hidden_size,
                          num_layers=self.config.num_hidden_layers, num_of_heads=5)
        self.orig_vocab_size = config.vocab_size
        self.truncated_vocab_size = config.truncated_vocab_size
        self.unpadded_vocab_size = self.truncated_vocab_size

        self.lm_heads = nn.ModuleList([
            ParallelLMHead(
                self.unpadded_vocab_size,
                config.hidden_size,
                org_num_embeddings=self.truncated_vocab_size,
                padding_size=DEFAULT_VOCAB_PADDING_SIZE,
            ) for _ in range(5)
        ])

        logit_scale = getattr(config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                self.truncated_vocab_size,
                                                logit_scale)

        # Token map is a idx to token mapping to reduce the vocab size for
        # the draft model. Using smaller vocab size for draft, containing
        # only most frequent tokens reduces the speculation overhead. This
        # doesn't affect the acceptance rate much and thus gives more speed
        # -up. By default, this is disabled and is only used if the EAGLE
        # checkpoint file has token_map tensor.
        self.token_map = None

    def forward(self, hidden_states: torch.Tensor) -> List[torch.Tensor]:
        return self.block(hidden_states).view(-1, 5, hidden_states.shape[-1]).permute(1, 0, 2)

    def compute_logits(
            self, hidden_states: List[torch.Tensor],
            sampling_metadata: SamplingMetadata) -> List[torch.Tensor]:
        logits_lst: List[torch.Tensor] = []

        for hs, lm_head in zip(hidden_states, self.lm_heads):
            _logits = self.logits_processor(lm_head, hs, sampling_metadata)

            if _logits is None:
                # _logits should only be None on rank > 0, in which case
                # it should remain true for every lm_head
                assert len(logits_lst) == 0
                continue

            if self.token_map is None:
                logits_lst.append(_logits)
            else:
                logits_lst.append(-torch.inf * torch.ones(
                    size=(*_logits.shape[:-1], self.orig_vocab_size),
                    device=_logits.device,
                    dtype=_logits.dtype))

                logits_lst[-1][..., self.token_map] = _logits

        # truncate to num_heads
        logits_lst = logits_lst[:self.config.num_heads]

        return logits_lst

    def sample(
        self,
        logits: List[torch.Tensor],
        sampling_metadata: SamplingMetadata,
    ) -> List[SamplerOutput]:
        logits = torch.stack(logits, dim=0).float()
        logprobs = torch.log_softmax(logits, dim=-1)
        token_ids = logits.argmax(-1)  # support only top-1 for now
        probs = torch.softmax(logits, dim=-1)

        token_id_list = []
        token_prob_list = []
        token_logprob_list = []

        for idx, seq_group in enumerate(sampling_metadata.seq_groups):
            token_id_list.append(token_ids[:, seq_group.sample_indices])
            token_prob_list.append(probs[:, seq_group.sample_indices])
            token_logprob_list.append(logprobs[:, seq_group.sample_indices])

        outputs: List[Optional[SamplerOutput]] = []
        for idx in range(len(sampling_metadata.seq_groups)):
            outputs.append(
                SamplerOutput(
                    outputs=None,
                    sampled_token_probs=token_prob_list[idx].squeeze(1),
                    logprobs=token_logprob_list[idx].squeeze(1),
                    sampled_token_ids=token_id_list[idx].squeeze(1),
                ))

        return outputs

    def generate_proposals(
        self,
        previous_hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> List[SamplerOutput]:
        return self.sample(
            logits=self.compute_logits(
                hidden_states=self.forward(previous_hidden_states),
                sampling_metadata=sampling_metadata,
            ),
            sampling_metadata=sampling_metadata,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        params_dict = dict(self.named_parameters())

        weights_map = {}

        for name, loaded_weight in weights:
            name = name.replace("medusa_heads.", "")

            if name == "token_map":
                if self.truncated_vocab_size < self.orig_vocab_size:
                    self.token_map = nn.Parameter(loaded_weight,
                                                  requires_grad=False)
            elif name in params_dict:
                weights_map[name] = loaded_weight

        for name, loaded_weight in weights_map.items():
            if "lm_head" in name and self.token_map is not None and\
                loaded_weight.shape[0] > self.token_map.shape[0]:

                loaded_weight = loaded_weight[self.token_map]

            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            weight_loader(param, loaded_weight)

        if self.token_map is not None:
            self.token_map.to(device=self.lm_heads[0].weight.device)

        assert (self.truncated_vocab_size
                == self.orig_vocab_size) or (self.token_map is not None)
