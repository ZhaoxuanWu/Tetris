from vllm import LLM, SamplingParams
import time
# Sample prompts.
batch_size = 64
output_len = 128
input_len = 1024
target_model = "lmsys/vicuna-7b-v1.5"
draft_model = "eqhylxx/vicuna-160m"

# prompts = ["Hello, my name is"] * batch_size
prompt_ids = [[1] * input_len] * batch_size

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0,
                                 max_tokens=output_len,
                                 ignore_eos=True)

max_num_seqs = 128
gpu_memory_utilization = 0.9
acc_rate = 0.8
num_speculative_tokens = 5

# Create an LLM.
llm = LLM(model=target_model,
          gpu_memory_utilization=gpu_memory_utilization,
          disable_async_output_proc=True,
          max_num_seqs=max_num_seqs)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
start = time.time()
outputs = llm.generate(prompt_token_ids=prompt_ids,
                       sampling_params=sampling_params)
end = time.time()
org_time = end - start
org_tpt = batch_size * output_len / org_time
del llm

# # Batch Expansion
# llm = LLM(model=target_model,
#           speculative_model=draft_model,
#           max_num_seqs=max_num_seqs,
#           num_speculative_tokens=num_speculative_tokens,
#           acceptance_rate=acc_rate,
#           gpu_memory_utilization=gpu_memory_utilization)
# # Generate texts from the prompts. The output is a list of RequestOutput objects
# # that contain the prompt, generated text, and other information.
# start = time.time()
# outputs = llm.generate(prompt_token_ids=prompt_ids,
#                        sampling_params=sampling_params)
# end = time.time()
# batch_expansion_time = end - start
# batch_expansion_tpt = batch_size * output_len / batch_expansion_time
# del llm

# Batch Expansion
llm = LLM(model=target_model,
          speculative_model=draft_model,
          max_num_seqs=max_num_seqs,
          num_speculative_tokens=num_speculative_tokens,
          acceptance_rate=acc_rate,
          gpu_memory_utilization=gpu_memory_utilization,
          force_mqa=True)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
start = time.time()
outputs = llm.generate(prompt_token_ids=prompt_ids,
                       sampling_params=sampling_params)
end = time.time()
mqa_time = end - start
mqa_tpt = batch_size * output_len / mqa_time
del llm

print(f"Original time: {org_time:.2f}s, Throughput: {org_tpt:.2f} tokens/s")
# print(f"Batch expansion time: {batch_expansion_time:.2f}s, Throughput: {batch_expansion_tpt:.2f} tokens/s")
print(
    f"Batch expansion time: {mqa_time:.2f}s, Throughput: {mqa_tpt:.2f} tokens/s"
)
