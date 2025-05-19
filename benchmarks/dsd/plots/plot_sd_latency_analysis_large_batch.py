import matplotlib.pyplot as plt
import pickle
import numpy as np


def load_data(model, context_len, min_batch_size, max_batch_size):
    data_path = f"/data/lily/vllm-dsd-osdi/{model}.pkl"
    with open(data_path, "rb") as file:  # 'rb' for read binary mode
        data = pickle.load(file)

    filtered_data = {}
    for batch_size in data[context_len]:
        if batch_size >= min_batch_size and batch_size <= max_batch_size:
            if batch_size in [2, 4, 8]:
                continue
            filtered_data[batch_size] = data[context_len][batch_size] * 1000
    return filtered_data


def plot_fit_line(xs, ys, color, label, scatter=True):
    xs, ys = np.array(list(xs)), np.array(list(ys))
    slope, intercept = np.polyfit(xs, ys, 1)
    fit_line = slope * xs + intercept
    if scatter:
        plt.scatter(xs, ys, color="orange")
    plt.plot(xs, fit_line, color=color, linewidth=2, label=f"{label}")
    return slope, intercept


MODEL_TO_NAME = {
    "eqhylxx_vicuna-160m_profile_data": "vicuna-160m",
    "lmsys_vicuna-7b-v1.5_profile_data": "vicuna-7b",
    "meta-llama_Llama-3.1-70B-Instruct_profile_data": "llama-70b",
    "meta-llama_Llama-3.2-1B-Instruct_profile_data": "llama-1b",
}

draft_model = "eqhylxx_vicuna-160m_profile_data"
target_model = "lmsys_vicuna-7b-v1.5_profile_data"
min_batch_size, max_batch_size = 1, 1024
draft_data = load_data(draft_model, 1, min_batch_size, max_batch_size)
target_data = load_data(target_model, 1, min_batch_size, max_batch_size)

draft_target_data = {}
k = 2
for batch_size in draft_data:
    draft_target_data[
        batch_size] = draft_data[batch_size] * k + target_data[batch_size]

plt.figure(figsize=(5, 4))
total_slope, total_intercept = plot_fit_line(draft_target_data.keys(),
                                             draft_target_data.values(),
                                             "mediumpurple",
                                             "Propose (k = 2) + Verify", False)
plot_fit_line(draft_data.keys(), draft_data.values(), "cadetblue",
              "Draft Decoding Time")
slope, intercept = plot_fit_line(target_data.keys(), target_data.values(),
                                 "gray", "Target Decoding Time")

batch_size = 200
maxx = 1024
maxy = 45
org_decoding_latency = slope * batch_size + intercept
new_batch_size = batch_size * (k + 1)
plt.axvline(x=batch_size,
            color="blue",
            linestyle="--",
            ymin=0,
            ymax=org_decoding_latency / maxy)
plt.axhline(y=org_decoding_latency,
            color="blue",
            linestyle="--",
            xmin=0,
            xmax=new_batch_size / maxx)
plt.scatter([batch_size], [org_decoding_latency],
            color="blue",
            marker="^",
            zorder=10)

new_decoding_latency = total_slope * new_batch_size + total_intercept
plt.axvline(x=new_batch_size,
            color="blue",
            linestyle="--",
            ymin=0,
            ymax=new_decoding_latency / maxy)
plt.axhline(y=new_decoding_latency,
            color="blue",
            linestyle="--",
            xmin=0,
            xmax=new_batch_size / maxx)
plt.scatter([new_batch_size], [new_decoding_latency],
            color="blue",
            marker="^",
            zorder=10)

for i in range(1, k + 2):
    y = new_decoding_latency / i
    plt.scatter([new_batch_size], [y],
                s=100,
                color="red",
                marker="*",
                zorder=10)

    suffix = "s" if i > 1 else ""
    plt.annotate(
        f"Generate {i} token{suffix}",  # Text to display
        xy=(new_batch_size, y),  # Point to label
        xytext=(new_batch_size + 10, y + 2),  # Position of the text
        arrowprops=dict(facecolor='black', arrowstyle="->"),  # Arrow style
        fontsize=12)

plt.xlim(left=0, right=maxx)
plt.ylim(bottom=0, top=maxy)
plt.grid()
plt.legend()
plt.xlabel("Number of Batched Tokens", size=12)
plt.ylabel("Batch Latency (ms)", size=12)
plt.tight_layout()
plt.savefig(f"benchmarks/dsd/figures/speedup_analysis_large.pdf")
