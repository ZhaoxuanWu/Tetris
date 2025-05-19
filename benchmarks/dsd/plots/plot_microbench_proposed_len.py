import json
import matplotlib.pyplot as plt

tracedir = "benchmarks/dsd/trace/"


def load(filename):
    with open(filename, "r") as f:
        data = json.load(f)
    return data['traces']


def load_all(acc, input_len, max_k, all_batch_sizes):
    data = {}
    for batch_size in all_batch_sizes:
        data[batch_size] = load(
            tracedir +
            f"input={input_len}_{batch_size}_{acc}_True_k={max_k}.json")
    return data


def get_avg_proposed_len(data):
    proposed_lens = []
    for trace in data:
        if trace['type'] == 'Request':
            continue
        if 'proposed_len' not in trace:
            continue
        proposed_lens.append(trace['proposed_len'])
    avg_proposed_len = sum(proposed_lens) / (len(proposed_lens) + 1e-5)
    return avg_proposed_len


if __name__ == "__main__":
    all_batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    acc_rates = [0.5, 0.7, 0.9]
    input_len = 256
    max_k = 7
    plt.figure(figsize=(3, 2.6))
    for acc in acc_rates:
        data = load_all(acc, input_len, max_k, all_batch_sizes)
        proposed_lens = []
        for batch_size in all_batch_sizes:
            proposed_lens.append(get_avg_proposed_len(data[batch_size]))
        plt.plot(all_batch_sizes, proposed_lens, marker='o', label=f"{acc}")

    plt.xscale("log", base=2)
    plt.gca().xaxis.set_major_formatter(plt.ScalarFormatter())
    plt.gca().xaxis.set_major_locator(plt.FixedLocator(all_batch_sizes))
    plt.xlabel("Batch size", fontsize=12)
    plt.ylabel("Avg proposed length", fontsize=12)
    plt.grid(axis='y')
    plt.legend(bbox_to_anchor=(0.5, 1.15), loc='center', ncol=3)
    plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
    plt.savefig("benchmarks/dsd/figures/h100_proposed_len.pdf")
