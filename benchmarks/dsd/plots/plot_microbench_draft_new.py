import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

model = "llama2-7b"
mode = "draft"
root = "benchmarks/dsd/results/"
input_len = 256
all_batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
COLORS = {1: 'orange', 3: 'green', 5: 'grey', 7: 'blue'}


def load_data(model, mode, bz, input_len, method, acc=None, k=None):
    data_dir = f"{root}/{model}-{mode}"
    if method == 'org':
        filename = f"{data_dir}/{method}_bz={bz}_input-len={input_len}.json"
    elif method == 'dsd':
        filename = f"{data_dir}/{method}_bz={bz}_input-len={input_len}_acc={acc}.json"
    else:
        filename = f"{data_dir}/{method}={k}_bz={bz}_input-len={input_len}_acc={acc}.json"
    if not os.path.exists(filename):
        return {"avg_latency": 0}
    with open(filename, 'r') as f:
        data = json.load(f)
    return data


def load_all(acc_rates):
    data = {}
    for method in ['dsd', 'org', 'vsd']:
        data[method] = {}
        for bz in all_batch_sizes:
            if method == 'org':
                data[method][bz] = load_data(model, mode, bz, input_len,
                                             method)
                continue
            data[method][bz] = {}
            for acc in acc_rates:
                if method == "dsd":
                    data[method][bz][acc] = load_data(model, mode, bz,
                                                      input_len, method, acc)
                    continue
                data[method][bz][acc] = {}
                for k in [1, 3, 5]:
                    data[method][bz][acc][k] = load_data(
                        model, mode, bz, input_len, method, acc, k)
    return data


def plot_seedup_bar(data, acc):
    x = np.arange(len(all_batch_sizes))
    fig, ax = plt.subplots(figsize=(4, 2.5))
    width = 0.15

    for k in [1, 3, 5]:
        vsd_speedups = []
        for bz in all_batch_sizes:
            org = data['org'][bz]
            vsd = data['vsd'][bz][acc][k]
            if vsd['avg_latency'] == 0:
                vsd_speedups.append(0)
                continue
            vsd_speedups.append(org['avg_latency'] / vsd['avg_latency'])

        ax.bar(x + (k - 3) * width / 2, vsd_speedups, width, label=f'k={k}')

    dsd_speedups = []
    for bz in all_batch_sizes:
        org = data['org'][bz]
        dsd = data['dsd'][bz][acc]
        dsd_speedups.append(org['avg_req_latency'] / dsd['avg_req_latency'])

    ax.bar(x + width * 3, dsd_speedups, width, label=f'DSD')

    # Set x-ticks with batch sizes
    ax.set_xticks(x + 0.5 * width)
    ax.set_xticklabels(all_batch_sizes)
    plt.axhline(y=1.0, color='red', linestyle='--', label='w/o SD')

    # Optional: Add x-label
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Speedup')

    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))
    plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
    # plt.legend(ncol=2, fontsize=9)
    plt.savefig(
        f"benchmarks/dsd/figures/h100_bar_7b_draft_speedup_acc={acc}.pdf")
    plt.close()


if __name__ == "__main__":
    acc_rates = [0.5, 0.7, 0.9]
    data = load_all(acc_rates)
    for acc in acc_rates:
        plot_seedup_bar(data, acc)
