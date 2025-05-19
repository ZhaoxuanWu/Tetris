import json
import matplotlib.pyplot as plt

model = "llama2-7b"
mode = "ngram"
root =  "/data/lily/vllm-dsd-osdi/benchmarks/dsd/results/"
input_len = 256
all_batch_sizes = [1, 2, 4, 8, 16, 32, 64]

COLORS = {
    1: 'orange',
    3: 'green',
    5: 'grey',
    7: 'purple'
}

def load_data(model, mode, bz, input_len, method, acc=None, match=None, k=None):
    data_dir = f"{root}/{model}-{mode}"
    if method == 'org':
        filename = f"{data_dir}/{method}_bz={bz}_input-len={input_len}.json"
    elif method == 'dsd':
        filename = f"{data_dir}/{method}_bz={bz}_input-len={input_len}_acc={acc}_match={match}.json"
    else:
        filename = f"{data_dir}/{method}={k}_bz={bz}_input-len={input_len}_acc={acc}_match={match}.json"
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

def load_all(accs, match):
    data = {}
    for method in ['dsd', 'org', 'vsd']:
        data[method] = {}
        for bz in all_batch_sizes:
            if method == 'org':
                data[method][bz] = load_data(model, mode, bz, input_len, method)
                continue
            data[method][bz] = {}
            for acc in accs:
                if method == "dsd":
                    data[method][bz][acc] = load_data(model, mode, bz, input_len, method, acc, match)
                    continue
                data[method][bz][acc] = {}
                for k in [1, 3, 5, 7]:
                    data[method][bz][acc][k] = load_data(model, mode, bz, input_len, method, acc, match, k)
    return data
       
def plot_speedup_acc(data, acc, match):
    plt.figure(figsize=(2.6, 2.6))
    speedups = []
    for bz in all_batch_sizes:
        org = data['org'][bz]
        dsd = data['dsd'][bz][acc]
        speedups.append(org['avg_req_latency'] / dsd['avg_req_latency'])
    plt.plot(all_batch_sizes, speedups, label=f"DSD", marker='o', zorder=10, markersize=5)
    
    # Plot vsd
    for k in [1, 3, 5, 7]:
        vsd_speedups = []
        for bz in all_batch_sizes:
            org = data['org'][bz]
            vsd = data['vsd'][bz][acc][k]
            vsd_speedups.append(org['avg_req_latency'] / vsd['avg_req_latency'])
        plt.plot(all_batch_sizes, vsd_speedups, label=f"k={k}", 
                 linestyle='--', marker='x', markersize=5, color=COLORS[k])
    plt.axhline(y=1.0, color='red', linestyle='--', label='w/o SD')
    plt.xscale('log', base=2)
    plt.gca().xaxis.set_major_formatter(plt.ScalarFormatter())
    plt.gca().xaxis.set_major_locator(plt.FixedLocator(all_batch_sizes))
    # plt.ylabel("Speedup")
    plt.xlabel("Batch size")
    plt.legend()
    plt.grid()
    plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
    plt.savefig(f"/data/lily/vllm-dsd-osdi/benchmarks/dsd/figures/7b_ngram_speedup_acc={acc}_match={match}.pdf")
    plt.close()
    
if __name__ == "__main__":
    accs = [0.7, 0.8, 0.9]
    for match in [0.5, 0.9]:
        data = load_all(accs, match)
        for acc in accs:
            plot_speedup_acc(data, acc, match)


            
            