from dataclasses import dataclass

import matplotlib.pyplot as plt


@dataclass
class LatencyResult:
    batch_size: int
    acceptance_rate: float
    input_len: int
    output_len: int
    num_spec_tokens: int

    avg_latency: float
    p90_latency: float
    p99_latency: float


def load_results(file_path):
    results = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            parts = line.split(',')
            num_spec_tokens = int(parts[3])
            batch_size = int(parts[4])
            acceptance_rate = float(parts[5])
            input_len = int(parts[6])
            output_len = int(parts[7])
            avg_latency = float(parts[13])
            p90_latency = float(parts[17])
            p99_latency = float(parts[18])
            results.append(
                LatencyResult(batch_size, acceptance_rate, input_len,
                              output_len, num_spec_tokens, avg_latency,
                              p90_latency, p99_latency))
    return results


def load_baseline_results(file_path):
    results = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            parts = line.split(',')
            batch_size = int(parts[2])
            input_len = int(parts[3])
            output_len = int(parts[4])
            avg_latency = float(parts[5])
            p90_latency = float(parts[9])
            p99_latency = float(parts[10])
            results.append(
                LatencyResult(batch_size, -1, input_len, output_len, -1,
                              avg_latency, p90_latency, p99_latency))
    return results


def plot_by_acceptance_rate(acceptance_rate, sd_results, dsd_results,
                            baseline_results, fig_dir):
    sd_results = [
        result for result in sd_results
        if result.acceptance_rate == acceptance_rate
    ]
    dsd_results = [
        result for result in dsd_results
        if result.acceptance_rate == acceptance_rate
    ]
    # baseline_results = [result for result
    # in baseline_results if result.acceptance_rate == acceptance_rate]

    sd_batch_sizes = set([result.batch_size for result in sd_results])
    dsd_batch_sizes = set([result.batch_size for result in dsd_results])
    batch_sizes = sorted(list(sd_batch_sizes & dsd_batch_sizes))
    num_spec_tokens = sorted(
        list(set([result.num_spec_tokens for result in sd_results])))

    for bsz in batch_sizes:
        threshold = [
            result for result in baseline_results if result.batch_size == bsz
        ][0].avg_latency
        print(f"Threshold: {threshold}")
        categories = [str(x) for x in num_spec_tokens] + ['DSD']
        values = [
            result.avg_latency
            for result in sd_results if result.batch_size == bsz
        ] + [
            result.avg_latency
            for result in dsd_results if result.batch_size == bsz
        ]
        plt.figure(figsize=(4, 4))
        plt.axhline(y=threshold,
                    color='red',
                    linestyle='--',
                    label='Threshold')
        plt.bar(categories, values)
        plt.title(f'Batch size: {bsz}, Acceptance rate: {acceptance_rate}',
                  size=12)
        plt.xlabel('Number of speculative tokens', size=12)
        plt.ylabel('Average latency (s)', size=12)
        plt.tight_layout()
        plt.savefig(f'{fig_dir}/latency_{acceptance_rate}_{bsz}.png')
        plt.close()


def plot_speedup_by_batch_size(acceptance_rate, sd_results, dsd_results,
                               baseline_results, fig_dir):
    # Filter results by acceptance rate
    sd_results = [
        result for result in sd_results
        if result.acceptance_rate == acceptance_rate
    ]
    dsd_results = [
        result for result in dsd_results
        if result.acceptance_rate == acceptance_rate
    ]

    # Get common batch sizes and spec tokens
    sd_batch_sizes = set([result.batch_size for result in sd_results])
    dsd_batch_sizes = set([result.batch_size for result in dsd_results])
    batch_sizes = sorted(list(sd_batch_sizes & dsd_batch_sizes))
    batch_sizes = [x for x in batch_sizes if x < 64]
    num_spec_tokens = sorted(
        list(set([result.num_spec_tokens for result in sd_results])))

    plt.figure(figsize=(3, 3))

    # Plot lines for each number of speculative tokens
    for spec_tokens in num_spec_tokens:
        if spec_tokens not in [1, 3, 5, 7]:
            continue
        speedups = []
        for bsz in batch_sizes:
            # Get baseline latency
            baseline_latency = [
                result.avg_latency for result in baseline_results
                if result.batch_size == bsz
            ][0]
            # Get SD latency for this spec token count
            sd_latency = [
                result.avg_latency for result in sd_results
                if result.batch_size == bsz
                and result.num_spec_tokens == spec_tokens
            ][0]
            speedup = baseline_latency / sd_latency
            # print(speedup, baseline_latency, sd_latency)
            speedups.append(speedup)
        plt.plot(batch_sizes, speedups, marker='o', label=f'k={spec_tokens}')

    # Plot DSD results
    dsd_speedups = []
    for bsz in batch_sizes:
        baseline_latency = [
            result.avg_latency for result in baseline_results
            if result.batch_size == bsz
        ][0]
        dsd_latency = [
            result.avg_latency for result in dsd_results
            if result.batch_size == bsz
        ][0]
        speedup = baseline_latency / dsd_latency
        dsd_speedups.append(speedup)
    plt.plot(batch_sizes,
             dsd_speedups,
             marker='^',
             label='DSD',
             linestyle='--')

    # Add baseline reference line
    plt.axhline(y=1.0, color='red', linestyle='--', label='Baseline')
    plt.xscale('log', base=2)
    plt.gca().xaxis.set_major_formatter(plt.ScalarFormatter())
    plt.gca().xaxis.set_major_locator(plt.FixedLocator(batch_sizes))
    plt.title(f'Acceptance Rate: {acceptance_rate}', size=12)
    plt.xlabel('Batch Size', size=12)
    plt.ylabel('Speedup (×)', size=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    # plt.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/speedup_{acceptance_rate}.pdf',
                bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    result_dir = "/data/lily/vllm-dsd-osdi/benchmarks/dsd/results/"
    sd_result_loc = f"{result_dir}7B_160M_Eager_False.csv"
    org_result_loc = f"{result_dir}baseline_7B_160M.csv"
    dsd_result_loc = f"{result_dir}dsd_7B_160M.csv"
    sd_results = load_results(sd_result_loc)
    dsd_results = load_results(dsd_result_loc)
    org_results = load_baseline_results(org_result_loc)
    fig_dir = "/data/lily/vllm-dsd-osdi/benchmarks/dsd/figures/7B/"
    # plot_by_acceptance_rate(0.6, sd_results,
    #                       dsd_results, org_results, fig_dir)
    plot_speedup_by_batch_size(0.7, sd_results, dsd_results, org_results,
                               fig_dir)
    plot_speedup_by_batch_size(0.8, sd_results, dsd_results, org_results,
                               fig_dir)

    sd_result_loc = f"{result_dir}70B_1B_Eager_False.csv"
    org_result_loc = f"{result_dir}baseline_70B_1B.csv"
    dsd_result_loc = f"{result_dir}dsd_70B_1B.csv"
    sd_results = load_results(sd_result_loc)
    dsd_results = load_results(dsd_result_loc)
    org_results = load_baseline_results(org_result_loc)
    fig_dir = "/data/lily/vllm-dsd-osdi/benchmarks/dsd/figures/70B"
    # # plot_by_acceptance_rate(0.8, sd_results, dsd_results,
    #                           org_results, fig_dir)
    plot_speedup_by_batch_size(0.7, sd_results, dsd_results, org_results,
                               fig_dir)
    plot_speedup_by_batch_size(0.8, sd_results, dsd_results, org_results,
                               fig_dir)
