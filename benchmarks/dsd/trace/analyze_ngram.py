import json
import matplotlib.pyplot as plt

match_9_filename = "/data/lily/vllm-dsd-osdi/benchmarks/dsd/trace/1_None_False_k=None.json"
match_5_filename = "/data/lily/vllm-dsd-osdi/benchmarks/dsd/trace/1_0.7_True_k=8.json"


def load(filename):
    with open(filename, "r") as f:
        data = json.load(f)
    return data['traces']


def get_request_latency(request_traces):
    request_latencies = {}
    for trace in request_traces:
        if trace['end_us'] is not None and trace['start_us'] is not None:
            request_latencies[trace['tid']] = (trace['end_us'] -
                                               trace['start_us']) / 1e6
    return request_latencies


def get_avg_request_latency(request_traces):
    request_latencies = get_request_latency(request_traces)
    return sum(request_latencies.values()) / len(request_latencies)


def get_req_exec_times(step_traces):
    req_exec_time = {}
    for trace in step_traces:
        step_duration = (trace['end_us'] - trace['start_us']) / 1e6
        if 'batched_requests' not in trace:
            # print('No batched_requests in trace', trace)
            continue
        for r in trace['batched_requests']:
            if r not in req_exec_time:
                req_exec_time[r] = 0
            req_exec_time[r] += step_duration
    return req_exec_time


def get_req_wait_times(request_traces, step_traces):
    req_wait_times = {}
    req_latencies = get_request_latency(request_traces)
    req_exec_times = get_req_exec_times(step_traces)
    for r in req_latencies:
        req_wait_times[r] = req_latencies[r] - req_exec_times[r]
    return req_wait_times


def get_avg_req_wait_time(request_traces, step_traces):
    req_wait_times = get_req_wait_times(request_traces, step_traces)
    return sum(req_wait_times.values()) / len(req_wait_times)


def get_avg_steps(step_traces):
    req_step_nums = {}
    print(len(step_traces))
    for trace in step_traces:
        if 'batched_requests' not in trace:
            # print('No batched_requests in trace', trace)
            continue
        for r in trace['batched_requests']:
            if r not in req_step_nums:
                req_step_nums[r] = 0
            req_step_nums[r] += 1
    return sum(req_step_nums.values()) / len(req_step_nums)


def get_avg_step_time(step_traces):
    step_duration = 0
    for trace in step_traces:
        step_duration += (trace['end_us'] - trace['start_us']) / 1e6
    return step_duration / len(step_traces)


def analyze(filename):
    data = load(filename)
    step_traces = [trace for trace in data if trace['type'] == 'Step']
    request_traces = [trace for trace in data if trace['type'] == 'Request']
    avg_req_latency = get_avg_request_latency(request_traces)
    print(f"Average request latency: {avg_req_latency:.2f} seconds")
    avg_req_exec_time = sum(
        get_req_exec_times(step_traces).values()) / len(request_traces)
    print(f"Average request execution time: {avg_req_exec_time:.2f} seconds")
    avg_req_wait_time = get_avg_req_wait_time(request_traces, step_traces)
    print(f"Average request wait time: {avg_req_wait_time:.2f} seconds")
    avg_step_num = get_avg_steps(step_traces)
    print(f"Average steps per request: {avg_step_num:.2f}")
    avg_step_time = get_avg_step_time(step_traces)
    print(f"Average step time: {avg_step_time:.4f} seconds")

    all_latencies = get_request_latency(request_traces).values()
    plt.hist(
        all_latencies,
        bins=30,  # Number of bins
        density=False,  # If True, show probability density
        alpha=0.7,  # Transparency
        color='skyblue',  # Color of bars
        edgecolor='black'  # Color of bar edges
    )
    plt.savefig(f"{filename}_request_latency.png")
    plt.close()


print("=======0.9===========")
analyze(match_9_filename)
print("=======0.5===========")
analyze(match_5_filename)
