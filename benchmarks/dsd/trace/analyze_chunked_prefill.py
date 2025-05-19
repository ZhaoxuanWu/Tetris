import json

chunked_prefill_filename = "/data/lily/vllm-dsd-osdi/benchmarks/dsd/trace/chunked_prefill_20.json"
org_filename = "/data/lily/vllm-dsd-osdi/benchmarks/dsd/trace/wo_chunked_prefill_20.json"


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


print("=======Chunked prefill===========")
analyze(chunked_prefill_filename)
print("=======w/o Chunked prefill===========")
analyze(org_filename)
