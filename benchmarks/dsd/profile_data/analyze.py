import pickle as pk
import matplotlib.pyplot as plt 

_BATCH_SIZE_ALIGNMENT = 32

def _get_graph_batch_size(batch_size: int) -> int:
    """Returns the padded batch size given actual batch size.

    Batch sizes are 1, 2, 4, _BATCH_SIZE_ALIGNMENT,
    2*_BATCH_SIZE_ALIGNMENT, 3*_BATCH_SIZE_ALIGNMENT...
    """
    if batch_size <= 32:
        return 1 << (batch_size - 1).bit_length()
    else:
        return ((batch_size + _BATCH_SIZE_ALIGNMENT - 1) //
                _BATCH_SIZE_ALIGNMENT * _BATCH_SIZE_ALIGNMENT)
        
if __name__ == '__main__':

    mqa_filename = 'mqa_lmsys_vicuna-7b-v1.5_profile_data.pkl'
    batch_expansion_filename = 'batch_expansion_lmsys_vicuna-7b-v1.5_profile_data.pkl'

    with open(mqa_filename, 'rb') as f:
        mqa_data = pk.load(f)
    with open(batch_expansion_filename, 'rb') as f:
        batch_expansion_data = pk.load(f)
    
    seq_lens = [1]
    k = 4
    batch_sizes = [1, 2, 4, 8, 16, 32]
    
    for seq_len in seq_lens:
        mqa_latency = []
        batch_expansion_latency = []
        for batch_size in batch_sizes:
            num_batched_tokens = k * batch_size
            graph_batch_size = _get_graph_batch_size(num_batched_tokens)
            batch_expansion_latency.append(batch_expansion_data[seq_len][graph_batch_size])
            print(mqa_data[seq_len][batch_size].keys())
            mqa_latency.append(mqa_data[seq_len][batch_size][k])
    
        plt.plot(batch_sizes, mqa_latency, label='MQA')
        plt.plot(batch_sizes, batch_expansion_latency, label='Batch Expansion')
        plt.xlabel('Number of requests')
        plt.ylabel('Latency')
        plt.title(f"seq_len={seq_len}_k={k}")
        plt.legend()
        plt.savefig(f"seq_len={seq_len}_k={k}_latency_vs_batch_size.png")