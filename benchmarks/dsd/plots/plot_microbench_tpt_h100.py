import matplotlib.pyplot as plt
import numpy as np


# Prepare data
def load_data(filename):
    data = {}
    with open(filename, "r") as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith("batch_size"):
            continue
        parts = line.strip().split(",")
        acc_rate = float(parts[3])
        num_spec_tokens = int(parts[4])
        org_tpt = float(parts[6])
        batch_expansion_tpt = float(parts[8])
        mqa_tpt = float(parts[10])

        if acc_rate not in data:
            data[acc_rate] = {}
        data[acc_rate][num_spec_tokens] = mqa_tpt / org_tpt
    return data


acc_rates = [0.7, 0.8, 0.9]
tokens = [1, 3, 5, 7, 9, 11]

filename = ""
data = load_data(filename)

# Create plot
fig, ax = plt.subplots(figsize=(5, 2))

# Plot each group
for i, acc in enumerate(acc_rates):
    # Create subplot position
    left = (i + 0.1) * (len(tokens) + 1)
    x = np.arange(len(tokens))

    # Plot bars
    ys = []
    for t in tokens:
        ys.append(data[acc][t])

    bars = ax.bar(x + left,
                  ys,
                  color=[
                      'dodgerblue' if j != len(tokens) - 1 else 'crimson'
                      for j in range(len(tokens))
                  ])

# Customize plot
# plt.title('Llama2-7b Throughput Speedup', pad=20)
# plt.xlabel('num_spec_token')
plt.xticks([])  # Hide x ticks

# Add vertical lines and acc labels
for i in range(len(acc_rates)):
    left = (i + 0.1) * (len(tokens) + 1)
    if i > 0:
        plt.axvline(x=left - 1, color='gray', linestyle='--')
    plt.text(left + 1, 1.62, f'acc={acc_rates[i]}', size=12)

    # Add x-axis labels
    for j, token in enumerate(tokens):
        if token == 11:
            label = 'DSD'
        else:
            label = str(token)
        plt.text(left + j, -0.2, label, ha='center', size=10)

# Add right boundary line
# plt.axvline(x=len(acc_rates) * (len(tokens) + 1) - 1.5, color='gray', linestyle='--')

plt.ylim(0, 1.6)
plt.grid(axis='y')
plt.tight_layout()
plt.ylabel('Speedup (x)', size=12)
# plt.xlabel('num_spec_token', loc)
plt.tight_layout()
plt.savefig(f"/data/lily/vllm-dsd-osdi/benchmarks/dsd/figures/h100_7b_tpt.pdf")
