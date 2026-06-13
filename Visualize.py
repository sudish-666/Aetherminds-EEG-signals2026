import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from spikingjelly.activation_based import neuron, functional
import torch

# Load data
X = np.load('./X_data.npy')
y = np.load('./y_labels.npy')

# Normalize + scale same as SNN.py
X_norm = np.zeros_like(X)
for i in range(len(X)):
    for c in range(X.shape[1]):
        ch = X[i, c]
        rng = ch.max() - ch.min()
        X_norm[i, c] = (ch - ch.min()) / rng if rng > 0 else 0.0
X_scaled = X_norm * 3.0

# Encode one epoch to spikes for visualization
lif = neuron.LIFNode(tau=2.0, v_threshold=1.0)

def encode_epoch(epoch):
    spike_trains = []
    for channel in epoch:
        functional.reset_net(lif)
        spikes = []
        for t in channel:
            spike = lif(torch.tensor([float(t)]))
            spikes.append(spike.item())
        spike_trains.append(spikes)
    return np.array(spike_trains)

# Pick one left fist and one right fist epoch
left_idx = np.where(y == 2)[0][0]
right_idx = np.where(y == 3)[0][0]

left_spikes = encode_epoch(X_scaled[left_idx])
right_spikes = encode_epoch(X_scaled[right_idx])

fig, axes = plt.subplots(3, 2, figsize=(16, 12))
fig.suptitle('Neuromorphic Motor Intent Detection — EEG Spike Analysis', 
             fontsize=14, fontweight='bold')

# Plot 1 & 2 — Raw EEG signal (first channel)
for col, (idx, label, color) in enumerate([
    (left_idx, 'Left Fist Intent', 'royalblue'),
    (right_idx, 'Right Fist Intent', 'crimson')
]):
    axes[0, col].plot(X[idx][0], color=color, linewidth=1.2)
    axes[0, col].set_title(f'Raw EEG Signal — {label}')
    axes[0, col].set_xlabel('Timepoints')
    axes[0, col].set_ylabel('Amplitude (µV)')
    axes[0, col].grid(alpha=0.3)

# Plot 3 & 4 — Spike raster (first 20 channels)
for col, (spikes, label) in enumerate([
    (left_spikes, 'Left Fist'), (right_spikes, 'Right Fist')
]):
    for ch in range(20):
        spike_times = np.where(spikes[ch] > 0)[0]
        axes[1, col].scatter(spike_times, [ch]*len(spike_times),
                            marker='|', s=100,
                            color='purple' if col == 0 else 'darkorange')
    axes[1, col].set_title(f'Spike Raster — {label} (20 channels)')
    axes[1, col].set_xlabel('Timepoints')
    axes[1, col].set_ylabel('EEG Channel')
    axes[1, col].grid(alpha=0.2)

# Plot 5 — Spike rate comparison across channels
left_rates = left_spikes.mean(axis=1)
right_rates = right_spikes.mean(axis=1)
x = np.arange(64)
axes[2, 0].bar(x, left_rates, color='royalblue', alpha=0.7, label='Left Fist')
axes[2, 0].bar(x, right_rates, color='crimson', alpha=0.5, label='Right Fist')
axes[2, 0].set_title('Spike Firing Rate per Channel — Left vs Right')
axes[2, 0].set_xlabel('EEG Channel')
axes[2, 0].set_ylabel('Spike Rate')
axes[2, 0].legend()
axes[2, 0].grid(alpha=0.3)

# Plot 6 — Results summary
metrics = ['Accuracy\n77.8%', 'Latency\n1.776ms', 'Sparsity\n56.6%']
values = [77.8, 100 - 1.776, 56.6]  # normalized for display
colors_bar = ['#2ecc71', '#3498db', '#9b59b6']
bars = axes[2, 1].bar(metrics, [77.8, 85, 56.6], color=colors_bar, alpha=0.85, width=0.5)
axes[2, 1].set_title('System Performance Metrics')
axes[2, 1].set_ylabel('Score (%)')
axes[2, 1].set_ylim(0, 100)
axes[2, 1].grid(alpha=0.3, axis='y')
for bar, val in zip(bars, [77.8, 85, 56.6]):
    axes[2, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{val}%', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('./results_visualization.png', dpi=150, bbox_inches='tight')
print(" Visualization saved as results_visualization.png")
plt.show()
