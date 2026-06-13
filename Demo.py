import numpy as np
import joblib
import torch
import time
from spikingjelly.activation_based import neuron, functional

# Load everything
X = np.load('./X_data.npy')
y = np.load('./y_labels.npy')
clf = joblib.load('./model.pkl')
scaler = joblib.load('./scaler.pkl')

# Same preprocessing
X_norm = np.zeros_like(X)
for i in range(len(X)):
    for c in range(X.shape[1]):
        ch = X[i, c]
        rng = ch.max() - ch.min()
        X_norm[i, c] = (ch - ch.min()) / rng if rng > 0 else 0.0
X_scaled = X_norm * 3.0

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

def extract_features(spike_trains):
    rate = spike_trains.mean(axis=1)
    count = spike_trains.sum(axis=1)
    variance = spike_trains.var(axis=1)
    return np.concatenate([rate, count, variance])

def trigger_device(label, latency_ms, sparsity):
    intent = "LEFT HAND" if label == 2 else "RIGHT HAND"
    print(f"\n{'='*50}")
    print(f" MOTOR INTENT DETECTED: {intent}")
    print(f" Inference Latency: {latency_ms:.3f}ms")
    print(f" Spike Sparsity: {sparsity:.1%} (low power mode)")
    print(f" ASSISTIVE DEVICE TRIGGERED → {intent} MOVEMENT")
    print(f"{'='*50}")

# Run demo on 5 test samples
print("\n NEURONEX'26 — Neuromorphic Motor Intent Detection")
print("Simulating real-time EEG processing for stroke patients...\n")

for i in range(5):
    start = time.time()
    st = encode_epoch(X_scaled[i])
    rate = st.mean(axis=1)
    count = st.sum(axis=1)
    variance = st.var(axis=1)
    feat = np.concatenate([rate, count, variance])
    feat_scaled = scaler.transform([feat])
    pred = clf.predict(feat_scaled)[0]
    latency = (time.time() - start) * 1000
    sparsity = (st == 0).mean()
    trigger_device(pred, latency, sparsity)
    time.sleep(0.5)

print("\n Demo complete. System ready for deployment.")
