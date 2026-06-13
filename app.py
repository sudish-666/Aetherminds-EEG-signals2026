from flask import Flask, render_template, jsonify
import numpy as np
import torch
import joblib
import time
from spikingjelly.activation_based import neuron, functional

app = Flask(__name__)

# Load model
clf = joblib.load('./model.pkl')
scaler = joblib.load('./scaler.pkl')
X = np.load('./X_data.npy')
y = np.load('./y_labels.npy')

# Normalize
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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict/<int:sample_id>')
def predict(sample_id):
    sample_id = sample_id % len(X_scaled)
    start = time.time()
    st = encode_epoch(X_scaled[sample_id])
    rate = st.mean(axis=1)
    count = st.sum(axis=1)
    variance = st.var(axis=1)
    feat = np.concatenate([rate, count, variance])
    feat_scaled = scaler.transform([feat])
    pred = clf.predict(feat_scaled)[0]
    latency = (time.time() - start) * 1000
    sparsity = float((st == 0).mean())

    intent = "LEFT HAND" if pred == 2 else "RIGHT HAND"
    true_label = "LEFT HAND" if y[sample_id] == 2 else "RIGHT HAND"

    # EEG signal for plot (channel 0, first 113 points)
    eeg_signal = X[sample_id][0].tolist()
    spike_data = st[0].tolist()  # spikes for channel 0

    return jsonify({
        'intent': intent,
        'true_label': true_label,
        'correct': intent == true_label,
        'latency': round(latency, 3),
        'sparsity': round(sparsity * 100, 1),
        'eeg_signal': eeg_signal,
        'spike_data': spike_data,
        'sample_id': sample_id
    })

if __name__ == '__main__':
    app.run(debug=True, port=5001)

@app.route('/accuracy')
def get_accuracy():
    # If you have precomputed accuracy, return it.
    # For example:
    return jsonify({'accuracy': 0.778})   # replace with your actual accuracy

@app.route('/total_samples')
def get_total_samples():
    return jsonify({'total': len(X_scaled)})