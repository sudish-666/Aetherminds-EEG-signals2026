import numpy as np
import torch
from spikingjelly.activation_based import neuron, functional
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import time

# Load data
X = np.load('./X_data.npy')   # (45, 64, 113)
y = np.load('./y_labels.npy')
print(f"Loaded: {X.shape}, Labels: {np.unique(y)}")

# Normalize per epoch per channel (better than global norm)
X_norm = np.zeros_like(X)
for i in range(len(X)):
    for c in range(X.shape[1]):
        ch = X[i, c]
        rng = ch.max() - ch.min()
        if rng > 0:
            X_norm[i, c] = (ch - ch.min()) / rng
        else:
            X_norm[i, c] = 0.0

# Scale up so LIF neurons actually fire
X_scaled = X_norm * 3.0  # key fix — push input above firing threshold

# LIF encoder
lif = neuron.LIFNode(tau=2.0, v_threshold=1.0)

def encode_to_spikes(epoch):
    spike_trains = []
    for channel in epoch:
        functional.reset_net(lif)
        spikes = []
        for t in channel:
            inp = torch.tensor([float(t)])
            spike = lif(inp)
            spikes.append(spike.item())
        spike_trains.append(spikes)
    return np.array(spike_trains)

def extract_features(spike_trains):
    rate = spike_trains.mean(axis=1)
    count = spike_trains.sum(axis=1)
    sparsity = (spike_trains == 0).mean()
    # Add variance of spike timing per channel
    variance = spike_trains.var(axis=1)
    return np.concatenate([rate, count, variance]), sparsity

# Encode all
print("Encoding to spikes...")
features, sparsities = [], []
for i in range(len(X_scaled)):
    st = encode_to_spikes(X_scaled[i])
    f, s = extract_features(st)
    features.append(f)
    sparsities.append(s)
    if i % 10 == 0:
        print(f"  {i+1}/{len(X_scaled)} done...")

features = np.array(features)
avg_sparsity = np.mean(sparsities)
print(f"\n Spike Sparsity: {avg_sparsity:.1%}")

# Scale features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Cross-validated training (better for small dataset of 45 samples)
from sklearn.model_selection import cross_val_score
clf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
cv_scores = cross_val_score(clf, features_scaled, y, cv=5)
print(f"\n Cross-validation accuracy: {cv_scores.mean():.1%} ± {cv_scores.std():.1%}")

# Final train/test
X_train, X_test, y_train, y_test = train_test_split(
    features_scaled, y, test_size=0.2, random_state=42, stratify=y
)
clf.fit(X_train, y_train)

start = time.time()
y_pred = clf.predict(X_test)
latency_ms = (time.time() - start) * 1000 / len(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"\n Motor Intent Detection Results:")
print(f"   Accuracy:  {acc:.1%}")
print(f"   Latency:   {latency_ms:.3f}ms per sample")
print(f"   Sparsity:  {avg_sparsity:.1%} (power efficiency)")
print(f"\n{classification_report(y_test, y_pred, target_names=['Left Fist','Right Fist'])}")

joblib.dump(clf, './model.pkl')
joblib.dump(scaler, './scaler.pkl')
np.save('./metrics.npy', np.array([acc, latency_ms, avg_sparsity]))
print(" Model saved!")
