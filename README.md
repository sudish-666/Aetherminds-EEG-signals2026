# Aetherminds — Neuromorphic Motor Intent Detection

**NeuroNex'26 Hackathon Project**

> A spike-inspired, low-power Brain-Computer Interface (BCI) system that detects motor intent from EEG signals using a Leaky Integrate-and-Fire (LIF) Spiking Neural Network — designed to assist stroke patients in controlling assistive devices.


## Problem Statement

Stroke patients often lose voluntary motor control. Traditional EEG-based BCI systems are computationally expensive and impractical for real-time, low-power deployment. This project applies **neuromorphic computing principles** — mimicking how biological neurons communicate via sparse spikes — to build an efficient, low-power motor intent classifier directly from EEG signals.

---

## How It Works

```
Raw EEG Data (MNE PhysioNet EEGBCI)
           ↓
  Preprocessing & Normalization
           ↓
  LIF Spike Encoding (spikingjelly)
           ↓
  Feature Extraction
  (spike rate · count · variance · sparsity)
           ↓
  Random Forest Classifier (scikit-learn)
           ↓
  Motor Intent → Assistive Device Trigger
  (LEFT HAND / RIGHT HAND)
```

### Neuromorphic Math

**LIF Membrane Update:**
```
U[t+1] = β · U[t] + W · I[t] - V_th · O[t]
```
where `β = e^(-Δt/τ)` is the leak factor, `V_th` is the firing threshold.

**Spike Sparsity:**
```
Sparsity = 1 - (#spikes / #total slots)
```
High sparsity → most synapses stay quiescent → lower dynamic power vs dense neural networks.

---

## Project Structure

```
Aetherminds-EEG-signals2026/
├── SNN.py              # LIF spike encoding + feature extraction + model training
├── Demo.py             # Runs inference on test samples (terminal demo)
├── Visualize.py        # Generates EEG spike raster visualizations
├── app.py              # Flask web API — /predict/<sample_id>
├── templates/
│   └── index.html      # Web dashboard frontend
├── requirements.txt    # Python dependencies
└── LICENSE             # MIT License
```

---

## Setup & Installation

### Prerequisites
- Python 3.10+
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/sudish-666/Aetherminds-EEG-signals2026.git
cd Aetherminds-EEG-signals2026
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Project

> **Important:** Run in this order — SNN.py must run first to generate the trained model and data files.

### Step 1 — Train the SNN Model

```bash
python SNN.py
```

Downloads the MNE EEGBCI motor imagery dataset, encodes EEG epochs into LIF spike trains, extracts features, and trains the classifier. Generates `X_data.npy`, `y_labels.npy`, `model.pkl`, and `scaler.pkl` locally.

### Step 2 — Run Terminal Demo

```bash
python Demo.py
```

Runs inference on 5 test samples and prints motor intent predictions:

```
====================================================
  MOTOR INTENT DETECTED: RIGHT HAND
  Inference Latency: 821.059ms
  Spike Sparsity: 62.2% (low power mode)
  ASSISTIVE DEVICE TRIGGERED → RIGHT HAND MOVEMENT
====================================================
```

### Step 3 — Generate Visualizations

```bash
python Visualize.py
```

Produces EEG spike raster plots comparing Left Fist vs Right Fist motor intent signals across 20 EEG channels.

### Step 4 — Launch Web API

```bash
python app.py
```

Open `http://localhost:5000` in your browser. Use `/predict/<sample_id>` endpoint to get real-time motor intent predictions.

---

## Demo Results

| Sample | Detected Intent | Latency (ms) | Spike Sparsity |
|--------|----------------|-------------|----------------|
| 1 | RIGHT HAND | 821.059 | 62.2% |
| 2 | LEFT HAND | 1148.640 | 56.7% |
| 3 | LEFT HAND | 1094.544 | 55.5% |
| 4 | RIGHT HAND | 1068.699 | 60.2% |
| 5 | RIGHT HAND | 1088.613 | 50.6% |

**Average Spike Sparsity ~57%** — meaning ~57% of neurons remain inactive per inference, significantly reducing dynamic power consumption compared to conventional dense neural networks.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| EEG Dataset | MNE-Python · PhysioNet EEGBCI |
| Spike Encoding | spikingjelly (LIF neurons) |
| Deep Learning | PyTorch |
| Classifier | scikit-learn (Random Forest) |
| Web API | Flask |
| Visualization | matplotlib |
| Data Processing | NumPy · SciPy · pandas |

---

## Key Features

- **Neuromorphic spike encoding** — maps raw EEG amplitude changes to binary spike events, mimicking biological neural communication
- **High sparsity inference** (~50–62%) — enables potential deployment on neuromorphic hardware such as Intel Loihi or IBM TrueNorth
- **Real-time latency** under 1.2 seconds — feasible for assistive device control loops
- **No GPU required** — LIF encoding + Random Forest runs entirely on CPU
- **Web API included** — Flask backend serves live predictions via REST endpoint

---

## Real-World Impact

Designed for stroke rehabilitation: the system detects imagined hand movements from scalp EEG and can trigger prosthetic limbs, robotic exoskeletons, or smart wheelchairs — giving stroke patients greater autonomy without invasive brain surgery.

---

## License

This project is licensed under the [MIT License](LICENSE).

---
