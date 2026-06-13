import mne
import numpy as np
from mne.datasets import eegbci

# Load data
fnames = eegbci.load_data(1, [3, 7, 11], path='./data/')  # runs 3,7,11 = motor imagery

raws = [mne.io.read_raw_edf(f, preload=True) for f in fnames]
raw = mne.concatenate_raws(raws)

# Standardize channel names
eegbci.standardize(raw)

# Extract events
events, event_dict = mne.events_from_annotations(raw)
print("Event types found:", event_dict)

# Create epochs — T1=left fist, T2=right fist
epochs = mne.Epochs(raw, events, event_id=dict(T1=2, T2=3),
                    tmin=-0.2, tmax=0.5, baseline=None, preload=True)

X = epochs.get_data()  # shape: (n_epochs, 64_channels, timepoints)
y = epochs.events[:, 2]  # labels

print(f"Data shape: {X.shape}")
print(f"Labels: {np.unique(y)} — count: {len(y)}")
np.save('./X_data.npy', X)
np.save('./y_labels.npy', y)
print(" Data saved! Ready for SNN.")
