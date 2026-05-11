import numpy as np

def extract_features(eeg_data):
    # assume 4 channels: TP9, AF7, AF8, TP10
    data = np.array(eeg_data)

    # simple averages (placeholder for real bands)
    alpha = np.mean(data[:, 1])  # AF7
    beta = np.mean(data[:, 2])   # AF8

    return {
        "alpha": float(alpha),
        "beta": float(beta)
    }