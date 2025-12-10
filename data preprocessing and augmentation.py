import numpy as np
from scipy.signal import butter, lfilter, iirnotch

# 1. Bandpass Filter (0.5–40 Hz)

def bandpass_filter(data, lowcut=0.5, highcut=40, fs=256, order=4):
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data, axis=1)


# 2. Notch Filter (50 Hz)

def notch_filter(data, notch_freq=50, fs=256, quality=30):
    b, a = iirnotch(notch_freq, quality, fs)
    return lfilter(b, a, data, axis=1)


# 3. Segmentation (2-second windows)

def segment_eeg(data, sampling_rate=256, segment_sec=2):
    segment_samples = segment_sec * sampling_rate
    num_segments = data.shape[1] // segment_samples
    segments = []
    for i in range(num_segments):
        seg = data[:, i*segment_samples:(i+1)*segment_samples]
        if seg.shape[1] == segment_samples:
            # Normalize each channel
            seg = (seg - seg.mean(axis=1, keepdims=True)) / (seg.std(axis=1, keepdims=True) + 1e-6)
            segments.append(seg.T.astype(np.float32))
    return np.array(segments)


# 4. Gaussian Noise Augmentation

def apply_gaussian_noise(X, noise_std=0.1, noise_fraction=0.2):
    n_noisy = int(len(X) * noise_fraction)
    idx = np.random.choice(len(X), n_noisy, replace=False)
    X_noisy = X.copy()
    X_noisy[idx] += np.random.normal(0, noise_std, X[idx].shape)
    return X_noisy



filtered = bandpass_filter(raw_data)
filtered = notch_filter(filtered)

segments = segment_eeg(filtered)

#Gaussian noise to 20% of segments
segments_augmented = apply_gaussian_noise(segments, noise_std=0.1, noise_fraction=0.2)
