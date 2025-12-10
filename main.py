
# Epilepsy Seizure Detection



import os
import numpy as np
import mne
from scipy.signal import butter, lfilter, iirnotch
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, ReLU, MaxPooling1D, Dropout,
    Bidirectional, LSTM, Dense, GlobalAveragePooling1D, Multiply, Permute
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt


# Parameters

DATA_FOLDER = r"C:\Users\Prakash\Desktop\REVANTH PROJECTS\CHB Dataset"
SEGMENT_SEC = 2
SAMPLING_RATE = 256
NUM_CHANNELS = 23
DROPOUT_RATE = 0.6
np.random.seed(42)
tf.random.set_seed(42)


# Filtering

def bandpass_filter(data, lowcut=0.5, highcut=40, fs=256, order=5):
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data, axis=1)

def notch_filter(data, notch_freq=50, fs=256, quality=30):
    b, a = iirnotch(notch_freq, quality, fs)
    return lfilter(b, a, data, axis=1)

#EEG Loading

def load_eeg(folder, segment_sec=5, sampling_rate=256, num_channels=23):
    X, y = [], []
    patients = sorted([p for p in os.listdir(folder) if p.lower().startswith("chb")])
    for patient in patients:
        patient_path = os.path.join(folder, patient)
        files = [f for f in os.listdir(patient_path) if f.endswith(".edf")]

        for file in files:
            filepath = os.path.join(patient_path, file)
            try:
                raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
            except Exception as e:
                print(f"⚠️ Skipped {file}: {e}")
                continue

            data = raw.get_data()[:num_channels]
            if data is None or data.size == 0:
                continue

            data = bandpass_filter(data)
            data = notch_filter(data)

            segment_samples = segment_sec * sampling_rate
            num_segments = data.shape[1] // segment_samples
            label = 1 if "seizure" in file.lower() or "+" in file else 0

            for i in range(num_segments):
                segment = data[:, i*segment_samples:(i+1)*segment_samples]
                if segment.shape[1] != segment_samples:
                    continue
                segment = (segment - segment.mean(axis=1, keepdims=True)) / \
                          (segment.std(axis=1, keepdims=True) + 1e-6)
                if np.any(np.isnan(segment)):
                    continue
                X.append(segment.T.astype(np.float32))
                y.append(int(label))

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    print(f"✅ Raw Dataset: X={X.shape}, y={y.shape}")
    return X, y


# Synthetic seizure augmentation

def augment_seizures(X, y, augment_factor=1):
    seizure_data = X[y==1]
    if len(seizure_data) == 0:
        print("⚠️ No real seizures found. Generating synthetic seizures...")
        # pick random normal segments to synthesize seizures
        normal_data = X[y==0]
        n_aug = max(10, int(len(normal_data)*0.08))
        syn_seizures = normal_data[:n_aug] + np.random.normal(0, 0.5, normal_data[:n_aug].shape)
        X = np.vstack([X, syn_seizures])
        y = np.hstack([y, np.ones(len(syn_seizures), dtype=np.int32)])
        print(f"✅ Added {len(syn_seizures)} synthetic seizures.")
        return X, y

    augmented = []
    for seg in seizure_data:
        for _ in range(augment_factor):
            noise = np.random.normal(0, 0.1, seg.shape)
            augmented.append(seg + noise)
    if augmented:
        X = np.vstack([X, np.array(augmented, dtype=np.float32)])
        y = np.hstack([y, np.ones(len(augmented), dtype=np.int32)])
        print(f"✅ Added {len(augmented)} synthetic seizures.")
    return X, y


# Attention Layer

from tensorflow.keras.layers import Lambda
import tensorflow.keras.backend as K

def attention_block(inputs):
    # Compute attention scores
    scores = Dense(1, activation='tanh')(inputs)        # shape: (batch, timesteps, 1)
    scores = Lambda(lambda x: K.softmax(x, axis=1))(scores)  # softmax over timesteps
    # Multiply attention weights
    weighted = Multiply()([inputs, scores])            # shape: (batch, timesteps, features)
    # Aggregate over time
    output = GlobalAveragePooling1D()(weighted)        # shape: (batch, features)
    return output



# Model
 
def build_model(input_shape):
    inp = Input(shape=input_shape)

    # CNN
    x = Conv1D(32, 3, padding='same')(inp)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling1D(2)(x)

    x = Conv1D(64, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(DROPOUT_RATE)(x)

    # BiLSTM
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Dropout(DROPOUT_RATE)(x)

    # Attention
    x = attention_block(x)

    # Dense
    x = Dense(32, activation='relu')(x)
    out = Dense(1, activation='sigmoid')(x)

    model = Model(inp, out)
    return model


# Main

X, y = load_eeg(DATA_FOLDER, SEGMENT_SEC, SAMPLING_RATE)
X, y = augment_seizures(X, y, augment_factor=1)

# Train/Val/Test Split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# Class weights
from sklearn.utils import class_weight
weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(weights))
print("Class Weights:", class_weights)


# Compile Model

model = build_model(X_train.shape[1:])
model.compile(optimizer=Adam(1e-5), loss='binary_crossentropy', metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)


# Train Model

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=5,
    batch_size=32,
    class_weight=class_weights,
    callbacks=[early_stop],
    verbose=1
)


# Evaluation

loss, acc = model.evaluate(X_test, y_test, verbose=0)
y_pred_prob = model.predict(X_test, verbose=0)
y_pred = (y_pred_prob >= 0.5).astype(int)

cm = confusion_matrix(y_test, y_pred)
TN, FP, FN, TP = cm.ravel() if cm.size == 4 else (0,0,0,0)
sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
auc_score = roc_auc_score(y_test, y_pred_prob) if len(np.unique(y_test))>1 else 0

print("\n========== Final Test Results ==========")
print(f"Accuracy: {acc:.3f}")
print(f"Sensitivity: {sensitivity:.3f}")
print(f"Specificity: {specificity:.3f}")
print(f"AUC: {auc_score:.3f}")
print("Confusion Matrix:\n", cm)


# Visualization

import seaborn as sns
from sklearn.metrics import roc_curve

#Accuracy-
plt.figure(figsize=(6,4))
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title("Accuracy Curve")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

#Loss
plt.figure(figsize=(6,4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

#ROC
plt.figure(figsize=(6,4))
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.plot(fpr, tpr, label=f"ROC curve (AUC={auc:.3f})")
plt.plot([0,1],[0,1],'k--')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

# Example: visualize attention for the first test segment; i need to work on this later the attention map isnt working I removed the entire block lmao..
sample = X_test[0]
attention_weights = get_attention_weights(model, sample)

plt.figure(figsize=(12,4))
sns.heatmap(attention_weights.T, cmap='viridis', cbar=True)
plt.title("Attention Heatmap for 1 EEG Segment")
plt.xlabel("Time Steps")
plt.ylabel("Channels / Features")
plt.show()
