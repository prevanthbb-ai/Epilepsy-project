import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

def build_baseline_model(input_shape):
    model = Sequential([
        Conv1D(32, 5, activation='relu', padding='same', input_shape=input_shape),
        MaxPooling1D(2),

        Conv1D(64, 5, activation='relu', padding='same'),
        MaxPooling1D(2),

        Conv1D(128, 5, activation='relu', padding='same'),
        MaxPooling1D(2),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model
