# train.py
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import joblib

def train_fd(data_path="data/train.csv", model_path="models/lstm_model.h5", scaler_path="models/scaler.pkl"):
    # Veri oku
    df = pd.read_csv(data_path)

    # Sensör kolonlarını seç
    features = [c for c in df.columns if "sensor" in c]
    target = "failure"

    X = df[features].values
    y = df[target].values

    # Normalizasyon
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # LSTM input shape -> (samples, timesteps, features)
    X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

    # Model oluştur
    model = Sequential()
    model.add(LSTM(64, input_shape=(X_scaled.shape[1], X_scaled.shape[2]), return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(32))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    # Modeli eğit
    model.fit(X_scaled, y, epochs=10, batch_size=32, validation_split=0.2, verbose=1)

    # Kayıt et
    os.makedirs("models", exist_ok=True)
    model.save(model_path)
    joblib.dump(scaler, scaler_path)

    print(f"✅ Model kaydedildi: {model_path}")
    print(f"✅ Scaler kaydedildi: {scaler_path}")

if __name__ == "__main__":
    train_fd()
