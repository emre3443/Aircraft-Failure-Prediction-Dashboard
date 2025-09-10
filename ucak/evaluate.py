# evaluate.py
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report

def evaluate_fd(test_path="data/test.csv", model_path="models/lstm_model.h5", scaler_path="models/scaler.pkl"):
    df = pd.read_csv(test_path)
    features = [c for c in df.columns if "sensor" in c]
    target = "failure"

    X = df[features].values
    y = df[target].values

   
    scaler = joblib.load(scaler_path)
    X_scaled = scaler.transform(X)
    X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

    model = load_model(model_path)
    y_pred = (model.predict(X_scaled) > 0.5).astype("int32")

    print(classification_report(y, y_pred))

if __name__ == "__main__":
    evaluate_fd()

