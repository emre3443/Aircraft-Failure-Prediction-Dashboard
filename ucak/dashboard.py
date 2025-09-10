# dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import plotly.express as px
import os

# Sayfa ayarları
st.set_page_config(page_title="✈️ Uçak Arıza Tahmin Dashboard", layout="wide")

# Arka plan resmi ve overlay
st.markdown(
    """
    <style>
    body {
        background-image: url("https://images.unsplash.com/photo-1581091012184-07d6d1be4a46?ixlib=rb-4.0.3&auto=format&fit=crop&w=1950&q=80");
        background-size: cover;
        background-attachment: fixed;
    }
    .stApp {
        background-color: rgba(0,0,0,0.5); /* overlay */
    }
    .stTitle, .stHeader, .stMarkdown {
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("✈️ Uçak Arıza Tahmin Dashboard")
st.write("Bu dashboard, uçak sensör verilerini kullanarak arıza olasılıklarını tahmin eder.")
st.markdown("**Created by Emre3443 & Mustafa Emre Gök**")

# Model ve scaler yolları
MODEL_PATH = "models/lstm_model.h5"
SCALER_PATH = "models/scaler.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    st.error("Model veya scaler bulunamadı. Lütfen train.py ile modeli eğit.")
    st.stop()

# Model ve scaler yükle
model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# CSV yükleme
uploaded_file = st.file_uploader("CSV dosyasını yükle", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Yüklenen Veri Örneği")
    st.dataframe(df.head())

    features = [c for c in df.columns if "sensor" in c]
    if len(features) == 0:
        st.error("CSV dosyasında sensör kolonları bulunamadı (sensor_1, sensor_2, ...)")
    else:
        # Normalize et ve tahmin
        X = df[features].values
        X_scaled = scaler.transform(X)
        X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

        y_pred_prob = model.predict(X_scaled)
        y_pred = (y_pred_prob > 0.5).astype(int)

        df["failure_prob"] = y_pred_prob
        df["failure_pred"] = y_pred

        st.subheader("Tahmin Sonuçları")
        st.dataframe(df)

        # Plotly grafiği
        fig = px.bar(df, x=df.index, y="failure_prob",
                     color="failure_pred",
                     color_discrete_map={0: "green", 1: "red"},
                     labels={"x": "Örnek", "failure_prob": "Arıza Olasılığı"},
                     title="Uçak Arıza Olasılıkları (LSTM Tahmini)")

        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)',
                          paper_bgcolor='rgba(0,0,0,0)',
                          font=dict(color="white"))

        st.plotly_chart(fig, use_container_width=True)