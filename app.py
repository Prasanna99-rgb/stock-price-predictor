import streamlit as st
import numpy as np
import pandas as pd
import pickle
import yfinance as yf
import tensorflow as tf

from tensorflow.keras.layers import (
    Dense,
    LSTM,
    Dropout,
    MultiHeadAttention,
    LayerNormalization,
    GlobalAveragePooling1D
)

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="AI Stock Price Predictor",
    page_icon="📈",
    layout="wide"
)

st.title("📈 AI Stock Price Predictor")
st.markdown("Predict the next day's stock closing price using Deep Learning")

# ---------------------------
# STOCKS
# ---------------------------
STOCKS = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Google",
    "AMZN": "Amazon",
    "NVDA": "NVIDIA",
    "TSLA": "Tesla",
    "META": "Meta",
    "BRK-B": "Berkshire Hathaway",
    "JPM": "JPMorgan",
    "V": "Visa"
}

# ---------------------------
# INDICATORS
# ---------------------------
def add_indicators(data):

    data["SMA20"] = data["Close"].rolling(20).mean()

    data["EMA20"] = data["Close"].ewm(span=20).mean()

    delta = data["Close"].diff()

    gain = delta.clip(lower=0).rolling(14).mean()

    loss = -delta.clip(upper=0).rolling(14).mean()

    rs = gain / loss

    data["RSI"] = 100 - (100 / (1 + rs))

    data["MA20"] = data["Close"].rolling(20).mean()

    data["STD"] = data["Close"].rolling(20).std()

    data["Upper"] = data["MA20"] + (2 * data["STD"])

    data["Lower"] = data["MA20"] - (2 * data["STD"])

    data.dropna(inplace=True)

    return data

# ---------------------------
# MODEL ARCHITECTURE
# ---------------------------
def build_model(input_shape):

    inputs = tf.keras.Input(shape=input_shape)

    attention = MultiHeadAttention(
        num_heads=4,
        key_dim=32
    )(inputs, inputs)

    x = LayerNormalization()(inputs + attention)

    x = LSTM(
        64,
        return_sequences=True
    )(x)

    x = Dropout(0.2)(x)

    x = GlobalAveragePooling1D()(x)

    outputs = Dense(1)(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    return model

# ---------------------------
# SIDEBAR
# ---------------------------
st.sidebar.header("Settings")

ticker = st.sidebar.selectbox(
    "Select Stock",
    list(STOCKS.keys())
)

# ---------------------------
# PREDICT BUTTON
# ---------------------------
if st.button("Predict Next Day Price"):

    with st.spinner("Loading model and fetching data..."):

        # Load scaler
        scaler = pickle.load(
            open(f"scalers/{ticker}_scaler.pkl", "rb")
        )

        # Build model
        model = build_model((60, 7))

        # Load weights
        model.load_weights(
            f"models/{ticker}_model.weights.h5"
        )

        # Download latest stock data
        data = yf.download(
            ticker,
            period="6mo",
            interval="1d",
            progress=False
        )

        data.columns = [
            col[0] if isinstance(col, tuple) else col
            for col in data.columns
        ]

        data = add_indicators(data)

        FEATURES = [
            "Close",
            "Volume",
            "SMA20",
            "EMA20",
            "RSI",
            "Upper",
            "Lower"
        ]

        latest = data[FEATURES].tail(60)

        scaled = scaler.transform(latest)

        X = np.array([scaled])

        prediction = model.predict(X, verbose=0)

        close_min = scaler.data_min_[0]
        close_max = scaler.data_max_[0]

        predicted_price = (
            prediction[0][0] *
            (close_max - close_min)
            + close_min
        )

        current_price = float(
            data["Close"].iloc[-1]
        )

    st.success("Prediction Complete!")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Current Price",
            f"${current_price:.2f}"
        )

    with col2:
        st.metric(
            "Predicted Next Day Price",
            f"${predicted_price:.2f}"
        )

    st.subheader("Recent Closing Prices")

    chart_data = pd.DataFrame({
        "Close": data["Close"].tail(90)
    })

    st.line_chart(chart_data)

# ---------------------------
# FOOTER
# ---------------------------
st.markdown("---")
st.markdown(
    "Developed by **Prasanna99-rgb** 🚀"
)
