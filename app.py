import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# --- App Title ---
st.title("☕ Coffee Shop Demand Predictor")
st.write("Predict daily sales based on weather and calendar data")

# --- Model Training ---
@st.cache_data
def train_model():
    # Generate synthetic data
    dates = pd.date_range("2023-01-01", periods=180)
    df = pd.DataFrame({
        "temperature": np.clip(np.random.normal(25, 7, 180), 5, 40),
        "rainfall": np.random.poisson(3, 180),
        "weekend": (dates.weekday >= 5).astype(int),
        "sales": np.clip(80 + 30*(dates.weekday >=5) - 10*np.random.poisson(3, 180), 0, 300)
    })
    
    # Train model
    model = RandomForestRegressor()
    model.fit(df[["temperature", "rainfall", "weekend"]], df["sales"])
    return model

# --- User Inputs ---
with st.sidebar:
    st.header("Input Parameters")
    temp = st.slider("Temperature (°C)", 5, 40, 25)
    rain = st.slider("Rainfall (mm)", 0, 10, 2)
    is_weekend = st.checkbox("Weekend")

# --- Prediction ---
if st.button("Predict Sales"):
    model = train_model()
    prediction = model.predict([[temp, rain, int(is_weekend)]])[0]
    st.success(f"Predicted sales: {int(prediction)} cups")
