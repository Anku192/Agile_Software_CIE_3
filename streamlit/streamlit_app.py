import streamlit as st
import numpy as np

# -----------------------------
# Dummy Models
# -----------------------------
def ml_model(data):
    return float(np.mean(data))

def dl_model(data):
    return float(np.sum(data) / len(data))

def qml_model(data):
    return float((np.mean(data[:2]) % 1))


# -----------------------------
# UI
# -----------------------------
st.title("🏠 AI House Price Predictor")

model = st.selectbox(
    "Select Model",
    ["Machine Learning", "Deep Learning", "Quantum ML"]
)

input_text = st.text_input(
    "Enter features (comma separated)",
    "8.3,41,6.9,1,322,2.5,37.88,-122.23"
)

if st.button("Predict"):

    try:
        values = [float(x.strip()) for x in input_text.split(",")]

        if model == "Machine Learning":
            result = ml_model(values)

        elif model == "Deep Learning":
            result = dl_model(values)

        elif model == "Quantum ML":
            if len(values) < 2:
                st.error("QML needs at least 2 features")
                st.stop()
            result = qml_model(values)

        st.success(f"Prediction: {result}")

    except:
        st.error("Invalid input format")