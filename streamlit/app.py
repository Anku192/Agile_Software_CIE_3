from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

# -----------------------------
# Initialize FastAPI app
# -----------------------------
app = FastAPI()

# -----------------------------
# Enable CORS
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Dummy Models (you can replace later)
# -----------------------------
print("Loading models...")

def ml_model(data):
    return float(np.mean(data))

def dl_model(data):
    return float(np.sum(data) / len(data))

def qml_model(data):
    return float((np.mean(data[:2]) % 1))  # uses first 2 features

print("All models loaded successfully!")

# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def home():
    return {"message": "AI Full Stack API is running 🚀"}

# -----------------------------
# ML Prediction
# -----------------------------
@app.post("/predict/ml")
async def predict_ml(data: list[float]):
    print("ML Input:", data)   # debug
    result = ml_model(data)
    return {"model": "ML", "prediction": result}

# -----------------------------
# DL Prediction
# -----------------------------
@app.post("/predict/dl")
async def predict_dl(data: list[float]):
    print("DL Input:", data)
    result = dl_model(data)
    return {"model": "DL", "prediction": result}

# -----------------------------
# QML Prediction
# -----------------------------
@app.post("/predict/qml")
async def predict_qml(data: list[float]):
    print("QML Input:", data)

    if len(data) < 2:
        return {"error": "QML needs at least 2 features"}

    result = qml_model(data)
    return {"model": "QML", "prediction": result}