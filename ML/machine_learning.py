# machine_learning.py

import numpy as np
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor


def train_model():
    # Load dataset
    data = fetch_california_housing()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['Price'] = data.target

    # Features & target
    X = df.drop('Price', axis=1)
    y = df['Price']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Train model (best performer from your script)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    return model, scaler


def predict(model, scaler, input_data):
    input_scaled = scaler.transform([input_data])
    prediction = model.predict(input_scaled)
    return float(prediction[0])


if __name__ == "__main__":
    model, scaler = train_model()

    sample = [8.3, 41, 6.9, 1.0, 322, 2.5, 37.88, -122.23]
    result = predict(model, scaler, sample)

    print("ML Prediction:", result)

