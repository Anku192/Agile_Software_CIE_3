"""Machine Learning model using California Housing dataset."""

import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor


def train_model():
    """Train model and return trained model and scaler."""
    # Load dataset
    data = fetch_california_housing(as_frame=True)
    df = data.frame

    # Features & target
    x = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]

    # Split
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    # Scale
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Train model
    trained_model = RandomForestRegressor(n_estimators=100, random_state=42)
    trained_model.fit(x_train_scaled, y_train)

    # Evaluate
    score = trained_model.score(x_test_scaled, y_test)
    print("Model Score:", score)

    return trained_model, scaler


def predict(model, scaler, input_data):
    """Predict output for given input data."""
    input_scaled = scaler.transform([input_data])
    prediction = model.predict(input_scaled)
    return float(prediction[0])


if __name__ == "__main__":
    model, scaler = train_model()

    sample = [8.3, 41, 6.9, 1.0, 322, 2.5, 37.88, -122.23]
    result = predict(model, scaler, sample)

    print("ML Prediction:", result)

    import pickle

def ml_model(data):
    result = sum(data) / len(data)

    # Save model
    with open("model.pkl", "wb") as f:
        pickle.dump(result, f)

    return result


# run once to generate file
if __name__ == "__main__":
    ml_model([1,2,3,4,5])