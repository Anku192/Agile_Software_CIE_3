import numpy as np
import pandas as pd

import pennylane as qml
from pennylane import numpy as pnp

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# Quantum setup
n_qubits = 2
dev = qml.device('default.qubit', wires=n_qubits)


@qml.qnode(dev)
def quantum_circuit(inputs, weights):
    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)

    qml.CNOT(wires=[0, 1])

    for i in range(n_qubits):
        qml.RY(weights[i], wires=i)

    return qml.expval(qml.PauliZ(0))


def train_model():
    # Load dataset
    data = fetch_california_housing()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['Price'] = data.target

    # Use only 2 features
    X = df[['MedInc', 'AveRooms']].values[:300]
    y = df['Price'].values[:300]

    # Scale
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_scaled = scaler.fit_transform(X)

    y_scaled = (y - y.min()) / (y.max() - y.min())

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )

    # Initialize weights
    np.random.seed(42)
    weights = pnp.array(np.random.randn(n_qubits), requires_grad=True)

    optimizer = qml.GradientDescentOptimizer(stepsize=0.1)

    # ✅ FIXED LOSS FUNCTION
    def mse_loss(weights, X_batch, y_batch):
        preds = []

        for x in X_batch:
            pred = quantum_circuit(x, weights)
            pred = (pred + 1) / 2
            preds.append(pred)

        preds = pnp.array(preds)
        return pnp.mean((preds - y_batch) ** 2)

    # ✅ FIXED TRAINING LOOP
    for _ in range(30):
        weights, _ = optimizer.step_and_cost(
            lambda w: mse_loss(w, X_train[:50], y_train[:50]),
            weights
        )

    return weights, scaler


def predict(weights, scaler, input_data):
    input_scaled = scaler.transform([input_data[:2]])  # only 2 features
    pred = quantum_circuit(input_scaled[0], weights)
    pred = (pred + 1) / 2
    return float(pred)


if __name__ == "__main__":
    weights, scaler = train_model()

    sample = [8.3, 6.9]
    result = predict(weights, scaler, sample)

    print("QML Prediction:", result)