import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
import numpy as np
import pickle

# dummy data
X = np.array([[1], [2], [3], [4]])
y = np.array([2, 4, 6, 8])

with mlflow.start_run():

    model = LinearRegression()
    model.fit(X, y)

    # log params
    mlflow.log_param("model_type", "LinearRegression")

    # log metric
    score = model.score(X, y)
    mlflow.log_metric("score", score)

    # log model
    mlflow.sklearn.log_model(model, "model")

    # save locally (for your Streamlit app)
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

print("Model trained & logged")