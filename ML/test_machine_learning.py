import unittest
from machine_learning import train_model, predict


class TestMLModel(unittest.TestCase):

    def setUp(self):
        """Runs before each test"""
        self.model, self.scaler = train_model()

    def test_prediction_type(self):
        """Check if prediction returns float"""
        sample = [8.3, 41, 6.9, 1.0, 322, 2.5, 37.88, -122.23]
        result = predict(self.model, self.scaler, sample)
        self.assertIsInstance(result, float)

    def test_prediction_value(self):
        """Check if prediction is reasonable"""
        sample = [8.3, 41, 6.9, 1.0, 322, 2.5, 37.88, -122.23]
        result = predict(self.model, self.scaler, sample)
        self.assertGreater(result, 0)


if __name__ == "__main__":
    unittest.main()
    
    import mlflow

def ml_model(data):
    with mlflow.start_run():
        result = sum(data) / len(data)

        # Log inputs
        mlflow.log_param("num_features", len(data))

        # Log output
        mlflow.log_metric("prediction", result)

        return result