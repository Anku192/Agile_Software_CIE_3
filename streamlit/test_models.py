import unittest
from models import ml_model, dl_model, qml_model


class TestModels(unittest.TestCase):
    """Unit tests for model functions"""

    def setUp(self):
        """Sample input data"""
        self.data = [1, 2, 3, 4]

    def test_ml_model(self):
        """Test ML model (mean)"""
        result = ml_model(self.data)
        self.assertEqual(result, 2.5)

    def test_dl_model(self):
        """Test DL model (average)"""
        result = dl_model(self.data)
        self.assertEqual(result, 2.5)

    def test_qml_model(self):
        """Test QML model (modulo logic)"""
        result = qml_model(self.data)
        expected = (sum(self.data[:2]) / 2) % 1
        self.assertEqual(result, expected)

    def test_qml_invalid_input(self):
        """Test QML with insufficient data"""
        with self.assertRaises(Exception):
            qml_model([1])  # should fail or behave unexpectedly


if __name__ == "__main__":
    unittest.main()