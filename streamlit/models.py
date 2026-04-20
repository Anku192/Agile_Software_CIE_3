"""
Models module for AI House Price Predictor.
Contains simple ML, DL, and QML dummy models.
"""

import numpy as np


def ml_model(data):
    """Returns mean of input data."""
    return float(np.mean(data))


def dl_model(data):
    """Returns average using sum/length."""
    return float(np.sum(data) / len(data))


def qml_model(data):
    """Returns pseudo quantum result using first two values."""
    if len(data) < 2:
        raise ValueError("Need at least 2 values")
    return float((np.mean(data[:2]) % 1))

def qml_model(data):
    if len(data) < 2:
        raise ValueError("Need at least 2 values")
    return float((sum(data[:2]) / 2) % 1)