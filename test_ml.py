import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from ml.data import process_data
from ml.model import train_model, compute_model_metrics
# TODO: add necessary import

def test_train_model():
    """
    Test if the train_model function returns a RandomForestClassifier
    """
    # Your code here
    X_train = np.random.rand(100, 5)
    y_train = np.random.randint(0,2,100)
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)
    # Model should be a RandomForestClassifier


def test_compute_model_metrics():
    """
    Test if compute_model_metrics returns expected values for known inputs
    """
    # Your code here
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)
    assert precision == 1.0, f"Expected precision 1.0, got {precision}"
    assert recall == 2/3, f"Expected recall 2/3, got {recall}"
    assert fbeta == 0.8, f"Expected F1 score 0.8, got {fbeta}"


def test_process_data():
    """
    Test if process_data returns the expected data types and shapes
    """
    data = pd.DataFrame({
        'age': [25, 30, 35, 40],
        'workclass': ['Private', 'Public', 'Private', 'Public'],
        'education': ['Bachelors', 'Masters', 'Doctorate', 'Bachelors'],
        'salary': ['<=50K', '>50K', '>50K', '<=50K']
    })
    cat_features = ['workclass', 'education']
    X, y, encoder, lb = process_data(
        data, categorical_features=cat_features, label="salary", training=True
    )
    assert isinstance(X, np.ndarray), "X should be a numpy array"
    assert isinstance(y, np.ndarray), "y should be a numpy array"
    assert X.shape[0] == 4, f"Expected 4 samples, got {X.shape[0]}"
    assert X.shape[1] == 6, f"Expected 6 features after one-hot encoding, got {X.shape[1]}"
    assert y.shape[0] == 4, f"Expected 4 labels, got {y.shape[0]}"

