"""Tests pour les modules churnguard.train et churnguard.evaluate."""

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from churnguard.evaluate import compute_metrics
from churnguard.train import train_model


@pytest.fixture
def sample_data():
    X = pd.DataFrame({
        "tenure": [1, 12, 24],
        "MonthlyCharges": [20.0, 50.0, 80.0],
        "TotalCharges": [20.0, 600.0, 1920.0],
        "SeniorCitizen": [0, 0, 1],
        "gender": ["Male", "Female", "Male"],
        "Partner": ["No", "Yes", "No"],
        "Dependents": ["No", "No", "Yes"],
        "PhoneService": ["Yes", "Yes", "No"],
        "MultipleLines": ["No", "Yes", "No"],
        "InternetService": ["DSL", "Fiber optic", "No"],
        "OnlineSecurity": ["No", "Yes", "No"],
        "OnlineBackup": ["No", "No", "Yes"],
        "DeviceProtection": ["No", "Yes", "No"],
        "TechSupport": ["No", "No", "Yes"],
        "StreamingTV": ["No", "Yes", "No"],
        "StreamingMovies": ["No", "No", "Yes"],
        "Contract": ["Month-to-month", "One year", "Two year"],
        "PaperlessBilling": ["Yes", "No", "Yes"],
        "PaymentMethod": ["Electronic check", "Bank transfer", "Credit card"],
    })
    y = pd.Series([0, 1, 0])
    return X, y


def test_train_model_returns_fitted_pipeline(sample_data):
    X, y = sample_data
    result = train_model(X, y, "lr", {})
    assert isinstance(result, Pipeline)
    assert result.predict(X).shape == (3,)


def test_compute_metrics_returns_expected_keys(sample_data):
    X, y = sample_data
    model = train_model(X, y, "lr", {})
    metrics = compute_metrics(model, X, y)
    expected_keys = {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert set(metrics.keys()) == expected_keys
