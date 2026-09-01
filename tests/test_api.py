"""
Unit tests for FastAPI serving endpoints (/health and /predict).
"""

from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import app, init_sqlite_db, model_store
from data.generate_dataset import generate_telco_churn_data
from src.preprocessing import prepare_data


@pytest.fixture(autouse=True)
def setup_mock_model_and_db(tmp_path):
    """Fixture to mock loaded model and preprocessor for API testing."""
    init_sqlite_db()

    # Fit real preprocessor on sample data
    df = generate_telco_churn_data(num_samples=20, seed=789)
    _, _, preprocessor, _ = prepare_data(df, fit=True)

    # Mock model
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.25, 0.75]])
    mock_model.predict.return_value = np.array([1])

    model_store["model"] = mock_model
    model_store["preprocessor"] = preprocessor
    model_store["version"] = "1.0-test"
    model_store["stage"] = "StagingTest"


client = TestClient(app)


def test_health_endpoint():
    """Test /health GET endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_name"] == "ChurnOps-Model"
    assert data["model_version"] == "1.0-test"


def test_predict_endpoint_success():
    """Test /predict POST endpoint with valid payload."""
    payload = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 65.50,
        "TotalCharges": 786.00,
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "churn_prediction" in data
    assert "churn_label" in data
    assert "churn_probability" in data
    assert data["churn_prediction"] in [0, 1]
    assert data["churn_label"] in ["Yes", "No"]
    assert 0.0 <= data["churn_probability"] <= 1.0


def test_predict_endpoint_malformed_input():
    """Test /predict POST endpoint with invalid data (out of range tenure)."""
    payload = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": -50,  # Invalid negative tenure
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 65.50,
        "TotalCharges": 786.00,
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 422
