"""
End-to-end unit tests verifying dataset-agnostic ML pipeline execution
on both Classification datasets (Telco Churn) and Regression datasets (Housing Prices).
"""

import os

import numpy as np
import pandas as pd
import pytest

from data.generate_dataset import generate_telco_churn_data
from monitoring.predict_utils import load_trained_artifacts, predict_customers, predict_single_row
from src.preprocessing import infer_task_type
from src.train import train_and_evaluate


@pytest.fixture
def regression_dataset():
    np.random.seed(42)
    n = 120
    df = pd.DataFrame(
        {
            "house_id": [f"H_{i:04d}" for i in range(n)],
            "square_feet": np.random.uniform(500, 3500, size=n),
            "bedrooms": np.random.randint(1, 6, size=n),
            "location": np.random.choice(["Urban", "Suburban", "Rural"], size=n),
            "has_garage": np.random.choice(["Yes", "No"], size=n),
            "sale_price": np.random.uniform(100000, 800000, size=n),
        }
    )
    return df


def test_infer_task_type(regression_dataset):
    """Test task type inference for classification vs continuous regression."""
    churn_df = generate_telco_churn_data(num_samples=20, seed=123)
    assert infer_task_type(churn_df["Churn"]) == "classification"
    assert infer_task_type(regression_dataset["sale_price"]) == "regression"


def test_classification_end_to_end(tmp_path):
    """Test full training, evaluation, saving, reloading, and prediction on classification dataset."""
    df = generate_telco_churn_data(num_samples=100, seed=777)
    csv_path = tmp_path / "cls_data.csv"
    df.to_csv(csv_path, index=False)

    train_res = train_and_evaluate(data_path=str(csv_path), target_col="Churn", fast_mode=True)
    assert train_res["task_type"] == "classification"
    assert os.path.exists("models/unified_pipeline.joblib")

    model, preprocessor, _opt_th, status = load_trained_artifacts()
    assert status == "OK"
    assert model is not None
    assert preprocessor is not None

    sample_row = df.drop(columns=["Churn"]).iloc[0]
    pred = predict_single_row(sample_row)
    assert "churn_prediction" in pred
    assert "churn_probability" in pred


def test_regression_end_to_end(regression_dataset, tmp_path):
    """Test full training, evaluation, saving, reloading, and prediction on regression dataset."""
    csv_path = tmp_path / "reg_data.csv"
    regression_dataset.to_csv(csv_path, index=False)

    train_res = train_and_evaluate(data_path=str(csv_path), target_col="sale_price", fast_mode=True)
    assert train_res["task_type"] == "regression"
    assert os.path.exists("models/unified_pipeline.joblib")

    model, preprocessor, _opt_th, status = load_trained_artifacts()
    assert status == "OK"
    assert model is not None
    assert preprocessor is not None

    sample_batch = regression_dataset.drop(columns=["sale_price"]).head(5)
    batch_preds = predict_customers(sample_batch)
    assert len(batch_preds) == 5
    assert "predicted_value" in batch_preds.columns


def test_infinities_and_missing_values_end_to_end(tmp_path):
    """Test full training and prediction pipeline on dataset containing inf, -inf, NaNs, None, and string 'nan'."""
    np.random.seed(99)
    n = 100
    df = pd.DataFrame(
        {
            "user_id": [f"U_{i:03d}" for i in range(n)],
            "feature_num_1": np.random.choice([10.0, np.nan, 45.5, 80.0], size=n),
            "feature_num_2": np.random.uniform(1.0, 100.0, size=n),
            "feature_cat_1": np.random.choice(["Alpha", "Beta", "Gamma", "nan", None], size=n),
            "Churn": np.random.choice(["Yes", "No", "Yes", "No", np.nan], size=n),
        }
    )

    # Set some inf/-inf explicitly
    df.loc[3, "feature_num_2"] = np.inf
    df.loc[7, "feature_num_2"] = -np.inf

    csv_path = tmp_path / "inf_data.csv"
    df.to_csv(csv_path, index=False)

    train_res = train_and_evaluate(data_path=str(csv_path), target_col="Churn", fast_mode=True)
    assert train_res["task_type"] == "classification"
    assert os.path.exists("models/unified_pipeline.joblib")

    _model, _preprocessor, _opt_th, status = load_trained_artifacts()
    assert status == "OK"

    batch_preds = predict_customers(df.head(10))
    assert len(batch_preds) == 10
    assert "churn_prediction" in batch_preds.columns
