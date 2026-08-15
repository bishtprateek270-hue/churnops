"""
Unit tests for arbitrary custom dataset validation, preprocessing, training, and inference.
"""

import numpy as np
import pandas as pd
import pytest

from src.data_validation import validate_data
from src.preprocessing import prepare_data
from src.train import train_and_evaluate
from monitoring.predict_utils import predict_customers, predict_single_row


@pytest.fixture
def custom_dataset():
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "user_id": [f"USR_{i:04d}" for i in range(n)],
        "age": np.random.randint(18, 70, size=n),
        "monthly_income": np.random.uniform(2000.0, 15000.0, size=n),
        "plan_type": np.random.choice(["Basic", "Premium", "Enterprise"], size=n),
        "is_active": np.random.choice(["Yes", "No"], size=n),
        "support_tickets": np.random.randint(0, 10, size=n),
        "target": np.random.choice([0, 1], size=n, p=[0.7, 0.3]),
    })
    return df


def test_custom_dataset_validation(custom_dataset):
    """Test that a dataset with custom column names passes validation."""
    assert validate_data(custom_dataset, is_training=True, target_col="target") is True


def test_custom_dataset_preprocessing(custom_dataset):
    """Test dynamic feature extraction and preprocessing on custom column names."""
    X_trans, y, preprocessor, feature_names = prepare_data(custom_dataset, fit=True, target_col="target")

    assert isinstance(X_trans, np.ndarray)
    assert X_trans.shape[0] == 100
    assert y is not None
    assert len(y) == 100
    assert hasattr(preprocessor, "feature_cols_")
    assert "user_id" not in preprocessor.feature_cols_
    assert "target" not in preprocessor.feature_cols_
    assert "age" in preprocessor.feature_cols_
    assert "monthly_income" in preprocessor.feature_cols_


def test_custom_dataset_training_and_prediction(custom_dataset, tmp_path):
    """Test end-to-end training and prediction on a custom dataset."""
    csv_path = tmp_path / "custom_data.csv"
    custom_dataset.to_csv(csv_path, index=False)

    train_res = train_and_evaluate(data_path=str(csv_path), target_col="target")
    assert train_res["best_model_name"] in ["Logistic_Regression", "Random_Forest", "XGBoost", "CatBoost"]

    # Test batch predictions on new sample rows
    sample_test = custom_dataset.drop(columns=["target"]).head(10)
    batch_preds = predict_customers(sample_test)
    assert len(batch_preds) == 10
    assert "churn_prediction" in batch_preds.columns
    assert "churn_probability" in batch_preds.columns

    # Test single row prediction
    single_row = sample_test.iloc[0]
    single_pred = predict_single_row(single_row)
    assert single_pred["churn_prediction"] in [0, 1]
    assert 0.0 <= single_pred["churn_probability"] <= 1.0


def test_custom_dataset_with_nan_target(custom_dataset):
    """Test that target series with NaNs/non-finite float values are safely handled without casting errors."""
    df_nan_target = custom_dataset.copy()
    df_nan_target.loc[0, "target"] = np.nan
    df_nan_target.loc[5, "target"] = None

    X_trans, y, preprocessor, feature_names = prepare_data(df_nan_target, fit=True, target_col="target")
    assert X_trans.shape[0] == 98
    assert y is not None
    assert len(y) == 98
    assert not np.isnan(y).any()
