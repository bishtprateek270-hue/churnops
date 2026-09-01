"""
Unit tests for Optuna HPO, plot artifact isolation, target identifier guards, and classification sanity floors.
"""

import os

import numpy as np
import pandas as pd
import pytest

from src.data_validation import DataValidationError
from src.preprocessing import is_identifier_column
from src.train import train_and_evaluate


def test_optuna_advanced_mode_classification(tmp_path):
    """Verify Optuna HPO (fast_mode=False) completes cleanly without kwarg collisions on classification."""
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "customer_id": [f"CUST_{i:04d}" for i in range(60)],
            "age": np.random.randint(18, 70, 60),
            "tenure": np.random.randint(1, 10, 60),
            "monthly_charges": np.random.uniform(20.0, 100.0, 60),
            "churn": np.random.choice([0, 1], size=60, p=[0.7, 0.3]),
        }
    )
    csv_path = tmp_path / "optuna_cls.csv"
    df.to_csv(csv_path, index=False)

    res = train_and_evaluate(data_path=str(csv_path), target_col="churn", fast_mode=False, n_optuna_trials=2)
    assert res["task_type"] == "classification"
    assert res["best_model_name"] in [
        "Logistic_Regression",
        "Random_Forest",
        "HistGradientBoosting",
        "XGBoost",
        "CatBoost",
    ]
    assert "best_test_metrics" in res


def test_optuna_advanced_mode_regression(tmp_path):
    """Verify Optuna HPO (fast_mode=False) completes cleanly without kwarg collisions on regression."""
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "house_id": range(1000, 1060),
            "square_feet": np.random.randint(800, 3500, 60),
            "bedrooms": np.random.randint(1, 5, 60),
            "price": np.random.uniform(150000.0, 500000.0, 60),
        }
    )
    csv_path = tmp_path / "optuna_reg.csv"
    df.to_csv(csv_path, index=False)

    res = train_and_evaluate(data_path=str(csv_path), target_col="price", fast_mode=False, n_optuna_trials=2)
    assert res["task_type"] == "regression"
    assert res["best_model_name"] in ["Ridge", "Random_Forest", "HistGradientBoosting", "XGBoost", "CatBoost"]
    assert "best_test_metrics" in res


def test_identifier_target_guard(tmp_path):
    """Verify selecting a row identifier as target raises DataValidationError unless explicitly allowed."""
    df = pd.DataFrame(
        {
            "CustomerId": range(10001, 10051),
            "Age": np.random.randint(20, 60, 50),
            "Exited": np.random.choice([0, 1], size=50),
        }
    )
    csv_path = tmp_path / "id_target.csv"
    df.to_csv(csv_path, index=False)

    # 1. Direct identifier check
    assert is_identifier_column(df, "CustomerId") is True

    # 2. Pipeline training should raise DataValidationError without override
    with pytest.raises(DataValidationError, match="unique row identifier"):
        train_and_evaluate(data_path=str(csv_path), target_col="CustomerId", allow_id_target=False, fast_mode=True)

    # 3. Pipeline training with allow_id_target=True should proceed
    res = train_and_evaluate(data_path=str(csv_path), target_col="CustomerId", allow_id_target=True, fast_mode=True)
    assert res["task_type"] is not None


def test_artifact_isolation_across_datasets(tmp_path):
    """Verify training on dataset A then dataset B cleans up and regenerates correct plots without stale leaks."""
    df_a = pd.DataFrame(
        {
            "User_Id": [f"U_{i}" for i in range(50)],
            "feature_a1": np.random.rand(50),
            "feature_a2": np.random.rand(50),
            "target_a": np.random.choice([0, 1], size=50),
        }
    )
    path_a = tmp_path / "dataset_a.csv"
    df_a.to_csv(path_a, index=False)

    train_and_evaluate(data_path=str(path_a), target_col="target_a", fast_mode=True)
    assert os.path.exists("reports/plots/confusion_matrix.png")

    df_b = pd.DataFrame(
        {
            "Item_Id": range(50),
            "feature_b1": np.random.rand(50),
            "feature_b2": np.random.rand(50),
            "target_b": np.random.rand(50) * 100,
        }
    )
    path_b = tmp_path / "dataset_b.csv"
    df_b.to_csv(path_b, index=False)

    res_b = train_and_evaluate(data_path=str(path_b), target_col="target_b", fast_mode=True)
    assert res_b["task_type"] == "regression"
    # Confusion matrix from classification dataset A must be cleaned up / removed for regression dataset B
    assert not os.path.exists("reports/plots/confusion_matrix.png")
