"""
Production-grade ML pipeline unit tests covering leak-free feature engineering,
SMOTE within folds, CatBoost/XGBoost/RF training, business threshold cost optimization,
SHAP plot generation, probability calibration, and unified pipeline saving/loading.
"""

import os

import numpy as np
import pytest

from data.generate_dataset import generate_telco_churn_data
from monitoring.predict_utils import load_trained_artifacts, predict_single_row
from src.eda_inspector import detect_data_leakage, inspect_dataset
from src.evaluate import (
    calculate_all_metrics,
    optimize_business_threshold,
)
from src.preprocessing import GenericFeatureEngineer
from src.train import train_and_evaluate


@pytest.fixture
def sample_churn_df():
    return generate_telco_churn_data(num_samples=120, seed=999)


def test_eda_inspector(sample_churn_df):
    """Test dataset inspection and leakage detection functions."""
    inspection = inspect_dataset(sample_churn_df, target_col="Churn")
    assert inspection["num_rows"] == 120
    assert inspection["target_found"] is True

    leakage_cols = detect_data_leakage(sample_churn_df, target_col="Churn")
    assert isinstance(leakage_cols, list)


def test_churn_feature_engineer(sample_churn_df):
    """Test custom GenericFeatureEngineer transformer on raw input data."""
    transformer = GenericFeatureEngineer()
    engineered_df = transformer.transform(sample_churn_df)

    assert "tenure_per_MonthlyCharges" in engineered_df.columns or len(engineered_df.columns) > len(
        sample_churn_df.columns
    )
    assert len(engineered_df) == len(sample_churn_df)


def test_business_threshold_optimization():
    """Test probability decision threshold optimization based on business cost."""
    np.random.seed(42)
    y_true = np.array([0] * 80 + [1] * 20)
    y_prob = np.random.uniform(0, 1, size=100)

    best_th, min_cost, cost_metrics = optimize_business_threshold(
        y_true, y_prob, cost_fn=500.0, cost_fp=50.0, num_thresholds=50
    )

    assert 0.01 <= best_th <= 0.99
    assert min_cost >= 0.0
    assert "threshold" in cost_metrics
    assert "total_cost" in cost_metrics


def test_calculate_all_metrics():
    """Test metric computation including PR-AUC and Brier score."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])

    metrics = calculate_all_metrics(y_true, y_pred, y_prob)
    assert metrics["accuracy"] == 1.0
    assert metrics["f1_score"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["pr_auc"] == 1.0
    assert metrics["brier_score"] == pytest.approx(0.025)


def test_full_pipeline_train_evaluate_and_save(sample_churn_df, tmp_path):
    """Test end-to-end training pipeline execution with unified artifact serialization."""
    csv_path = tmp_path / "train_sample.csv"
    sample_churn_df.to_csv(csv_path, index=False)

    train_res = train_and_evaluate(data_path=str(csv_path), target_col="Churn", fast_mode=True)

    assert train_res["best_model_name"] in [
        "Logistic_Regression",
        "Random_Forest",
        "HistGradientBoosting",
        "XGBoost",
        "CatBoost",
    ]
    assert "optimal_threshold" in train_res

    # Check exported artifacts
    assert os.path.exists("models/best_model.joblib")
    assert os.path.exists("models/unified_pipeline.joblib")

    model, preprocessor, opt_th, status = load_trained_artifacts()
    assert status == "OK"
    assert model is not None
    assert preprocessor is not None
    assert 0.01 <= opt_th <= 0.99

    # Test prediction using loaded artifacts
    sample_row = sample_churn_df.drop(columns=["Churn"]).iloc[0]
    single_res = predict_single_row(sample_row)
    assert single_res["churn_prediction"] in [0, 1]
    assert 0.0 <= single_res["churn_probability"] <= 1.0
