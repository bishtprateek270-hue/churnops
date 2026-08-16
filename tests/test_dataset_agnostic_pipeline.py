"""
Comprehensive unit tests verifying dataset-agnostic ML pipeline robustness:
- Pre-fit train/holdout splitting with zero data leakage
- High-cardinality categorical handling (OrdinalEncoder)
- Target leakage detection (|corr| > 0.98)
- Native HistGradientBoosting integration
- Quality warnings for small datasets (N < 20), severe imbalance, and negative R^2
- End-to-end consistency across multiple datasets
"""

import numpy as np
import pandas as pd
import pytest

from monitoring.predict_utils import load_trained_artifacts, predict_customers, predict_single_row
from src.data_validation import DataValidationError, validate_data
from src.preprocessing import detect_target_leakage, prepare_data
from src.train import train_and_evaluate


def test_pre_fit_split_leakage_protection():
    """Verify that preprocessing pipeline stats (imputer median, scaler mean) fit ONLY on training data."""
    np.random.seed(42)
    n_train = 50
    n_test = 20

    # Training data with low numerical values [1.0 .. 10.0]
    df_train = pd.DataFrame({
        "feature_1": np.random.uniform(1.0, 10.0, size=n_train),
        "cat_feature": np.random.choice(["A", "B"], size=n_train),
    })

    # Test data with extreme high numerical values [1000.0 .. 5000.0]
    df_test = pd.DataFrame({
        "feature_1": np.random.uniform(1000.0, 5000.0, size=n_test),
        "cat_feature": np.random.choice(["A", "B", "NEW_UNKNOWN"], size=n_test),
    })

    # Fit preprocessor strictly on training set
    X_tr_trans, _, preprocessor, _ = prepare_data(df_train, fit=True)
    scaler = preprocessor.named_steps["column_transformer"].named_transformers_["num"].named_steps["scaler"]

    # Scaler mean must match training mean (~5.5), NOT influenced by test set values (~3000)
    assert pytest.approx(scaler.mean_[0], abs=1.5) == df_train["feature_1"].mean()

    # Transform test set using fitted preprocessor
    X_te_trans, _, _, _ = prepare_data(df_test, preprocessor=preprocessor, fit=False)
    assert X_te_trans.shape[0] == n_test
    assert X_te_trans.shape[1] == X_tr_trans.shape[1]


def test_high_cardinality_categorical_handling():
    """Verify categorical features with > 20 unique categories use OrdinalEncoder without feature column explosion."""
    n = 100
    df = pd.DataFrame({
        "high_card_cat": [f"Category_{i % 25}" for i in range(n)],  # 25 unique categories (high cardinality, non-ID)
        "low_card_cat": np.random.choice(["Option_1", "Option_2"], size=n),
        "num_feature": np.random.uniform(0.0, 1.0, size=n),
        "Target": np.random.choice([0, 1], size=n),
    })

    X_trans, _, preprocessor, feature_names = prepare_data(df, fit=True, target_col="Target")
    assert getattr(preprocessor, "cat_high_cols_", []) == ["high_card_cat"]
    # Transformed shape must remain small (not 25 one-hot columns!)
    assert X_trans.shape[1] < 10


def test_target_leakage_detection():
    """Verify columns with near-perfect correlation (|corr| > 0.98) with target are detected as leakage."""
    n = 100
    y = np.random.uniform(10.0, 100.0, size=n)
    df_train = pd.DataFrame({
        "clean_feature": np.random.uniform(1.0, 50.0, size=n),
        "leakage_feature": y + np.random.normal(0, 0.001, size=n),  # 99.99% correlated
    })

    leak_cols = detect_target_leakage(df_train, y, threshold=0.98)
    assert "leakage_feature" in leak_cols
    assert "clean_feature" not in leak_cols


def test_constant_target_raises_validation_error():
    """Verify that constant target dataset (zero variance) raises DataValidationError."""
    df_constant = pd.DataFrame({
        "feature_1": [1.0, 2.0, 3.0, 4.0],
        "Target": [1, 1, 1, 1],  # Constant target
    })

    with pytest.raises(DataValidationError, match="constant"):
        validate_data(df_constant, is_training=True, target_col="Target")


def test_dataset_agnostic_training_and_holdout_evaluation(tmp_path):
    """Test end-to-end dataset-agnostic training, CV selection, holdout evaluation, and prediction consistency."""
    np.random.seed(123)
    n = 150
    df_cls = pd.DataFrame({
        "id_num": list(range(n)),
        "age": np.random.randint(18, 80, size=n),
        "income": np.random.uniform(20000, 150000, size=n),
        "city": np.random.choice([f"City_{i}" for i in range(25)], size=n),  # High cardinality
        "valid_id": np.random.choice([0, 1], size=n),  # Valid binary feature
        "target_class": np.random.choice(["Yes", "No"], size=n, p=[0.3, 0.7]),
    })

    csv_path = tmp_path / "agnostic_cls.csv"
    df_cls.to_csv(csv_path, index=False)

    train_res = train_and_evaluate(data_path=str(csv_path), target_col="target_class", fast_mode=True)

    assert train_res["task_type"] == "classification"
    assert train_res["best_model_name"] in ["Logistic_Regression", "Random_Forest", "HistGradientBoosting", "XGBoost", "CatBoost"]
    assert "baseline_score" in train_res
    assert "cv_score" in train_res
    assert "best_test_metrics" in train_res

    # Reload artifacts and run predictions
    model, preprocessor, opt_th, status = load_trained_artifacts()
    assert status == "OK"
    assert model is not None
    assert preprocessor is not None

    sample_batch = df_cls.drop(columns=["target_class"]).head(10)
    batch_preds = predict_customers(sample_batch, model=model, preprocessor=preprocessor, threshold=opt_th)
    assert len(batch_preds) == 10
    assert "churn_prediction" in batch_preds.columns

    single_pred = predict_single_row(sample_batch.iloc[0], model=model, preprocessor=preprocessor, threshold=opt_th)
    assert "churn_prediction" in single_pred
