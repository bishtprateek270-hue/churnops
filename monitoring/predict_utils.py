"""
Helpers for Streamlit train-and-predict workflow with robust reload error recovery and schema validation.
"""

import os
import sys

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import numpy as np
import pandas as pd

import __main__
from src.data_validation import DataValidationError, validate_data
from src.preprocessing import GenericFeatureEngineer, find_target_col, load_preprocessor, prepare_data
from src.train import train_and_evaluate

__main__.GenericFeatureEngineer = GenericFeatureEngineer
__main__.ChurnFeatureEngineer = GenericFeatureEngineer

MODEL_PATH = "models/best_model.joblib"
PREPROCESSOR_PATH = "models/preprocessor.joblib"
UNIFIED_PIPELINE_PATH = "models/unified_pipeline.joblib"
UPLOAD_PATH = "data/raw/user_upload.csv"


def save_uploaded_dataset(df: pd.DataFrame, path: str = UPLOAD_PATH) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return path


def validate_upload(df: pd.DataFrame, target_col: str | None = None) -> tuple[bool, str]:
    try:
        found_target = find_target_col(df, target_col)
        has_target = found_target is not None and found_target in df.columns
        validate_data(df, is_training=has_target, target_col=target_col)
        msg = f"Dataset passed validation. Target column detected: '{found_target}'." if has_target else "Dataset passed validation (inference mode)."
        return True, msg
    except DataValidationError as exc:
        return False, str(exc)


def run_training(
    data_path: str = UPLOAD_PATH,
    target_col: str | None = None,
    fast_mode: bool = True,
    allow_id_target: bool = False,
    progress_callback: object | None = None,
    **kwargs,
) -> dict:
    return train_and_evaluate(
        data_path=data_path,
        target_col=target_col,
        fast_mode=fast_mode,
        allow_id_target=allow_id_target,
        progress_callback=progress_callback,
    )


def load_trained_artifacts() -> tuple[object | None, object | None, float, str]:
    """Safely load unified pipeline artifact with graceful error recovery for outdated or corrupted files."""
    try:
        if os.path.exists(UNIFIED_PIPELINE_PATH):
            pipeline_dict = joblib.load(UNIFIED_PIPELINE_PATH)
            model = pipeline_dict.get("model")
            preprocessor = pipeline_dict.get("preprocessor")
            optimal_threshold = pipeline_dict.get("optimal_threshold", 0.5)
            if model is not None and preprocessor is not None:
                return model, preprocessor, optimal_threshold, "OK"

        if os.path.exists(MODEL_PATH) and os.path.exists(PREPROCESSOR_PATH):
            model = joblib.load(MODEL_PATH)
            preprocessor = load_preprocessor(PREPROCESSOR_PATH)
            return model, preprocessor, 0.5, "OK"
    except (AttributeError, KeyError, ModuleNotFoundError, Exception) as exc:
        print(f"Notice: Outdated/corrupted artifact load failed gracefully: {exc}")
        return None, None, 0.5, f"Saved model artifact is incompatible or outdated ({exc}). Please train a new model."

    return None, None, 0.5, "No trained model found. Upload a dataset and train a model first."


def predict_customers(df: pd.DataFrame, model=None, preprocessor=None, threshold: float | None = None) -> pd.DataFrame:
    opt_th = 0.5
    if model is None or preprocessor is None:
        model, preprocessor, loaded_th, status_msg = load_trained_artifacts()
        if model is None or preprocessor is None:
            raise ValueError(status_msg)
        if threshold is None:
            opt_th = loaded_th
    else:
        if threshold is not None:
            opt_th = threshold

    predict_df = df.copy()
    found_target = getattr(preprocessor, "target_col_", None) or find_target_col(predict_df)
    task_type = getattr(preprocessor, "task_type_", "classification")

    actual = None
    if found_target and found_target in predict_df.columns:
        actual = predict_df[found_target].copy()
        predict_df = predict_df.drop(columns=[found_target])

    validate_data(predict_df, is_training=False)

    X, _, _, _ = prepare_data(predict_df, preprocessor=preprocessor, fit=False)

    results = predict_df.copy()

    if task_type == "classification":
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X)[:, 1]
        else:
            probabilities = model.predict(X)
            if probabilities.ndim > 1:
                probabilities = probabilities[:, 1]

        predictions = np.where(probabilities >= opt_th, 1, 0)

        results["churn_prediction"] = predictions
        results["churn_label"] = ["Yes" if p == 1 else "No" for p in predictions]
        results["churn_probability"] = probabilities.round(4)

        if actual is not None:
            if actual.dtype == object or str(actual.dtype) in ["string", "category", "bool"]:
                actual_binary = np.where(actual.astype(str).str.strip().str.lower().isin(["yes", "true", "1", "churn", "churned"]), 1, 0)
            else:
                actual_num = pd.to_numeric(actual, errors="coerce").fillna(0.0)
                actual_binary = np.where(actual_num.values > 0.5, 1, 0)
            results["actual_churn"] = actual
            results["prediction_correct"] = (predictions == actual_binary)
    else:
        predictions = model.predict(X)
        results["predicted_value"] = predictions.round(4)
        if actual is not None:
            actual_numeric = pd.to_numeric(actual, errors="coerce").fillna(0.0).values
            results["actual_value"] = actual_numeric
            results["prediction_error"] = (predictions - actual_numeric).round(4)

    return results


def predict_single_row(row: pd.Series | dict, model=None, preprocessor=None, threshold: float | None = None) -> dict:
    if isinstance(row, dict):
        df = pd.DataFrame([row])
    elif isinstance(row, pd.Series):
        df = pd.DataFrame([row.to_dict()])
    else:
        df = pd.DataFrame([dict(row)])
    results = predict_customers(df, model=model, preprocessor=preprocessor, threshold=threshold)
    row_result = results.iloc[0]
    return row_result.to_dict()

