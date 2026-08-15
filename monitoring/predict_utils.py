"""
Helpers for Streamlit train-and-predict workflow.
"""

import os
import sys

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import pandas as pd

from src.data_validation import REQUIRED_COLUMNS, DataValidationError, validate_data
from src.preprocessing import load_preprocessor, prepare_data
from src.train import train_and_evaluate

MODEL_PATH = "models/best_model.joblib"
PREPROCESSOR_PATH = "models/preprocessor.joblib"
UPLOAD_PATH = "data/raw/user_upload.csv"


def save_uploaded_dataset(df: pd.DataFrame, path: str = UPLOAD_PATH) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return path


def validate_upload(df: pd.DataFrame) -> tuple[bool, str]:
    try:
        has_target = "Churn" in df.columns
        validate_data(df, is_training=has_target)
        return True, "Dataset passed validation."
    except DataValidationError as exc:
        return False, str(exc)


def run_training(data_path: str = UPLOAD_PATH) -> dict:
    return train_and_evaluate(data_path=data_path)


def load_trained_artifacts() -> tuple[object, object]:
    if not os.path.exists(MODEL_PATH) or not os.path.exists(PREPROCESSOR_PATH):
        raise FileNotFoundError("No trained model found. Upload a dataset and train first.")
    model = joblib.load(MODEL_PATH)
    preprocessor = load_preprocessor(PREPROCESSOR_PATH)
    return model, preprocessor


def predict_customers(df: pd.DataFrame, model=None, preprocessor=None) -> pd.DataFrame:
    if model is None or preprocessor is None:
        model, preprocessor = load_trained_artifacts()

    predict_df = df.copy()
    actual = None
    if "Churn" in predict_df.columns:
        actual = predict_df["Churn"].copy()
        predict_df = predict_df.drop(columns=["Churn"])

    feature_df = predict_df.drop(columns=["customerID"], errors="ignore")
    missing = [col for col in REQUIRED_COLUMNS if col not in feature_df.columns]
    if missing:
        raise DataValidationError(f"Missing required columns for prediction: {missing}")

    validate_data(feature_df, is_training=False)

    X, _, _, _ = prepare_data(feature_df, preprocessor=preprocessor, fit=False)
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities > 0.5).astype(int)

    results = predict_df.copy()
    results["churn_prediction"] = predictions
    results["churn_label"] = ["Yes" if p == 1 else "No" for p in predictions]
    results["churn_probability"] = probabilities.round(4)

    if actual is not None:
        if actual.dtype == object:
            actual_binary = (actual.str.lower() == "yes").astype(int)
        else:
            actual_binary = actual.astype(int)
        results["actual_churn"] = actual
        results["prediction_correct"] = (predictions == actual_binary.values)

    return results


def predict_single_row(row: pd.Series, model=None, preprocessor=None) -> dict:
    df = pd.DataFrame([row.drop(labels=["Churn", "customerID"], errors="ignore")])
    results = predict_customers(df, model=model, preprocessor=preprocessor)
    row_result = results.iloc[0]
    return {
        "churn_prediction": int(row_result["churn_prediction"]),
        "churn_label": row_result["churn_label"],
        "churn_probability": float(row_result["churn_probability"]),
    }
