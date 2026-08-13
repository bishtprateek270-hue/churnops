"""
Evaluation module comparing candidate models against current Production model for promotion decisions.
"""

import os
from typing import Dict, Any, Tuple
import joblib
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score
import mlflow
import mlflow.sklearn
import mlflow.xgboost

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
MODEL_NAME = "ChurnOps-Model"


def load_model_from_registry(stage: str):
    """Load model from MLflow Model Registry by stage (e.g. 'Production' or 'Staging')."""
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = mlflow.tracking.MlflowClient()
    try:
        latest_versions = client.get_latest_versions(MODEL_NAME, stages=[stage])
        if not latest_versions:
            return None, None
        version = latest_versions[0].version
        model_uri = f"models:/{MODEL_NAME}/{stage}"
        print(f"Loading model '{MODEL_NAME}' stage '{stage}' (Version {version})...")
        model = mlflow.pyfunc.load_model(model_uri)
        return model, version
    except Exception as e:
        print(f"Could not load model for stage '{stage}': {e}")
        return None, None


def evaluate_model_metrics(model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """Calculate F1, ROC-AUC, Precision, Recall, Accuracy for a model."""
    y_pred = model.predict(X_test)
    
    # Check if predict returns probabilities or labels
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        # Fallback if pyfunc returns class array or proba
        y_prob = y_pred

    # Ensure y_pred is binary integers
    if y_pred.dtype.kind not in 'iuf':
        y_pred = (y_pred == "Yes").astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    try:
        auc = roc_auc_score(y_test, y_prob)
    except Exception:
        auc = 0.5

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "roc_auc": float(auc)
    }


def compare_and_promote(promote: bool = True) -> Tuple[bool, Dict[str, Any]]:
    """Compare Staging model vs Production model on test dataset and promote if performance is superior."""
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = mlflow.tracking.MlflowClient()

    # Load test set
    test_path = "data/processed/test_set.npz"
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test dataset not found at {test_path}. Please run train.py first.")

    data = np.load(test_path)
    X_test, y_test = data["X_test"], data["y_test"]

    # Load Staging model (candidate)
    staging_model, staging_version = load_model_from_registry("Staging")
    if staging_model is None:
        # Fallback to local best model if registry empty
        if os.path.exists("models/best_model.joblib"):
            staging_model = joblib.load("models/best_model.joblib")
            staging_version = "local-fallback"
        else:
            raise RuntimeError("No candidate/staging model found for evaluation.")

    candidate_metrics = evaluate_model_metrics(staging_model, X_test, y_test)
    print(f"Candidate Model (Version {staging_version}) Metrics: {candidate_metrics}")

    # Load Production model
    prod_model, prod_version = load_model_from_registry("Production")

    if prod_model is None:
        print("No existing Production model found in registry. Candidate automatically approved for Promotion!")
        should_promote = True
        prod_metrics = None
    else:
        prod_metrics = evaluate_model_metrics(prod_model, X_test, y_test)
        print(f"Current Production Model (Version {prod_version}) Metrics: {prod_metrics}")
        
        # Criteria: Candidate F1 score must exceed Production F1 score
        diff = candidate_metrics["f1_score"] - prod_metrics["f1_score"]
        should_promote = diff > 0.001
        print(f"F1 Difference (Candidate - Production): {diff:+.4f}")

    if should_promote and promote and staging_version != "local-fallback":
        print(f"Promoting model version {staging_version} to 'Production'...")
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=staging_version,
            stage="Production",
            archive_existing_versions=True
        )
        print(f"Model version {staging_version} successfully promoted to Production!")
    elif not should_promote:
        print(f"Candidate model version {staging_version} did not beat Production model. Skipping promotion.")

    report = {
        "candidate_version": staging_version,
        "candidate_metrics": candidate_metrics,
        "production_version": prod_version,
        "production_metrics": prod_metrics,
        "promoted": should_promote and promote
    }

    return should_promote, report


if __name__ == "__main__":
    compare_and_promote(promote=True)
