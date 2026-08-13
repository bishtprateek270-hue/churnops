"""
Training module for ChurnOps pipeline with MLflow tracking and Model Registry.
"""

import os
import sys

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

# Ensure workspace root directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import tempfile

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.data_validation import validate_data
from src.preprocessing import prepare_data, save_preprocessor

# Configure local MLflow experiment
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
EXPERIMENT_NAME = "ChurnOps_Churn_Prediction"
MODEL_NAME = "ChurnOps-Model"


def setup_mlflow():
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"MLflow tracking initialized at {MLFLOW_URI}")


def eval_metrics(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob)
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": roc_auc
    }


def log_plots_to_mlflow(y_true, y_pred, y_prob, model_name: str):
    """Generate and log confusion matrix and ROC curve plots."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title(f"Confusion Matrix - {model_name}")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        cm_path = os.path.join(tmp_dir, "confusion_matrix.png")
        plt.tight_layout()
        plt.savefig(cm_path)
        plt.close()
        mlflow.log_artifact(cm_path, artifact_path="plots")

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_true, y_prob):.4f}")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {model_name}")
        plt.legend(loc="lower right")
        plt.tight_layout()
        roc_path = os.path.join(tmp_dir, "roc_curve.png")
        plt.savefig(roc_path)
        plt.close()
        mlflow.log_artifact(roc_path, artifact_path="plots")


def train_and_evaluate(data_path: str = "data/raw/telco_churn.csv"):
    setup_mlflow()

    print(f"Loading raw data from {data_path}...")
    df = pd.read_csv(data_path)

    # 1. Validate data
    print("Validating dataset schema and data quality...")
    validate_data(df, is_training=True)

    # 2. Preprocess data
    print("Preprocessing features...")
    X, y, preprocessor, _feature_names = prepare_data(df, fit=True)
    save_preprocessor(preprocessor, "models/preprocessor.joblib")

    # Split dataset: 70% train, 15% val, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

    # Save test set for retrain/eval pipeline comparison
    os.makedirs("data/processed", exist_ok=True)
    np.savez("data/processed/test_set.npz", X_test=X_test, y_test=y_test)
    print("Saved test set to data/processed/test_set.npz")

    # Models to train
    models = {
        "Logistic_Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random_Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, eval_metric="logloss")
    }

    best_f1 = -1.0
    best_run_id = None
    best_model_name = None
    best_model_obj = None

    print("\n--- Starting Training Runs ---")
    for name, model in models.items():
        with mlflow.start_run(run_name=name) as run:
            print(f"Training model: {name}...")
            model.fit(X_train, y_train)

            # Validation metrics
            y_val_pred = model.predict(X_val)
            y_val_prob = model.predict_proba(X_val)[:, 1]
            metrics = eval_metrics(y_val, y_val_pred, y_val_prob)

            # Log parameters
            params = model.get_params()
            for k, v in params.items():
                if isinstance(v, (int, float, str, bool)):
                    mlflow.log_param(k, v)

            # Log metrics
            for metric_name, score in metrics.items():
                mlflow.log_metric(f"val_{metric_name}", score)
                print(f"  val_{metric_name}: {score:.4f}")

            # Log test set metrics for baseline reference
            y_test_pred = model.predict(X_test)
            y_test_prob = model.predict_proba(X_test)[:, 1]
            test_metrics = eval_metrics(y_test, y_test_pred, y_test_prob)
            for metric_name, score in test_metrics.items():
                mlflow.log_metric(f"test_{metric_name}", score)

            # Log plots
            log_plots_to_mlflow(y_val, y_val_pred, y_val_prob, name)

            # Log model artifact
            if name == "XGBoost":
                mlflow.xgboost.log_model(model, artifact_path="model")
            else:
                mlflow.sklearn.log_model(model, artifact_path="model")

            # Check best model based on validation F1 score
            if metrics["f1_score"] > best_f1:
                best_f1 = metrics["f1_score"]
                best_run_id = run.info.run_id
                best_model_name = name
                best_model_obj = model

    print(f"\nBest Model: {best_model_name} with Val F1-score: {best_f1:.4f} (Run ID: {best_run_id})")

    # Register best model in MLflow Model Registry and tag as Staging
    model_uri = f"runs:/{best_run_id}/model"
    print(f"Registering model '{MODEL_NAME}' from URI: {model_uri}...")
    registered_model = mlflow.register_model(model_uri, MODEL_NAME)

    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=registered_model.version,
        stage="Staging",
        archive_existing_versions=False
    )
    print(f"Promoted {MODEL_NAME} version {registered_model.version} to stage 'Staging'")

    # Save best model locally as fallback artifact
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model_obj, "models/best_model.joblib")

    return {
        "best_model_name": best_model_name,
        "best_f1": best_f1,
        "best_run_id": best_run_id,
        "version": registered_model.version
    }


if __name__ == "__main__":
    train_and_evaluate()
