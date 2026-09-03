"""
Evaluation module supporting Classification (Precision, Recall, F1, ROC-AUC, PR-AUC, Brier)
and Regression (MAE, RMSE, R2) metrics, business threshold optimization, calibration, SHAP, and error analysis.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import settings


def calculate_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None
) -> dict[str, float]:
    """Compute comprehensive classification performance metrics."""
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    roc_auc = 0.5
    pr_auc = 0.0
    brier = 1.0

    if y_prob is not None:
        try:
            roc_auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            roc_auc = 0.5

        try:
            pr_auc = float(average_precision_score(y_true, y_prob))
        except Exception:
            pr_auc = 0.0

        try:
            brier = float(brier_score_loss(y_true, y_prob))
        except Exception:
            brier = 1.0

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier_score": brier,
    }


def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute comprehensive regression metrics (MAE, RMSE, R2)."""
    mae = float(np.asarray(mean_absolute_error(y_true, y_pred)).item())
    mse = float(np.asarray(mean_squared_error(y_true, y_pred)).item())
    rmse = float(np.sqrt(mse))
    r2 = float(np.asarray(r2_score(y_true, y_pred)).item())

    return {
        "mae": mae,
        "rmse": rmse,
        "r2_score": r2,
    }


def calculate_all_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None, task_type: str = "classification"
) -> dict[str, float]:
    """Wrapper function to compute all relevant metrics based on task type."""
    if task_type == "classification":
        return calculate_classification_metrics(y_true, y_pred, y_prob)
    else:
        return calculate_regression_metrics(y_true, y_pred)


def optimize_business_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_fn: float = settings.COST_FN,
    cost_fp: float = settings.COST_FP,
    num_thresholds: int = 100,
) -> tuple[float, float, dict]:
    """Optimize probability decision threshold to maximize F1 score while minimizing business cost.

    cost_fn: Cost of false negative (e.g. losing customer)
    cost_fp: Cost of false positive (e.g. unnecessary retention offer)
    """
    thresholds = np.linspace(0.1, 0.9, num_thresholds)
    best_threshold = 0.5
    max_f1 = -1.0
    best_cost = float("inf")
    best_metrics = {}

    for th in thresholds:
        preds = np.where(y_prob >= th, 1, 0)
        cm = confusion_matrix(y_true, preds, labels=[0, 1])
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn, fp, fn, tp = 0, 0, 0, 0

        total_cost = (fn * cost_fn) + (fp * cost_fp)
        f1 = float(f1_score(y_true, preds, zero_division=0))

        if f1 > max_f1 or (f1 == max_f1 and total_cost < best_cost):
            max_f1 = f1
            best_cost = total_cost
            best_threshold = th
            best_metrics = {
                "threshold": float(th),
                "total_cost": float(total_cost),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "precision": float(precision_score(y_true, preds, zero_division=0)),
                "recall": float(recall_score(y_true, preds, zero_division=0)),
                "f1_score": f1,
            }

    return float(best_threshold), float(best_cost), best_metrics


def log_classification_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    model_name: str = "Best_Model",
    output_dir: str = settings.REPORTS_PLOTS_DIR,
) -> dict[str, str]:
    """Generate and save Confusion Matrix, ROC Curve, and PR Curve plots to disk."""
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = {}

    for fname in ["confusion_matrix.png", "roc_curve.png", "pr_curve.png", "calibration_curve.png"]:
        p = os.path.join(output_dir, fname)
        if os.path.exists(p):
            try:
                os.remove(p)
            except Exception as exc:
                print(f"Notice: Stale plot cleanup note: {exc}")

    # 1. Confusion Matrix Plot
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Negative (0)", "Positive (1)"],
        yticklabels=["Negative (0)", "Positive (1)"],
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    saved_paths["confusion_matrix"] = cm_path

    if y_prob is not None:
        # 2. ROC Curve Plot
        try:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc_val = auc(fpr, tpr)
            plt.figure(figsize=(6, 5))
            plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc_val:.3f})")
            plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC Curve - {model_name}")
            plt.legend(loc="lower right")
            plt.tight_layout()
            roc_path = os.path.join(output_dir, "roc_curve.png")
            plt.savefig(roc_path)
            plt.close()
            saved_paths["roc_curve"] = roc_path
        except Exception as e:
            print(f"ROC plot notice: {e}")

        # 3. Precision-Recall Curve Plot
        try:
            prec, rec, _ = precision_recall_curve(y_true, y_prob)
            pr_auc_val = average_precision_score(y_true, y_prob)
            plt.figure(figsize=(6, 5))
            plt.plot(rec, prec, color="blue", lw=2, label=f"PR curve (AUC = {pr_auc_val:.3f})")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title(f"Precision-Recall Curve - {model_name}")
            plt.legend(loc="lower left")
            plt.tight_layout()
            pr_path = os.path.join(output_dir, "pr_curve.png")
            plt.savefig(pr_path)
            plt.close()
            saved_paths["pr_curve"] = pr_path
        except Exception as e:
            print(f"PR plot notice: {e}")

    return saved_paths


def plot_calibration_curve_to_file(
    y_true: np.ndarray, y_prob: np.ndarray, model_name: str, output_path: str = "reports/plots/calibration_curve.png"
) -> str:
    """Generate and save probability calibration curve."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)

    plt.figure(figsize=(6, 5))
    plt.plot(prob_pred, prob_true, "s-", label=model_name)
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title(f"Calibration Curve - {model_name}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def generate_shap_plots(
    model: object, X_sample: np.ndarray, feature_names: list[str], output_dir: str = "reports/plots"
) -> tuple[dict[str, str], str | None]:
    """Generate SHAP summary plot and feature importance chart using memory-safe sampling."""
    os.makedirs(output_dir, exist_ok=True)
    shap_paths = {}
    shap_warning = None

    summary_path = os.path.join(output_dir, "shap_summary.png")
    if os.path.exists(summary_path):
        try:
            os.remove(summary_path)
        except Exception as exc:
            print(f"Notice: Stale SHAP plot cleanup note: {exc}")

    try:
        # Cap sample to 10 rows and background to 5 rows to prevent RAM OOM (Status 137) on 512MB containers
        X_sub = X_sample[: min(10, len(X_sample))]
        predict_fn = getattr(model, "predict_proba", getattr(model, "predict", None))
        if predict_fn:
            background = shap.sample(X_sub, min(5, len(X_sub)))
            explainer = shap.Explainer(predict_fn, background)
            max_ev = 2 * X_sub.shape[1] + 1
            shap_values = explainer(X_sub, max_evals=max_ev)

            plt.figure(figsize=(8, 6))
            if hasattr(shap_values, "values") and shap_values.values.ndim == 3:
                shap.summary_plot(shap_values.values[:, :, 1], X_sub, feature_names=feature_names, show=False)
            else:
                shap.summary_plot(shap_values, X_sub, feature_names=feature_names, show=False)

            plt.tight_layout()
            plt.savefig(summary_path)
            plt.close()
            shap_paths["shap_summary"] = summary_path
    except Exception as e:
        shap_warning = f"SHAP explanation unavailable for this run: {e}"
        print(f"Notice: {shap_warning}")

    return shap_paths, shap_warning


def perform_error_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    X_df: pd.DataFrame | None = None,
    task_type: str = "classification",
) -> dict:
    """Analyze classification False Positives/False Negatives or regression Residual Errors."""
    if task_type == "regression":
        residuals = np.abs(y_true - y_pred)
        return {
            "mean_residual": float(np.mean(residuals)),
            "max_residual": float(np.max(residuals)),
            "std_residual": float(np.std(residuals)),
        }

    fn_mask = (y_true == 1) & (y_pred == 0)
    fp_mask = (y_true == 0) & (y_pred == 1)

    fn_count = int(np.sum(fn_mask))
    fp_count = int(np.sum(fp_mask))

    analysis = {
        "false_negatives_count": fn_count,
        "false_positives_count": fp_count,
        "fn_avg_churn_prob": float(np.mean(y_prob[fn_mask])) if (y_prob is not None and fn_count > 0) else 0.0,
        "fp_avg_churn_prob": float(np.mean(y_prob[fp_mask])) if (y_prob is not None and fp_count > 0) else 0.0,
    }
    return analysis


def _get_model_version_metrics(
    client: mlflow.tracking.MlflowClient, model_name: str, version: str
) -> tuple[dict[str, float], float]:
    """Retrieve run metrics associated with a registered model version."""
    try:
        mv = client.get_model_version(model_name, version)
        if mv.run_id:
            run = client.get_run(mv.run_id)
            metrics = {k: float(v) for k, v in run.data.metrics.items()}
            f1 = metrics.get(
                "val_f1_score", metrics.get("f1_score", metrics.get("val_roc_auc", metrics.get("roc_auc", 0.0)))
            )
            return metrics, f1
    except Exception as exc:
        print(f"Notice: Could not fetch metrics for model version {version}: {exc}")
    return {}, 0.0


def compare_and_promote(promote: bool = True) -> tuple[bool, dict]:
    """Compare candidate Staging model vs current Production model and promote if superior."""
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
    client = mlflow.tracking.MlflowClient()
    model_name = "ChurnOps-Model"

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        staging_versions = client.get_latest_versions(model_name, stages=["Staging"])
    if not staging_versions:
        return False, {"error": "No Staging model found."}

    candidate_version = staging_versions[0].version
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        prod_versions = client.get_latest_versions(model_name, stages=["Production"])
    prod_version = prod_versions[0].version if prod_versions else None

    # Fetch candidate metrics dynamically from MLflow run
    candidate_metrics, cand_f1 = _get_model_version_metrics(client, model_name, candidate_version)

    # Fallback to local unified pipeline artifact if run metrics missing
    if cand_f1 == 0.0 and os.path.exists("models/unified_pipeline.joblib"):
        try:
            import joblib

            unified = joblib.load("models/unified_pipeline.joblib")
            holdout = unified.get("holdout_metrics", {})
            cand_f1 = float(holdout.get("f1_score", holdout.get("r2_score", 0.80)))
            candidate_metrics = holdout
        except Exception as exc:
            print(f"Notice: Fallback metric loading note: {exc}")

    prod_metrics = {}
    prod_f1 = 0.0
    if prod_version:
        prod_metrics, prod_f1 = _get_model_version_metrics(client, model_name, prod_version)

    # Promote if candidate is superior or if no production model exists yet
    should_promote = (prod_version is None) or (cand_f1 >= prod_f1)
    promoted = False

    if promote and should_promote:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            client.transition_model_version_stage(
                name=model_name, version=candidate_version, stage="Production", archive_existing_versions=True
            )
        promoted = True

    report = {
        "candidate_version": candidate_version,
        "production_version": prod_version,
        "promoted": promoted,
        "candidate_metrics": candidate_metrics or {"f1_score": cand_f1},
        "production_metrics": prod_metrics or ({"f1_score": prod_f1} if prod_version else None),
    }
    return promoted, report
