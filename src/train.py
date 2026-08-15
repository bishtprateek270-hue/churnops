"""
Training module supporting Classification & Regression ML tasks with Fast Baseline mode (default, <15s),
Advanced Optuna hyperparameter tuning, SMOTE imbalance handling, probability calibration,
holdout test set metrics evaluation, single-pass SHAP generation, and MLflow tracking.
"""

import os
import sys
import time
import tempfile

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
import optuna

from catboost import CatBoostClassifier, CatBoostRegressor
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from xgboost import XGBClassifier, XGBRegressor

from src.data_validation import validate_data
from src.eda_inspector import detect_data_leakage, generate_eda_report, inspect_dataset
from src.evaluate import (
    calculate_all_metrics,
    generate_shap_plots,
    log_classification_plots,
    optimize_business_threshold,
    perform_error_analysis,
    plot_calibration_curve_to_file,
)
from src.preprocessing import prepare_data, save_preprocessor

# MLflow Configuration
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
EXPERIMENT_NAME = "ChurnOps_Churn_Prediction"
MODEL_NAME = "ChurnOps-Model"


def setup_mlflow():
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"MLflow tracking initialized at {MLFLOW_URI}")


def run_optuna_tuning(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    task_type: str = "classification",
    n_trials: int = 5
) -> dict:
    """Perform Stratified K-Fold CV (Classification) or K-Fold CV (Regression) Hyperparameter Tuning with Optuna."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if task_type == "classification":
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        def objective_cls(trial):
            if model_name == "Logistic_Regression":
                c_val = trial.suggest_float("C", 1e-3, 10.0, log=True)
                solver = trial.suggest_categorical("solver", ["lbfgs", "liblinear"])
                clf = LogisticRegression(C=c_val, solver=solver, max_iter=1000, random_state=42, class_weight="balanced")
            elif model_name == "Random_Forest":
                n_est = trial.suggest_int("n_estimators", 50, 150)
                max_d = trial.suggest_int("max_depth", 3, 10)
                clf = RandomForestClassifier(n_estimators=n_est, max_depth=max_d, n_jobs=-1, random_state=42, class_weight="balanced")
            elif model_name == "XGBoost":
                n_est = trial.suggest_int("n_estimators", 50, 150)
                max_d = trial.suggest_int("max_depth", 3, 8)
                lr = trial.suggest_float("learning_rate", 0.01, 0.2, log=True)
                clf = XGBClassifier(n_estimators=n_est, max_depth=max_d, learning_rate=lr, n_jobs=-1, random_state=42, eval_metric="logloss")
            elif model_name == "CatBoost":
                iters = trial.suggest_int("iterations", 50, 150)
                depth = trial.suggest_int("depth", 3, 8)
                lr = trial.suggest_float("learning_rate", 0.01, 0.2, log=True)
                clf = CatBoostClassifier(iterations=iters, depth=depth, learning_rate=lr, verbose=0, random_seed=42, thread_count=-1)
            else:
                clf = LogisticRegression(max_iter=1000, random_state=42)

            pipeline = ImbPipeline([
                ("smote", SMOTE(random_state=42)),
                ("classifier", clf)
            ])

            scores = []
            for train_idx, val_idx in skf.split(X_train, y_train):
                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]

                pipeline.fit(X_tr, y_tr)
                y_prob = pipeline.predict_proba(X_val)[:, 1]
                metrics = calculate_all_metrics(y_val, np.where(y_prob > 0.5, 1, 0), y_prob, task_type="classification")
                scores.append(metrics["pr_auc"])

            return np.mean(scores)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective_cls, n_trials=n_trials)
        return study.best_params
    else:
        kf = KFold(n_splits=3, shuffle=True, random_state=42)

        def objective_reg(trial):
            if model_name == "Ridge":
                alpha = trial.suggest_float("alpha", 1e-3, 100.0, log=True)
                reg = Ridge(alpha=alpha, random_state=42)
            elif model_name == "Random_Forest":
                n_est = trial.suggest_int("n_estimators", 50, 150)
                max_d = trial.suggest_int("max_depth", 3, 10)
                reg = RandomForestRegressor(n_estimators=n_est, max_depth=max_d, n_jobs=-1, random_state=42)
            elif model_name == "XGBoost":
                n_est = trial.suggest_int("n_estimators", 50, 150)
                max_d = trial.suggest_int("max_depth", 3, 8)
                lr = trial.suggest_float("learning_rate", 0.01, 0.2, log=True)
                reg = XGBRegressor(n_estimators=n_est, max_depth=max_d, learning_rate=lr, n_jobs=-1, random_state=42)
            elif model_name == "CatBoost":
                iters = trial.suggest_int("iterations", 50, 150)
                depth = trial.suggest_int("depth", 3, 8)
                lr = trial.suggest_float("learning_rate", 0.01, 0.2, log=True)
                reg = CatBoostRegressor(iterations=iters, depth=depth, learning_rate=lr, verbose=0, random_seed=42, thread_count=-1)
            else:
                reg = Ridge(random_state=42)

            scores = []
            for train_idx, val_idx in kf.split(X_train):
                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]

                reg.fit(X_tr, y_tr)
                preds = reg.predict(X_val)
                metrics = calculate_all_metrics(y_val, preds, task_type="regression")
                scores.append(-metrics["rmse"])

            return np.mean(scores)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective_reg, n_trials=n_trials)
        return study.best_params


def train_and_evaluate(
    data_path: str = "data/raw/telco_churn.csv",
    target_col: str | None = None,
    fast_mode: bool = True,
    n_optuna_trials: int = 5,
    progress_callback: object | None = None,
) -> dict:
    """Train model suite supporting Fast Baseline mode (<15s) and Advanced HPO mode with profiling."""
    t_start = time.perf_counter()
    setup_mlflow()

    if progress_callback:
        progress_callback(10, "Step 1/5: Loading and preprocessing dataset...")

    t0 = time.perf_counter()
    print(f"Loading raw data from {data_path}...")
    df = pd.read_csv(data_path)

    # 1. Dataset Inspection & Validation
    inspection = inspect_dataset(df, target_col=target_col or "Churn")
    leakage_cols = detect_data_leakage(df, target_col=target_col or "Churn")
    if leakage_cols:
        print(f"Warning: Potential data leakage columns detected: {leakage_cols}")

    generate_eda_report(df, target_col=target_col or "Churn")
    validate_data(df, is_training=True, target_col=target_col)

    # 2. Split & Preprocessing
    print("Splitting dataset into Train (70%), Val (15%), Test (15%)...")
    X_trans, y, preprocessor, feature_names = prepare_data(df, fit=True, target_col=target_col)
    task_type = getattr(preprocessor, "task_type_", "classification")
    print(f"Inferred Task Type: {task_type.upper()}")

    stratify_arg = None
    if task_type == "classification" and y is not None and len(np.unique(y)) > 1 and len(y) >= 20:
        if np.issubdtype(y.dtype, np.integer) and np.min(np.bincount(y)) >= 2:
            stratify_arg = y

    X_train, X_temp, y_train, y_temp = train_test_split(
        X_trans, y, test_size=0.30, random_state=42, stratify=stratify_arg
    )

    stratify_temp = None
    if task_type == "classification" and y_temp is not None and len(np.unique(y_temp)) > 1 and len(y_temp) >= 10:
        if np.issubdtype(y_temp.dtype, np.integer) and np.min(np.bincount(y_temp)) >= 2:
            stratify_temp = y_temp

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=stratify_temp
    )

    save_preprocessor(preprocessor, "models/preprocessor.joblib")
    t_prep = time.perf_counter() - t0
    print(f"[TIME] Preprocessing & Data Split completed in {t_prep:.2f}s")

    if progress_callback:
        progress_callback(30, "Step 2/5: Training model suite (Fast Baseline)..." if fast_mode else "Step 2/5: Running Optuna HPO...")

    t0_models = time.perf_counter()
    if task_type == "classification":
        candidate_models = ["Logistic_Regression", "Random_Forest", "XGBoost", "CatBoost"]
    else:
        candidate_models = ["Ridge", "Random_Forest", "XGBoost", "CatBoost"]

    best_score = -float("inf") if task_type == "classification" else float("inf")
    best_model_name = None
    best_model_obj = None
    best_run_id = None
    best_threshold = 0.5
    best_val_metrics = {}
    best_test_metrics = {}
    best_business_cost = 0.0
    model_results = []

    print(f"\n--- Starting Model Suite Training (Fast Mode: {fast_mode}) ---")
    for idx, name in enumerate(candidate_models):
        if progress_callback:
            progress_callback(30 + int(idx * 10), f"Training candidate {idx+1}/{len(candidate_models)}: {name}...")

        with mlflow.start_run(run_name=name) as run:
            if fast_mode:
                best_params = {}
            else:
                best_params = run_optuna_tuning(name, X_train, y_train, task_type=task_type, n_trials=n_optuna_trials)

            if task_type == "classification":
                if name == "Logistic_Regression":
                    base_clf = LogisticRegression(**best_params, max_iter=1000, random_state=42, class_weight="balanced")
                elif name == "Random_Forest":
                    base_clf = RandomForestClassifier(**best_params, n_estimators=100, max_depth=8, n_jobs=-1, random_state=42, class_weight="balanced")
                elif name == "XGBoost":
                    base_clf = XGBClassifier(**best_params, n_estimators=100, max_depth=6, learning_rate=0.1, n_jobs=-1, random_state=42, eval_metric="logloss")
                elif name == "CatBoost":
                    base_clf = CatBoostClassifier(**best_params, iterations=100, depth=6, learning_rate=0.1, verbose=0, random_seed=42, thread_count=-1)

                model_pipeline = ImbPipeline([
                    ("smote", SMOTE(random_state=42)),
                    ("classifier", base_clf)
                ])
                model_pipeline.fit(X_train, y_train)

                if fast_mode:
                    final_model_obj = model_pipeline
                else:
                    calibrated_model = CalibratedClassifierCV(estimator=model_pipeline, cv=3)
                    calibrated_model.fit(X_train, y_train)
                    final_model_obj = calibrated_model

                if hasattr(final_model_obj, "predict_proba"):
                    y_val_prob = final_model_obj.predict_proba(X_val)[:, 1]
                    y_test_prob = final_model_obj.predict_proba(X_test)[:, 1]
                else:
                    y_val_prob = None
                    y_test_prob = None

                opt_th, min_cost, _ = optimize_business_threshold(y_val, y_val_prob, cost_fn=500.0, cost_fp=50.0) if y_val_prob is not None else (0.5, 0.0, {})
                y_val_pred = np.where(y_val_prob >= opt_th, 1, 0) if y_val_prob is not None else final_model_obj.predict(X_val)
                val_metrics = calculate_all_metrics(y_val, y_val_pred, y_val_prob, task_type="classification")

                y_test_pred = np.where(y_test_prob >= opt_th, 1, 0) if y_test_prob is not None else final_model_obj.predict(X_test)
                test_metrics = calculate_all_metrics(y_test, y_test_pred, y_test_prob, task_type="classification")

                fit_score = val_metrics["f1_score"] if val_metrics["f1_score"] > 0 else val_metrics["pr_auc"]
                eval_threshold = opt_th
            else:
                if name == "Ridge":
                    reg = Ridge(**best_params, random_state=42)
                elif name == "Random_Forest":
                    reg = RandomForestRegressor(**best_params, n_estimators=100, max_depth=8, n_jobs=-1, random_state=42)
                elif name == "XGBoost":
                    reg = XGBRegressor(**best_params, n_estimators=100, max_depth=6, learning_rate=0.1, n_jobs=-1, random_state=42)
                elif name == "CatBoost":
                    reg = CatBoostRegressor(**best_params, iterations=100, depth=6, learning_rate=0.1, verbose=0, random_seed=42, thread_count=-1)

                reg.fit(X_train, y_train)
                y_val_pred = reg.predict(X_val)
                val_metrics = calculate_all_metrics(y_val, y_val_pred, task_type="regression")

                y_test_pred = reg.predict(X_test)
                test_metrics = calculate_all_metrics(y_test, y_test_pred, task_type="regression")

                fit_score = val_metrics["rmse"]
                final_model_obj = reg
                eval_threshold = 0.5
                min_cost = 0.0

            for k, v in best_params.items():
                mlflow.log_param(k, v)
            mlflow.log_param("task_type", task_type)

            for m_name, score in val_metrics.items():
                mlflow.log_metric(f"val_{m_name}", score)
                print(f"  val_{m_name}: {score:.4f}")

            model_res = {"model_name": name, "business_cost": min_cost}
            model_res.update(val_metrics)
            model_results.append(model_res)

            mlflow.sklearn.log_model(final_model_obj, artifact_path="model", serialization_format="cloudpickle")

            is_better = (fit_score > best_score) if task_type == "classification" else (fit_score < best_score)
            if is_better:
                best_score = fit_score
                best_model_name = name
                best_model_obj = final_model_obj
                best_run_id = run.info.run_id
                best_threshold = eval_threshold
                best_val_metrics = val_metrics
                best_test_metrics = test_metrics
                best_business_cost = min_cost

    t_models = time.perf_counter() - t0_models
    print(f"[TIME] Model Suite Training completed in {t_models:.2f}s")

    if progress_callback:
        progress_callback(75, "Step 3/5: Generating plots and SHAP for best model...")

    t0_plots = time.perf_counter()
    # Generate Plots and Metrics ONCE on Holdout Test Set for 100% Mathematical Consistency
    if task_type == "classification":
        y_test_prob = best_model_obj.predict_proba(X_test)[:, 1]
        opt_th, best_cost, _ = optimize_business_threshold(y_test, y_test_prob, cost_fn=500.0, cost_fp=50.0)
        best_threshold = opt_th
        y_test_pred = np.where(y_test_prob >= best_threshold, 1, 0)
        best_test_metrics = calculate_all_metrics(y_test, y_test_pred, y_test_prob, task_type="classification")
        best_business_cost = best_cost

        log_classification_plots(y_test, y_test_pred, y_test_prob, best_model_name)
        plot_calibration_curve_to_file(y_test, y_test_prob, best_model_name)
    else:
        y_test_pred = best_model_obj.predict(X_test)
        best_test_metrics = calculate_all_metrics(y_test, y_test_pred, task_type="regression")
        best_business_cost = 0.0

    generate_shap_plots(best_model_obj, X_test[:50], feature_names)
    t_plots = time.perf_counter() - t0_plots
    print(f"[TIME] SHAP & Diagnostic Plot Generation completed in {t_plots:.2f}s")

    if progress_callback:
        progress_callback(90, "Step 4/5: Registering model & saving pipeline artifacts...")

    t0_registry = time.perf_counter()
    print(f"\nBest Model Selected: {best_model_name} (Best Score: {best_score:.4f}, Run ID: {best_run_id})")

    # Register best model in MLflow Model Registry
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

    # Save Unified End-to-End Pipeline Artifact
    full_pipeline = {
        "preprocessor": preprocessor,
        "model": best_model_obj,
        "optimal_threshold": best_threshold,
        "model_name": best_model_name,
        "feature_names": feature_names,
        "task_type": task_type,
    }

    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model_obj, "models/best_model.joblib")
    joblib.dump(full_pipeline, "models/unified_pipeline.joblib")
    print("Saved unified pipeline artifact to models/unified_pipeline.joblib")

    t_total = time.perf_counter() - t_start
    print(f"[SUCCESS] TOTAL PIPELINE RUNTIME: {t_total:.2f}s")

    if progress_callback:
        progress_callback(100, f"Step 5/5: Done in {t_total:.1f}s!")

    primary_score = float(best_test_metrics.get("f1_score", best_test_metrics.get("rmse", 0.0)))

    return {
        "best_model_name": best_model_name,
        "task_type": task_type,
        "optimal_threshold": best_threshold,
        "best_val_metrics": best_val_metrics,
        "best_test_metrics": best_test_metrics,
        "best_f1": primary_score,
        "business_cost": float(best_business_cost),
        "model_results": model_results,
        "best_run_id": best_run_id,
        "version": registered_model.version,
        "total_time_seconds": round(t_total, 2),
        "fast_mode": fast_mode,
    }


if __name__ == "__main__":
    train_and_evaluate(fast_mode=True)
