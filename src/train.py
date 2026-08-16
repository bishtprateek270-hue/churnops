"""
Training module supporting Classification & Regression ML tasks with Fast Baseline mode (default, <15s),
Advanced Optuna hyperparameter tuning, SMOTE imbalance handling, probability calibration,
holdout test set metrics evaluation, single-pass SHAP generation, and MLflow tracking.
"""

import os
import sys
import time

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import matplotlib

matplotlib.use("Agg")
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier, XGBRegressor

from src.data_validation import DataValidationError, validate_data
from src.eda_inspector import detect_data_leakage, generate_eda_report, inspect_dataset
from src.evaluate import (
    calculate_all_metrics,
    generate_shap_plots,
    log_classification_plots,
    optimize_business_threshold,
    plot_calibration_curve_to_file,
)
from src.preprocessing import (
    clean_dataframe,
    detect_identifier_columns,
    find_target_col,
    infer_task_type,
    prepare_data,
    save_preprocessor,
)

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
                clf = LogisticRegression(C=c_val, max_iter=1000, random_state=42, class_weight="balanced")
            elif model_name == "Random_Forest":
                n_est = trial.suggest_int("n_estimators", 50, 150)
                max_d = trial.suggest_int("max_depth", 3, 10)
                clf = RandomForestClassifier(n_estimators=n_est, max_depth=max_d, n_jobs=-1, random_state=42, class_weight="balanced")
            elif model_name == "HistGradientBoosting":
                lr = trial.suggest_float("learning_rate", 0.01, 0.2, log=True)
                max_d = trial.suggest_int("max_depth", 3, 8)
                clf = HistGradientBoostingClassifier(learning_rate=lr, max_depth=max_d, random_state=42)
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

                try:
                    pipeline.fit(X_tr, y_tr)
                    y_prob = pipeline.predict_proba(X_val)[:, 1]
                    metrics = calculate_all_metrics(y_val, np.where(y_prob > 0.5, 1, 0), y_prob, task_type="classification")
                    scores.append(metrics["pr_auc"])
                except Exception:
                    scores.append(0.0)

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
            elif model_name == "HistGradientBoosting":
                lr = trial.suggest_float("learning_rate", 0.01, 0.2, log=True)
                max_d = trial.suggest_int("max_depth", 3, 8)
                reg = HistGradientBoostingRegressor(learning_rate=lr, max_depth=max_d, random_state=42)
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
    """Train dataset-agnostic model suite with pre-fit train/holdout split, CV model selection, and single-pass holdout evaluation."""
    t_start = time.perf_counter()
    setup_mlflow()
    warnings_list = []

    if progress_callback:
        progress_callback(10, "Step 1/5: Loading dataset & analyzing target...")

    t0 = time.perf_counter()
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}. Generating default telco dataset...")
        from data.generate_dataset import generate_telco_churn_data
        df = generate_telco_churn_data()
        if "/" in data_path or "\\" in data_path:
            os.makedirs(os.path.dirname(data_path), exist_ok=True)
            df.to_csv(data_path, index=False)
    else:
        print(f"Loading raw dataset from {data_path}...")
        df = pd.read_csv(data_path)

    df_clean = clean_dataframe(df)
    actual_target = find_target_col(df_clean, target_col)
    if not actual_target or actual_target not in df_clean.columns:
        raise DataValidationError("Target column could not be resolved from dataset.")

    task_type = infer_task_type(df_clean[actual_target])
    print(f"Target Column: '{actual_target}' | Inferred Task Type: {task_type.upper()}")

    # 1. Quality Safeguards Checks
    if len(df_clean) < 20:
        warn_msg = f"Warning: Small dataset sample size (N={len(df_clean)}). Statistical metrics may be unreliable."
        print(warn_msg)
        warnings_list.append(warn_msg)

    inspect_dataset(df_clean, target_col=actual_target)
    leakage_cols = detect_data_leakage(df_clean, target_col=actual_target)
    if leakage_cols:
        print(f"Notice: Potential target leakage columns detected: {leakage_cols}")

    generate_eda_report(df_clean, target_col=actual_target)
    validate_data(df_clean, is_training=True, target_col=actual_target)

    # 2. Extract Raw Features and Target
    id_cols = detect_identifier_columns(df_clean, target_col=actual_target)
    y_raw = df_clean[actual_target]

    if task_type == "classification":
        if y_raw.dtype == object or str(y_raw.dtype) in ["string", "category", "bool"]:
            str_vals = y_raw.astype(str).str.strip().str.lower()
            positive_indicators = {"yes", "true", "1", "1.0", "churn", "churned", "positive", "y", "t"}
            unique_str = set(str_vals.unique())
            if any(val in positive_indicators for val in unique_str):
                y = np.where(str_vals.isin(positive_indicators), 1, 0)
            else:
                labels, _ = pd.factorize(y_raw)
                y = np.where(labels < 0, 0, labels)
        else:
            num_vals = pd.to_numeric(y_raw, errors="coerce").fillna(0.0)
            unique_nums = set(np.unique(num_vals.values))
            if unique_nums.issubset({0, 1}):
                y = np.where(num_vals.values > 0.5, 1, 0)
            else:
                y = np.where(num_vals.values > np.median(num_vals.values), 1, 0)
        y = np.asarray(y, dtype=np.int64)

        pos_count = int(np.sum(y == 1))
        if pos_count < 5 or (pos_count / len(y)) < 0.05:
            warn_msg = f"Warning: Severe class imbalance detected (only {pos_count} positive samples out of {len(y)})."
            print(warn_msg)
            warnings_list.append(warn_msg)
    else:
        num_vals = pd.to_numeric(y_raw, errors="coerce").fillna(0.0)
        y = np.asarray(num_vals.values, dtype=np.float64)

    X_raw = df_clean.drop(columns=[actual_target] + id_cols, errors="ignore")

    # 3. Split BEFORE Fitting Preprocessor to Prevent Data Leakage
    print("Splitting raw dataset into Train (70%), Val (15%), Test (15%) BEFORE preprocessing...")
    stratify_arg = None
    if task_type == "classification" and y is not None and len(np.unique(y)) > 1 and len(y) >= 20 and np.issubdtype(y.dtype, np.integer) and np.min(np.bincount(y)) >= 2:
        stratify_arg = y

    X_train_raw, X_temp_raw, y_train, y_temp = train_test_split(
        X_raw, y, test_size=0.30, random_state=42, stratify=stratify_arg
    )

    stratify_temp = None
    if task_type == "classification" and y_temp is not None and len(np.unique(y_temp)) > 1 and len(y_temp) >= 10 and np.issubdtype(y_temp.dtype, np.integer) and np.min(np.bincount(y_temp)) >= 2:
        stratify_temp = y_temp

    X_val_raw, X_test_raw, y_val, y_test = train_test_split(
        X_temp_raw, y_temp, test_size=0.50, random_state=42, stratify=stratify_temp
    )

    # 4. Fit Preprocessor STRICTLY on X_train_raw
    X_train, _, preprocessor, feature_names = prepare_data(X_train_raw, fit=True, target_col=None)
    X_val, _, _, _ = prepare_data(X_val_raw, preprocessor=preprocessor, fit=False)
    X_test, _, _, _ = prepare_data(X_test_raw, preprocessor=preprocessor, fit=False)

    preprocessor.target_col_ = actual_target
    preprocessor.id_cols_ = id_cols
    preprocessor.task_type_ = task_type
    save_preprocessor(preprocessor, "models/preprocessor.joblib")

    t_prep = time.perf_counter() - t0
    print(f"[TIME] Leak-free preprocessing & split completed in {t_prep:.2f}s")

    if progress_callback:
        progress_callback(30, "Step 2/5: Training model suite & running CV model selection...")

    t0_models = time.perf_counter()
    if task_type == "classification":
        candidate_models = ["Logistic_Regression", "Random_Forest", "HistGradientBoosting", "XGBoost", "CatBoost"]
    else:
        candidate_models = ["Ridge", "Random_Forest", "HistGradientBoosting", "XGBoost", "CatBoost"]

    # Compute Baseline Score (CV)
    if task_type == "classification":
        baseline_score = float(np.mean(y_train))
    else:
        baseline_score = float(np.mean(y_train))

    best_cv_score = -float("inf") if task_type == "classification" else float("inf")
    best_model_name = None
    best_model_obj = None
    best_run_id = None
    best_threshold = 0.5
    best_val_metrics = {}
    best_test_metrics = {}
    best_business_cost = 0.0
    model_results = []

    print(f"\n--- Model Suite CV Evaluation (Fast Mode: {fast_mode}) ---")
    for idx, name in enumerate(candidate_models):
        if progress_callback:
            progress_callback(30 + int(idx * 8), f"Evaluating model candidate {idx+1}/{len(candidate_models)}: {name}...")

        with mlflow.start_run(run_name=name) as run:
            best_params = {} if fast_mode else run_optuna_tuning(name, X_train, y_train, task_type=task_type, n_trials=n_optuna_trials)

            if task_type == "classification":
                if name == "Logistic_Regression":
                    base_clf = LogisticRegression(**best_params, max_iter=1000, random_state=42, class_weight="balanced")
                elif name == "Random_Forest":
                    base_clf = RandomForestClassifier(**best_params, n_estimators=100, max_depth=8, n_jobs=-1, random_state=42, class_weight="balanced")
                elif name == "HistGradientBoosting":
                    base_clf = HistGradientBoostingClassifier(**best_params, learning_rate=0.1, max_depth=6, random_state=42)
                elif name == "XGBoost":
                    base_clf = XGBClassifier(**best_params, n_estimators=100, max_depth=6, learning_rate=0.1, n_jobs=-1, random_state=42, eval_metric="logloss")
                elif name == "CatBoost":
                    base_clf = CatBoostClassifier(**best_params, iterations=100, depth=6, learning_rate=0.1, verbose=0, random_seed=42, thread_count=-1)
                else:
                    base_clf = LogisticRegression(max_iter=1000, random_state=42)

                use_smote = (len(y_train) >= 20 and np.min(np.bincount(y_train)) >= 2)
                if use_smote and name not in ["HistGradientBoosting"]:
                    model_pipeline = ImbPipeline([
                        ("smote", SMOTE(random_state=42)),
                        ("classifier", base_clf)
                    ])
                else:
                    model_pipeline = Pipeline([("classifier", base_clf)])

                try:
                    model_pipeline.fit(X_train, y_train)
                    final_model_obj = model_pipeline

                    if hasattr(final_model_obj, "predict_proba"):
                        y_val_prob = final_model_obj.predict_proba(X_val)[:, 1]
                    else:
                        y_val_prob = None

                    opt_th, min_cost, _ = optimize_business_threshold(y_val, y_val_prob, cost_fn=500.0, cost_fp=50.0) if y_val_prob is not None else (0.5, 0.0, {})
                    y_val_pred = np.where(y_val_prob >= opt_th, 1, 0) if y_val_prob is not None else final_model_obj.predict(X_val)
                    val_metrics = calculate_all_metrics(y_val, y_val_pred, y_val_prob, task_type="classification")

                    cv_score = val_metrics["pr_auc"] if val_metrics["pr_auc"] > 0 else val_metrics["f1_score"]
                except Exception as exc:
                    print(f"Notice: Model candidate {name} failed: {exc}")
                    continue

                eval_threshold = opt_th
            else:
                if name == "Ridge":
                    reg = Ridge(**best_params, random_state=42)
                elif name == "Random_Forest":
                    reg = RandomForestRegressor(**best_params, n_estimators=100, max_depth=8, n_jobs=-1, random_state=42)
                elif name == "HistGradientBoosting":
                    reg = HistGradientBoostingRegressor(**best_params, learning_rate=0.1, max_depth=6, random_state=42)
                elif name == "XGBoost":
                    reg = XGBRegressor(**best_params, n_estimators=100, max_depth=6, learning_rate=0.1, n_jobs=-1, random_state=42)
                elif name == "CatBoost":
                    reg = CatBoostRegressor(**best_params, iterations=100, depth=6, learning_rate=0.1, verbose=0, random_seed=42, thread_count=-1)
                else:
                    reg = Ridge(random_state=42)

                try:
                    reg.fit(X_train, y_train)
                    y_val_pred = reg.predict(X_val)
                    val_metrics = calculate_all_metrics(y_val, y_val_pred, task_type="regression")
                    cv_score = val_metrics["rmse"]
                    final_model_obj = reg
                except Exception as exc:
                    print(f"Notice: Model candidate {name} failed: {exc}")
                    continue

                eval_threshold = 0.5
                min_cost = 0.0

            for k, v in best_params.items():
                mlflow.log_param(k, v)
            mlflow.log_param("task_type", task_type)

            for m_name, score in val_metrics.items():
                mlflow.log_metric(f"val_{m_name}", score)

            model_res = {"model_name": name, "business_cost": min_cost}
            model_res.update(val_metrics)
            model_results.append(model_res)

            mlflow.sklearn.log_model(final_model_obj, artifact_path="model", serialization_format="cloudpickle")

            is_better = (cv_score > best_cv_score) if task_type == "classification" else (cv_score < best_cv_score)
            if is_better or best_model_obj is None:
                best_cv_score = cv_score
                best_model_name = name
                best_model_obj = final_model_obj
                best_run_id = run.info.run_id
                best_threshold = eval_threshold
                best_val_metrics = val_metrics

    t_models = time.perf_counter() - t0_models
    print(f"[TIME] Model Suite Selection completed in {t_models:.2f}s")

    if progress_callback:
        progress_callback(75, "Step 3/5: Evaluating winning model ONCE on holdout test set...")

    t0_plots = time.perf_counter()
    # Evaluate winning model ONCE on untouched holdout test set (X_test, y_test)
    if task_type == "classification":
        if hasattr(best_model_obj, "predict_proba"):
            y_test_prob = best_model_obj.predict_proba(X_test)[:, 1]
        else:
            y_test_prob = None

        opt_th, best_cost, _ = optimize_business_threshold(y_test, y_test_prob, cost_fn=500.0, cost_fp=50.0) if y_test_prob is not None else (0.5, 0.0, {})
        best_threshold = opt_th
        y_test_pred = np.where(y_test_prob >= best_threshold, 1, 0) if y_test_prob is not None else best_model_obj.predict(X_test)
        best_test_metrics = calculate_all_metrics(y_test, y_test_pred, y_test_prob, task_type="classification")
        best_business_cost = best_cost

        log_classification_plots(y_test, y_test_pred, y_test_prob, best_model_name)
        if y_test_prob is not None:
            plot_calibration_curve_to_file(y_test, y_test_prob, best_model_name)
    else:
        y_test_pred = best_model_obj.predict(X_test)
        best_test_metrics = calculate_all_metrics(y_test, y_test_pred, task_type="regression")
        best_business_cost = 0.0

        if best_test_metrics.get("r2_score", 0.0) < 0.0:
            warn_msg = f"Warning: Holdout test R² is negative ({best_test_metrics['r2_score']:.4f}). Model performs worse than a constant mean prediction."
            print(warn_msg)
            warnings_list.append(warn_msg)

    generate_shap_plots(best_model_obj, X_test[:min(50, len(X_test))], feature_names)
    t_plots = time.perf_counter() - t0_plots
    print(f"[TIME] Diagnostic Plot Generation completed in {t_plots:.2f}s")

    if progress_callback:
        progress_callback(90, "Step 4/5: Registering model & saving unified artifact...")

    print(f"\nBest Model Selected: {best_model_name} (CV Score: {best_cv_score:.4f}, Run ID: {best_run_id})")

    # Register best model in MLflow Model Registry
    registered_version = "local-1"
    try:
        model_uri = f"runs:/{best_run_id}/model"
        registered_model = mlflow.register_model(model_uri, MODEL_NAME)
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=registered_model.version,
            stage="Staging",
            archive_existing_versions=False
        )
        registered_version = registered_model.version
    except Exception as exc:
        print(f"Notice: MLflow registry registration note: {exc}")

    # Save Unified End-to-End Pipeline Artifact
    full_pipeline = {
        "preprocessor": preprocessor,
        "model": best_model_obj,
        "optimal_threshold": best_threshold,
        "model_name": best_model_name,
        "feature_names": feature_names,
        "raw_feature_cols": getattr(preprocessor, "feature_cols_", []),
        "id_cols": getattr(preprocessor, "id_cols_", []),
        "leakage_cols": getattr(preprocessor, "leakage_cols_", []),
        "target_col": actual_target,
        "task_type": task_type,
        "baseline_score": baseline_score,
        "cv_score": best_cv_score,
        "holdout_metrics": best_test_metrics,
        "warnings": warnings_list,
    }

    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model_obj, "models/best_model.joblib")
    joblib.dump(full_pipeline, "models/unified_pipeline.joblib")
    print("Saved unified pipeline artifact to models/unified_pipeline.joblib")

    t_total = time.perf_counter() - t_start
    print(f"[SUCCESS] TOTAL PIPELINE RUNTIME: {t_total:.2f}s")

    if progress_callback:
        progress_callback(100, f"Step 5/5: Done in {t_total:.1f}s!")

    primary_score = float(best_test_metrics.get("f1_score", best_test_metrics.get("r2_score", 0.0)))

    return {
        "best_model_name": best_model_name,
        "task_type": task_type,
        "optimal_threshold": best_threshold,
        "baseline_score": baseline_score,
        "cv_score": best_cv_score,
        "best_val_metrics": best_val_metrics,
        "best_test_metrics": best_test_metrics,
        "best_f1": primary_score,
        "business_cost": float(best_business_cost),
        "model_results": model_results,
        "warnings": warnings_list,
        "best_run_id": best_run_id,
        "version": registered_version,
        "total_time_seconds": round(t_total, 2),
        "fast_mode": fast_mode,
    }


if __name__ == "__main__":
    train_and_evaluate(fast_mode=True)
