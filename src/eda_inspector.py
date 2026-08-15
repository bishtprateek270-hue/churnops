"""
Dataset inspection, exploratory data analysis (EDA), and data leakage detection module.
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

from src.preprocessing import infer_task_type


def inspect_dataset(df: pd.DataFrame, target_col: str = "Churn") -> dict:
    """Perform dataset inspection for missing values, duplicates, dtypes, and target properties."""
    num_rows, num_cols = df.shape
    missing_summary = df.isna().sum().to_dict()
    missing_ratios = (df.isna().mean()).to_dict()
    duplicates_count = int(df.duplicated().sum())

    target_found = target_col in df.columns
    task_type = "classification"
    target_summary = {}

    if target_found:
        target_series = df[target_col]
        task_type = infer_task_type(target_series)

        if task_type == "classification":
            class_counts = target_series.value_counts().to_dict()
            class_ratios = target_series.value_counts(normalize=True).to_dict()
            target_summary = {
                "counts": {str(k): int(v) for k, v in class_counts.items()},
                "ratios": {str(k): float(v) for k, v in class_ratios.items()},
            }
        else:
            target_summary = {
                "mean": float(target_series.mean()),
                "std": float(target_series.std()),
                "min": float(target_series.min()),
                "max": float(target_series.max()),
            }

    dtypes_summary = {col: str(dtype) for col, dtype in df.dtypes.items()}

    return {
        "num_rows": num_rows,
        "num_cols": num_cols,
        "duplicates_count": duplicates_count,
        "missing_summary": missing_summary,
        "missing_ratios": missing_ratios,
        "dtypes": dtypes_summary,
        "target_col": target_col,
        "target_found": target_found,
        "task_type": task_type,
        "target_summary": target_summary,
    }


def detect_data_leakage(df: pd.DataFrame, target_col: str = "Churn") -> list[str]:
    """Identify potential data leakage columns (e.g. constant features or 100% correlated features)."""
    suspicious_cols = []
    if target_col not in df.columns:
        return suspicious_cols

    target_numeric = pd.to_numeric(df[target_col], errors="coerce")
    if target_numeric.notna().sum() == 0:
        target_numeric, _ = pd.factorize(df[target_col])

    for col in df.columns:
        if col == target_col or col.lower() in ["customerid", "id", "index", "user_id"]:
            continue

        # Check perfect correlation with target
        if pd.api.types.is_numeric_dtype(df[col]):
            corr = pd.Series(df[col]).corr(pd.Series(target_numeric))
            if not np.isnan(corr) and abs(corr) > 0.98:
                suspicious_cols.append(f"{col} (extremely high correlation: {corr:.4f})")

        # Check single-value columns (zero variance)
        if df[col].nunique() <= 1:
            suspicious_cols.append(f"{col} (zero variance / constant column)")

    return suspicious_cols


def generate_eda_report(df: pd.DataFrame, target_col: str = "Churn", output_dir: str = "reports/eda") -> dict:
    """Generate task-appropriate EDA charts and log to MLflow/disk."""
    os.makedirs(output_dir, exist_ok=True)
    report_artifacts = {}

    if target_col in df.columns:
        task_type = infer_task_type(df[target_col])
        plt.figure(figsize=(6, 4))
        if task_type == "classification":
            sns.countplot(x=target_col, data=df, hue=target_col, legend=False, palette="Blues_d")
            plt.title(f"Target Classification Distribution: {target_col}")
        else:
            sns.histplot(df[target_col].dropna(), kde=True, color="blue")
            plt.title(f"Target Regression Distribution: {target_col}")

        target_plot_path = os.path.join(output_dir, "target_distribution.png")
        plt.tight_layout()
        plt.savefig(target_plot_path)
        plt.close()
        report_artifacts["target_distribution_plot"] = target_plot_path

    # Numerical Correlation Heatmap
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] > 1:
        plt.figure(figsize=(8, 6))
        sns.heatmap(num_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
        plt.title("Numerical Feature Correlation Matrix")
        corr_path = os.path.join(output_dir, "numerical_correlation_matrix.png")
        plt.tight_layout()
        plt.savefig(corr_path)
        plt.close()
        report_artifacts["correlation_matrix_plot"] = corr_path

    if mlflow.active_run():
        for path in report_artifacts.values():
            mlflow.log_artifact(path, artifact_path="eda")

    return report_artifacts
