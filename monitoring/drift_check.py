"""
Data drift monitoring using Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) test.
"""

import os
import sys

# Ensure workspace root directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import sqlite3
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.config import settings

DB_PATH = settings.PREDICTIONS_DB_PATH
TRAIN_DATA_PATH = settings.DEFAULT_DATA_PATH


def calculate_psi_numerical(expected: np.ndarray, actual: np.ndarray, num_bins: int = 10) -> float:
    """Calculate Population Stability Index (PSI) for continuous numerical variables."""
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    # Define bin edges based on expected distribution quantiles
    percentiles = np.linspace(0, 100, num_bins + 1)
    bins = np.percentile(expected, percentiles)
    bins = np.unique(bins)  # Deduplicate identical quantiles

    if len(bins) < 2:
        return 0.0

    bins[0] = -np.inf
    bins[-1] = np.inf

    expected_counts, _ = np.histogram(expected, bins=bins)
    actual_counts, _ = np.histogram(actual, bins=bins)

    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)

    # Avoid division by zero or log(0) using epsilon smoothing
    eps = 1e-4
    expected_pct = np.where(expected_pct == 0, eps, expected_pct)
    actual_pct = np.where(actual_pct == 0, eps, actual_pct)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


def calculate_psi_categorical(expected: pd.Series, actual: pd.Series) -> float:
    """Calculate PSI for discrete categorical variables."""
    all_categories = set(expected.unique()).union(set(actual.unique()))

    expected_vc = expected.value_counts(normalize=True)
    actual_vc = actual.value_counts(normalize=True)

    eps = 1e-4
    psi = 0.0
    for cat in all_categories:
        exp_pct = expected_vc.get(cat, eps)
        act_pct = actual_vc.get(cat, eps)
        psi += (act_pct - exp_pct) * np.log(act_pct / exp_pct)

    return float(psi)


def compute_ks_test(expected: np.ndarray, actual: np.ndarray) -> tuple[float, float]:
    """Compute 2-sample Kolmogorov-Smirnov test statistic and p-value."""
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if len(expected) == 0 or len(actual) == 0:
        return 0.0, 1.0
    stat, p_val = ks_2samp(expected, actual)
    return float(stat), float(p_val)


def run_drift_analysis(db_path: str = DB_PATH, train_path: str = TRAIN_DATA_PATH) -> dict[str, Any]:
    """Compare distributions of live prediction inputs vs reference training dataset."""
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Baseline training dataset not found at {train_path}")

    if not os.path.exists(db_path):
        return {
            "status": "error",
            "message": f"No inference logs found at {db_path}. Submit prediction requests first.",
        }

    # Load reference training data
    df_ref = pd.read_csv(train_path)

    # Load actual production inputs from SQLite
    conn = sqlite3.connect(db_path)
    df_act = pd.read_sql_query("SELECT * FROM predictions", conn)
    conn.close()

    if df_act.empty:
        return {"status": "error", "message": "SQLite predictions table is empty."}

    total_logged_requests = len(df_act)
    print(f"Running data drift check on {total_logged_requests} production prediction logs...")

    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    cat_cols = ["Contract", "InternetService", "PaymentMethod", "PaperlessBilling", "OnlineSecurity", "TechSupport"]

    feature_metrics = {}
    max_psi = 0.0
    drift_detected = False

    # Check numerical features
    for col in num_cols:
        if col in df_ref.columns and col in df_act.columns:
            exp_vals = pd.to_numeric(df_ref[col], errors="coerce").values
            act_vals = pd.to_numeric(df_act[col], errors="coerce").values

            psi = calculate_psi_numerical(exp_vals, act_vals)
            ks_stat, ks_pval = compute_ks_test(exp_vals, act_vals)

            if psi > 0.25:
                status_str = "CRITICAL DRIFT"
                drift_detected = True
            elif psi > 0.10:
                status_str = "MODERATE DRIFT"
            else:
                status_str = "STABLE"

            max_psi = max(max_psi, psi)
            feature_metrics[col] = {
                "type": "numerical",
                "psi": round(psi, 4),
                "ks_stat": round(ks_stat, 4),
                "ks_pvalue": round(ks_pval, 4),
                "status": status_str,
            }

    # Check categorical features
    for col in cat_cols:
        if col in df_ref.columns and col in df_act.columns:
            psi = calculate_psi_categorical(df_ref[col], df_act[col])

            if psi > 0.25:
                status_str = "CRITICAL DRIFT"
                drift_detected = True
            elif psi > 0.10:
                status_str = "MODERATE DRIFT"
            else:
                status_str = "STABLE"

            max_psi = max(max_psi, psi)
            feature_metrics[col] = {
                "type": "categorical",
                "psi": round(psi, 4),
                "ks_stat": None,
                "ks_pvalue": None,
                "status": status_str,
            }

    report = {
        "status": "success",
        "total_requests": total_logged_requests,
        "max_psi": round(max_psi, 4),
        "drift_alert": drift_detected,
        "feature_metrics": feature_metrics,
    }

    print("\n--- Data Drift Analysis Summary ---")
    print(f"Total Prediction Logs Evaluated: {total_logged_requests}")
    print(f"Max PSI Score Across Features: {max_psi:.4f}")
    print(f"Drift Alert Triggered (PSI > 0.25): {'YES [ALERT]' if drift_detected else 'NO [OK]'}\n")
    for feat, metrics in feature_metrics.items():
        print(f"  {feat:20s} | PSI: {metrics['psi']:.4f} | Status: {metrics['status']}")

    return report


if __name__ == "__main__":
    run_drift_analysis()
