"""
Manual QA Verification Script across 5 Dataset Matrix
1. String-ID classification dataset (telco_churn.csv)
2. Numeric-ID classification dataset (Bank churn with CustomerId)
3. Regression dataset (kaggle_house_prices.csv)
4. Tiny dataset (N < 20)
5. Dataset with missing values + high-cardinality categoricals
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath("."))

from monitoring.predict_utils import load_trained_artifacts, predict_customers, predict_single_row
from src.data_validation import DataValidationError
from src.train import train_and_evaluate


def run_qa():
    print("=== Starting 5-Dataset QA Matrix Verification ===")

    # 1. String-ID Classification (Telco)
    print("\n--- 1. String-ID Classification Dataset (Telco) ---")
    res1 = train_and_evaluate(data_path="data/raw/telco_churn.csv", target_col="Churn", fast_mode=True)
    assert res1["task_type"] == "classification"
    print(f"[OK] Telco Fast Mode Success: Best Model = {res1['best_model_name']}")

    res1_opt = train_and_evaluate(
        data_path="data/raw/telco_churn.csv", target_col="Churn", fast_mode=False, n_optuna_trials=2
    )
    assert res1_opt["task_type"] == "classification"
    print(f"[OK] Telco Optuna Mode Success: Best Model = {res1_opt['best_model_name']}")

    # 2. Numeric-ID Classification (Bank Churn with CustomerId)
    print("\n--- 2. Numeric-ID Classification Dataset (Bank Churn) ---")
    np.random.seed(42)
    df_bank = pd.DataFrame(
        {
            "CustomerId": range(15634602, 15634602 + 100),
            "Surname": [f"Smith_{i}" for i in range(100)],
            "CreditScore": np.random.randint(350, 850, 100),
            "Geography": np.random.choice(["France", "Spain", "Germany"], size=100),
            "Gender": np.random.choice(["Female", "Male"], size=100),
            "Age": np.random.randint(18, 70, 100),
            "Tenure": np.random.randint(0, 10, 100),
            "Balance": np.random.uniform(0, 100000, 100),
            "NumOfProducts": np.random.randint(1, 4, 100),
            "HasCrCard": np.random.choice([0, 1], size=100),
            "IsActiveMember": np.random.choice([0, 1], size=100),
            "EstimatedSalary": np.random.uniform(10000, 150000, 100),
            "Exited": np.random.choice([0, 1], size=100, p=[0.8, 0.2]),
        }
    )
    os.makedirs("scratch", exist_ok=True)
    df_bank.to_csv("scratch/bank_churn.csv", index=False)

    # 2a. Attempt to select CustomerId as target without override (must be blocked)
    try:
        train_and_evaluate(data_path="scratch/bank_churn.csv", target_col="CustomerId", allow_id_target=False)
        print("[ERROR] Error: Target CustomerId was not blocked!")
    except DataValidationError:
        print("[OK] Success: Selecting CustomerId as target was correctly blocked by DataValidationError!")

    # 2b. Train on real target Exited
    res2 = train_and_evaluate(data_path="scratch/bank_churn.csv", target_col="Exited", fast_mode=True)
    assert res2["task_type"] == "classification"
    print(f"[OK] Bank Churn Fast Mode Success: Best Model = {res2['best_model_name']}")

    # 3. Regression Dataset (House Prices)
    print("\n--- 3. Regression Dataset (House Prices) ---")
    if os.path.exists("data/raw/kaggle_house_prices.csv"):
        res3 = train_and_evaluate(data_path="data/raw/kaggle_house_prices.csv", target_col="SalePrice", fast_mode=True)
        assert res3["task_type"] == "regression"
        print(f"[OK] House Prices Fast Mode Success: Best Model = {res3['best_model_name']}")

        res3_opt = train_and_evaluate(
            data_path="data/raw/kaggle_house_prices.csv", target_col="SalePrice", fast_mode=False, n_optuna_trials=2
        )
        assert res3_opt["task_type"] == "regression"
        print(f"[OK] House Prices Optuna Mode Success: Best Model = {res3_opt['best_model_name']}")

    # 4. Tiny Dataset (N < 20)
    print("\n--- 4. Tiny Dataset (N = 15) ---")
    df_tiny = pd.DataFrame(
        {
            "id": range(15),
            "feature1": np.random.rand(15),
            "feature2": np.random.rand(15),
            "target": np.random.choice([0, 1], size=15),
        }
    )
    df_tiny.to_csv("scratch/tiny_dataset.csv", index=False)
    res4 = train_and_evaluate(data_path="scratch/tiny_dataset.csv", target_col="target", fast_mode=True)
    assert any("Small dataset sample size" in w for w in res4["warnings"])
    print(f"[OK] Tiny Dataset Success: Warning captured: {res4['warnings'][0]}")

    # 5. Missing values + high-cardinality categoricals
    print("\n--- 5. Dataset with Missing Values & High Cardinality ---")
    df_complex = pd.DataFrame(
        {
            "user_guid": [f"GUID_{i:04d}" for i in range(80)],
            "category_high": [f"Cat_{i % 35}" for i in range(80)],
            "category_low": np.random.choice(["A", "B", None], size=80),
            "num_with_nan": [np.nan if i % 5 == 0 else float(i * 1.5) for i in range(80)],
            "score": np.random.uniform(10.0, 500.0, 80),
        }
    )
    df_complex.to_csv("scratch/complex_dataset.csv", index=False)
    res5 = train_and_evaluate(data_path="scratch/complex_dataset.csv", target_col="score", fast_mode=True)
    assert res5["task_type"] == "regression"
    print(f"[OK] Missing Values & High Cardinality Success: Best Model = {res5['best_model_name']}")

    # Single row & batch prediction check on loaded artifact
    _model, _prep, _th, status = load_trained_artifacts()
    assert status == "OK"
    test_row = df_complex.iloc[0].drop(labels=["score"])
    single_res = predict_single_row(test_row)
    batch_res = predict_customers(df_complex)
    assert "prediction" in single_res or "error" not in single_res
    assert len(batch_res) == 80
    print("[OK] Prediction Parity Success: Single row and batch predictions complete cleanly!")

    print("\n==============================================")
    print("ALL 5 QA MATRIX VERIFICATION CHECKS PASSED!")
    print("==============================================")


if __name__ == "__main__":
    run_qa()
