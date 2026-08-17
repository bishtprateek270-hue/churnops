"""
Unit tests verifying heuristic identifier column detection, exact feature schema alignment,
exclusion of IDs from training/SHAP, and prediction parity between dataset-row selection and manual input.
"""

import os

import numpy as np
import pandas as pd
import pytest

from monitoring.predict_utils import load_trained_artifacts, predict_single_row
from src.preprocessing import (
    detect_identifier_columns,
    is_identifier_column,
    prepare_data,
)
from src.train import train_and_evaluate


def test_id_detection_heuristics():
    """Verify true identifiers are detected and valid features with 'id' substring are preserved."""
    n = 100
    df = pd.DataFrame({
        "Id": list(range(1, n + 1)),
        "customerID": [f"CUST_{i:04d}" for i in range(n)],
        "HouseID": list(range(1001, 1001 + n)),
        "TransactionID": [f"TX_{i:05d}" for i in range(n)],
        "valid_id": np.random.choice([0, 1], size=n),
        "idea_score": np.random.uniform(0.1, 0.9, size=n),
        "fluid_level": np.random.uniform(10.0, 50.0, size=n),
        "grid_id": np.random.choice([1, 2, 3, 4, 5], size=n),
        "Target": np.random.choice([0, 1], size=n),
    })

    assert is_identifier_column(df, "Id") is True
    assert is_identifier_column(df, "customerID") is True
    assert is_identifier_column(df, "HouseID") is True
    assert is_identifier_column(df, "TransactionID") is True

    assert is_identifier_column(df, "valid_id") is False
    assert is_identifier_column(df, "idea_score") is False
    assert is_identifier_column(df, "fluid_level") is False
    assert is_identifier_column(df, "grid_id") is False

    detected_ids = detect_identifier_columns(df, target_col="Target")
    assert set(detected_ids) == {"Id", "customerID", "HouseID", "TransactionID"}


def test_house_prices_and_telco_churn_id_exclusion():
    """Verify that Kaggle House Prices dataset ('Id') and Telco Churn ('customerID') exclude IDs from feature schema."""
    os.makedirs("data/raw", exist_ok=True)
    if not os.path.exists("data/raw/kaggle_house_prices.csv"):
        from data.generate_dataset import generate_kaggle_house_prices_data
        generate_kaggle_house_prices_data().to_csv("data/raw/kaggle_house_prices.csv", index=False)
    if not os.path.exists("data/raw/telco_churn.csv"):
        from data.generate_dataset import generate_telco_churn_data
        generate_telco_churn_data().to_csv("data/raw/telco_churn.csv", index=False)

    house_df = pd.read_csv("data/raw/kaggle_house_prices.csv")
    telco_df = pd.read_csv("data/raw/telco_churn.csv")

    house_target_options = [c for c in house_df.columns if not is_identifier_column(house_df, c)]
    assert "Id" not in house_target_options
    assert "SalePrice" in house_target_options

    _, _, house_prep, house_names = prepare_data(house_df, fit=True, target_col="SalePrice")
    assert "Id" not in house_prep.feature_cols_
    assert "SalePrice" not in house_prep.feature_cols_
    assert not any(name.startswith(("Id", "SalePrice")) for name in house_names)

    _, _, telco_prep, telco_names = prepare_data(telco_df, fit=True, target_col="Churn")
    assert "customerID" not in telco_prep.feature_cols_
    assert "Churn" not in telco_prep.feature_cols_
    assert not any("customerid" in name.lower() or "churn" in name.lower() for name in telco_names)


def test_dataset_row_vs_manual_input_parity(tmp_path):
    """Verify that dataset-row prediction and manual input prediction produce 100% identical results."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "UserID": [f"USR_{i:04d}" for i in range(n)],
        "tenure": np.random.randint(1, 72, size=n),
        "MonthlyCharges": np.random.uniform(20.0, 120.0, size=n),
        "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], size=n),
        "valid_id": np.random.choice([0, 1], size=n),
        "Churn": np.random.choice(["Yes", "No"], size=n),
    })

    csv_path = tmp_path / "parity_dataset.csv"
    df.to_csv(csv_path, index=False)

    train_and_evaluate(data_path=str(csv_path), target_col="Churn", fast_mode=True)

    model, preprocessor, opt_th, status = load_trained_artifacts()
    assert status == "OK"

    # Pick dataset row 5 (includes UserID and Churn)
    dataset_row = df.iloc[5]

    # Predict via dataset row
    pred_from_row = predict_single_row(dataset_row, model=model, preprocessor=preprocessor, threshold=opt_th)

    # Construct manual feature row (without UserID or Churn, or with a modified UserID)
    manual_row = pd.Series({
        "UserID": "MODIFIED_USR_9999",
        "tenure": dataset_row["tenure"],
        "MonthlyCharges": dataset_row["MonthlyCharges"],
        "Contract": dataset_row["Contract"],
        "valid_id": dataset_row["valid_id"],
    })

    pred_from_manual = predict_single_row(manual_row, model=model, preprocessor=preprocessor, threshold=opt_th)

    # Identical feature values must produce identical predictions regardless of ID
    assert pred_from_row["churn_prediction"] == pred_from_manual["churn_prediction"]
    assert pytest.approx(pred_from_row["churn_probability"], abs=1e-5) == pred_from_manual["churn_probability"]
