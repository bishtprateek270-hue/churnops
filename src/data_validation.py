"""
Data validation module for ChurnOps pipeline.
Supports dataset-agnostic schema, target, and null-value validation.
"""

import numpy as np
import pandas as pd

from src.preprocessing import is_identifier_column


class DataValidationError(Exception):
    """Custom exception raised when data validation checks fail."""


REQUIRED_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges"
]


class DataValidator:
    """Validates schema, missing values, and target presence dynamically."""

    def __init__(
        self,
        is_training: bool = True,
        required_columns: list[str] | None = None,
        target_col: str | None = None,
    ):
        self.is_training = is_training
        self.required_columns = required_columns
        self.target_col = target_col

    def validate(self, df: pd.DataFrame) -> bool:
        """Run all data validation rules on the DataFrame."""
        if df.empty:
            raise DataValidationError("DataFrame is empty.")

        target_name = self._find_target_column(df)
        self._check_schema(df, target_name)
        self._check_null_values(df)
        self._check_numerical_ranges(df)
        self._check_categorical_values(df)

        if self.is_training:
            self._check_target(df, target_name)

        return True

    def _find_target_column(self, df: pd.DataFrame) -> str | None:
        if self.target_col and self.target_col in df.columns:
            return self.target_col
        candidates = ["Churn", "churn", "target", "label", "class", "is_churned", "price", "sale_price", "salary"]
        for c in candidates:
            for col in df.columns:
                if col.lower() == c.lower():
                    return col
        return None

    def _check_schema(self, df: pd.DataFrame, target_name: str | None):
        if self.required_columns:
            missing_cols = [col for col in self.required_columns if col not in df.columns]
            if missing_cols:
                raise DataValidationError(f"Missing required columns in dataset: {missing_cols}")
            return

        feature_cols = [c for c in df.columns if c != target_name and not is_identifier_column(df, c)]
        if len(feature_cols) == 0:
            raise DataValidationError("Dataset must contain at least one feature column.")

    def _check_null_values(self, df: pd.DataFrame):
        total_rows = len(df)
        if total_rows == 0:
            return

        # Ensure no column is 100% empty
        for col in df.columns:
            col_nulls = df[col].replace([np.inf, -np.inf], np.nan).isna().sum()
            if col_nulls == total_rows:
                raise DataValidationError(f"Column '{col}' is completely empty (100% missing values).")

    def _check_numerical_ranges(self, df: pd.DataFrame):
        if "tenure" in df.columns:
            tenure_series = pd.to_numeric(df["tenure"], errors="coerce")
            if (tenure_series < 0).any():
                raise DataValidationError("Column 'tenure' contains values outside range [0, 120].")

    def _check_categorical_values(self, df: pd.DataFrame):
        if self.required_columns and "Contract" in df.columns:
            allowed = ["Month-to-month", "One year", "Two year"]
            unique_vals = set(df["Contract"].dropna().unique())
            invalid = unique_vals - set(allowed)
            if invalid:
                raise DataValidationError(f"Column 'Contract' contains invalid categorical values: {invalid}")

    def _check_target(self, df: pd.DataFrame, target_name: str | None):
        if not target_name or target_name not in df.columns:
            raise DataValidationError("Target column missing from training dataset.")
        valid_targets = df[target_name].replace([np.inf, -np.inf], np.nan).dropna()
        if len(valid_targets) < 2:
            raise DataValidationError(f"Target column '{target_name}' must contain at least 2 non-null observations.")
        if valid_targets.nunique() <= 1:
            raise DataValidationError(f"Target column '{target_name}' is constant (only 1 unique value). Training requires target variance.")


def validate_data(
    df: pd.DataFrame,
    is_training: bool = True,
    required_columns: list[str] | None = None,
    target_col: str | None = None,
) -> bool:
    """Helper function to validate DataFrame."""
    validator = DataValidator(is_training=is_training, required_columns=required_columns, target_col=target_col)
    return validator.validate(df)
