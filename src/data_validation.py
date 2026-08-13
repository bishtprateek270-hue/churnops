"""
Data validation module for ChurnOps pipeline.
"""

import pandas as pd


class DataValidationError(Exception):
    """Custom exception raised when data validation checks fail."""


REQUIRED_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges"
]

NUMERICAL_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]

CATEGORICAL_COLUMNS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod"
]

ALLOWED_CATEGORIES: dict[str, list[str]] = {
    "gender": ["Male", "Female"],
    "SeniorCitizen": [0, 1],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["Yes", "No", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["Yes", "No", "No internet service"],
    "OnlineBackup": ["Yes", "No", "No internet service"],
    "DeviceProtection": ["Yes", "No", "No internet service"],
    "TechSupport": ["Yes", "No", "No internet service"],
    "StreamingTV": ["Yes", "No", "No internet service"],
    "StreamingMovies": ["Yes", "No", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ],
}


class DataValidator:
    """Validates schema, value ranges, categorical values, and missing values."""

    def __init__(self, is_training: bool = True):
        self.is_training = is_training

    def validate(self, df: pd.DataFrame) -> bool:
        """Run all data validation rules on the DataFrame."""
        if df.empty:
            raise DataValidationError("DataFrame is empty.")

        self._check_schema(df)
        self._check_null_values(df)
        self._check_numerical_ranges(df)
        self._check_categorical_values(df)

        if self.is_training and "Churn" in df.columns:
            self._check_target(df)

        return True

    def _check_schema(self, df: pd.DataFrame):
        missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            raise DataValidationError(f"Missing required columns in dataset: {missing_cols}")

    def _check_null_values(self, df: pd.DataFrame):
        # Allow small percentage of nulls (handled by imputer), but fail if excessive nulls (>10%)
        total_rows = len(df)
        for col in REQUIRED_COLUMNS:
            col_nulls = df[col].isna().sum()
            if col_nulls / total_rows > 0.10:
                raise DataValidationError(
                    f"Column '{col}' exceeds maximum allowed null ratio (10%): {col_nulls}/{total_rows}"
                )

    def _check_numerical_ranges(self, df: pd.DataFrame):
        if (df["tenure"] < 0).any() or (df["tenure"] > 120).any():
            raise DataValidationError("Column 'tenure' contains values outside range [0, 120].")

        if (df["MonthlyCharges"] < 0).any() or (df["MonthlyCharges"] > 300).any():
            raise DataValidationError("Column 'MonthlyCharges' contains values outside range [0, 300].")

        # Coerce TotalCharges for range check
        tc = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
        if (tc < 0).any():
            raise DataValidationError("Column 'TotalCharges' contains negative values.")

        if not set(df["SeniorCitizen"].unique()).issubset({0, 1}):
            raise DataValidationError("Column 'SeniorCitizen' must only contain 0 or 1.")

    def _check_categorical_values(self, df: pd.DataFrame):
        for col, allowed in ALLOWED_CATEGORIES.items():
            if col in df.columns and col != "SeniorCitizen":
                unique_vals = set(df[col].dropna().unique())
                invalid = unique_vals - set(allowed)
                if invalid:
                    raise DataValidationError(
                        f"Column '{col}' contains invalid categorical values: {invalid}. Allowed: {allowed}"
                    )

    def _check_target(self, df: pd.DataFrame):
        if "Churn" not in df.columns:
            raise DataValidationError("Target column 'Churn' missing from training dataset.")
        unique_targets = set(df["Churn"].unique())
        valid_targets = {"Yes", "No", 0, 1}
        if not unique_targets.issubset(valid_targets):
            raise DataValidationError(f"Target column 'Churn' contains invalid values: {unique_targets}")


def validate_data(df: pd.DataFrame, is_training: bool = True) -> bool:
    """Helper function to validate DataFrame."""
    validator = DataValidator(is_training=is_training)
    return validator.validate(df)
