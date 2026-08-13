"""
Unit tests for data validation logic.
"""

import pytest
import pandas as pd
from data.generate_dataset import generate_telco_churn_data
from src.data_validation import DataValidator, DataValidationError, validate_data


@pytest.fixture
def valid_sample_df():
    df = generate_telco_churn_data(num_samples=50, seed=123)
    return df


def test_data_validation_success(valid_sample_df):
    """Test that a valid dataframe passes validation."""
    assert validate_data(valid_sample_df, is_training=True) is True


def test_missing_column_raises_error(valid_sample_df):
    """Test that missing required columns raise DataValidationError."""
    df_missing = valid_sample_df.drop(columns=["MonthlyCharges"])
    with pytest.raises(DataValidationError, match="Missing required columns"):
        validate_data(df_missing, is_training=True)


def test_invalid_numerical_range(valid_sample_df):
    """Test that negative tenure raises DataValidationError."""
    df_invalid = valid_sample_df.copy()
    df_invalid.loc[0, "tenure"] = -10
    with pytest.raises(DataValidationError, match="outside range"):
        validate_data(df_invalid, is_training=True)


def test_invalid_categorical_value(valid_sample_df):
    """Test that invalid categorical options raise DataValidationError."""
    df_invalid = valid_sample_df.copy()
    df_invalid.loc[0, "Contract"] = "Super Long Term"
    with pytest.raises(DataValidationError, match="invalid categorical values"):
        validate_data(df_invalid, is_training=True)


def test_empty_dataframe_raises_error():
    """Test that an empty DataFrame raises DataValidationError."""
    df_empty = pd.DataFrame()
    with pytest.raises(DataValidationError, match="empty"):
        validate_data(df_empty, is_training=False)
