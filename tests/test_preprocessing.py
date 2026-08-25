"""
Unit tests for scikit-learn preprocessing pipeline.
"""

import os

import numpy as np
import pytest

from data.generate_dataset import generate_telco_churn_data
from src.preprocessing import load_preprocessor, prepare_data, save_preprocessor


@pytest.fixture
def sample_data():
    return generate_telco_churn_data(num_samples=100, seed=456)


def test_prepare_data_fit_and_transform(sample_data):
    """Test fitting and transforming dataset yields non-empty numpy arrays without NaNs."""
    X_trans, y, _preprocessor, feature_names = prepare_data(sample_data, fit=True)

    assert isinstance(X_trans, np.ndarray)
    assert X_trans.shape[0] == 100
    assert X_trans.shape[1] > 15  # Expanded one-hot columns
    assert not np.isnan(X_trans).any()
    assert y is not None
    assert set(np.unique(y)).issubset({0, 1})
    assert len(feature_names) == X_trans.shape[1]


def test_prepare_data_transform_only(sample_data):
    """Test transforming new data using an already fitted preprocessor."""
    X_train, _, preprocessor, _ = prepare_data(sample_data[:80], fit=True)
    X_test, _, _, _ = prepare_data(sample_data[80:], preprocessor=preprocessor, fit=False)

    assert isinstance(X_test, np.ndarray)
    assert X_test.shape[0] == 20
    assert X_test.shape[1] == X_train.shape[1]
    assert not np.isnan(X_test).any()


def test_preprocessor_save_and_load(sample_data, tmp_path):
    """Test saving preprocessor to disk and loading it back."""
    _, _, preprocessor, _ = prepare_data(sample_data, fit=True)
    save_path = os.path.join(tmp_path, "test_preprocessor.joblib")

    save_preprocessor(preprocessor, save_path)
    assert os.path.exists(save_path)

    loaded_preproc = load_preprocessor(save_path)
    assert loaded_preproc is not None

    X_test, _, _, _ = prepare_data(sample_data[:10], preprocessor=loaded_preproc, fit=False)
    assert X_test.shape[0] == 10
