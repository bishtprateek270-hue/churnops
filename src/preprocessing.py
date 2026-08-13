"""
Preprocessing module for ChurnOps pipeline.
"""

import os
from typing import Tuple, List, Dict, Any, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_validation import NUMERICAL_COLUMNS, CATEGORICAL_COLUMNS, CATEGORICAL_COLUMNS, ALLOWED_CATEGORIES


def build_preprocessor() -> ColumnTransformer:
    """Build scikit-learn ColumnTransformer for numerical scaling and categorical one-hot encoding."""
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
    cat_cols = [
        "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
        "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
        "PaperlessBilling", "PaymentMethod"
    ]

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols)
        ]
    )

    return preprocessor


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean DataFrame types before passing to pipeline."""
    df_clean = df.copy()
    if "TotalCharges" in df_clean.columns:
        df_clean["TotalCharges"] = pd.to_numeric(df_clean["TotalCharges"].replace(" ", np.nan), errors="coerce")
    return df_clean


def prepare_data(
    df: pd.DataFrame,
    preprocessor: Optional[ColumnTransformer] = None,
    fit: bool = True
) -> Tuple[np.ndarray, Optional[np.ndarray], ColumnTransformer, List[str]]:
    """Clean, preprocess data, and optionally extract target variable."""
    df_clean = clean_dataframe(df)

    y = None
    if "Churn" in df_clean.columns:
        target_series = df_clean["Churn"]
        if target_series.dtype == object:
            y = (target_series.str.lower() == "yes").astype(int).values
        else:
            y = target_series.astype(int).values
        X_df = df_clean.drop(columns=["Churn", "customerID"], errors="ignore")
    else:
        X_df = df_clean.drop(columns=["customerID"], errors="ignore")

    if preprocessor is None:
        preprocessor = build_preprocessor()

    if fit:
        X_trans = preprocessor.fit_transform(X_df)
    else:
        X_trans = preprocessor.transform(X_df)

    feature_names = get_feature_names(preprocessor)
    return X_trans, y, preprocessor, feature_names


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """Retrieve output feature names after transformation."""
    feature_names = []
    for name, trans, cols in preprocessor.transformers_:
        if trans == "drop" or trans is None:
            continue
        if hasattr(trans, "get_feature_names_out"):
            names = trans.get_feature_names_out(cols)
        elif hasattr(trans, "named_steps"):
            last_step = list(trans.named_steps.values())[-1]
            if hasattr(last_step, "get_feature_names_out"):
                names = last_step.get_feature_names_out(cols)
            else:
                names = cols
        else:
            names = cols
        feature_names.extend(list(names))
    return feature_names


def save_preprocessor(preprocessor: ColumnTransformer, filepath: str = "models/preprocessor.joblib"):
    """Save fitted preprocessor pipeline to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(preprocessor, filepath)
    print(f"Saved preprocessor to {filepath}")


def load_preprocessor(filepath: str = "models/preprocessor.joblib") -> ColumnTransformer:
    """Load fitted preprocessor pipeline from disk."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Preprocessor artifact not found at {filepath}")
    return joblib.load(filepath)
