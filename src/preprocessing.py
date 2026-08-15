"""
Dataset-agnostic preprocessing and leak-free feature engineering module for ChurnOps pipeline.
Supports both Classification and Regression tasks with robust missing value and infinity handling.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

__all__ = [
    "ChurnFeatureEngineer",
    "GenericFeatureEngineer",
    "build_preprocessor",
    "clean_dataframe",
    "find_target_col",
    "get_feature_names",
    "infer_task_type",
    "load_preprocessor",
    "prepare_data",
    "save_preprocessor",
]


def infer_task_type(target_series: pd.Series) -> str:
    """Infer whether task is 'classification' or 'regression' based on target dtype and unique value counts."""
    valid_series = target_series.replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid_series) == 0:
        return "classification"

    dtype_str = str(valid_series.dtype).lower()
    num_unique = valid_series.nunique()

    # Object, string, category, bool dtypes -> classification
    if dtype_str in ["object", "string", "category", "bool"] or pd.api.types.is_bool_dtype(valid_series):
        return "classification"

    # Discrete integers or floats with few unique values -> classification
    if num_unique <= 20 and (num_unique / len(valid_series)) < 0.1:
        return "classification"

    # Continuous numerical floats or high-cardinality numbers -> regression
    return "regression"


class GenericFeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom scikit-learn transformer for dataset-agnostic leak-free feature engineering."""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()
        if not isinstance(X_out, pd.DataFrame):
            return X_out

        num_cols = X_out.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = X_out.select_dtypes(exclude=[np.number]).columns.tolist()

        # 1. Numerical interaction features (ratios between top numerical pairs)
        if len(num_cols) >= 2:
            col1, col2 = num_cols[0], num_cols[1]
            val1 = pd.to_numeric(X_out[col1], errors="coerce").fillna(0.0)
            val2 = pd.to_numeric(X_out[col2], errors="coerce").fillna(1e-5)
            denom = np.where(val2.abs() < 1e-5, 1e-5, val2)
            ratio = val1 / denom
            ratio = np.where(np.isinf(ratio), np.nan, ratio)
            X_out[f"{col1}_per_{col2}"] = ratio

        # 2. Total active non-null categorical services/features count
        if cat_cols:
            non_null_counts = pd.Series(0.0, index=X_out.index)
            for c in cat_cols:
                is_valid = X_out[c].notna() & (~X_out[c].astype(str).str.strip().str.lower().isin(["no", "none", "false", "0", "nan", ""]))
                non_null_counts += np.where(is_valid, 1.0, 0.0)
            X_out["active_cat_features_count"] = non_null_counts

        return X_out


import __main__

__main__.GenericFeatureEngineer = GenericFeatureEngineer
__main__.ChurnFeatureEngineer = GenericFeatureEngineer
ChurnFeatureEngineer = GenericFeatureEngineer


def build_preprocessor(
    num_cols: list[str] | None = None,
    cat_cols: list[str] | None = None
) -> ColumnTransformer:
    """Build ColumnTransformer for numerical scaling and categorical one-hot encoding with unknown handling."""
    num_cols = num_cols or []
    cat_cols = cat_cols or []

    transformers = []
    if num_cols:
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        transformers.append(("num", num_pipeline, num_cols))

    if cat_cols:
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        transformers.append(("cat", cat_pipeline, cat_cols))

    preprocessor = ColumnTransformer(transformers=transformers)
    return preprocessor


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean DataFrame types, convert inf/-inf to NaN, and standardize missing string representations."""
    df_clean = df.copy()

    # Convert inf/-inf to NaN across entire dataframe
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)

    missing_str_regex = r"^\s*(nan|none|null|n/a|na|inf|-inf|<na>|\s*)\s*$"
    for col in df_clean.columns:
        if df_clean[col].dtype == object or str(df_clean[col].dtype) == "string":
            df_clean[col] = df_clean[col].replace(missing_str_regex, np.nan, regex=True)
            # Safely coerce numeric strings to float where possible without astype(int)
            coerced = pd.to_numeric(df_clean[col], errors="coerce")
            if coerced.notna().sum() > 0 and (coerced.notna().mean() > 0.8):
                df_clean[col] = coerced

    return df_clean


def find_target_col(df: pd.DataFrame, target_col: str | None = None) -> str | None:
    """Identify the target column in DataFrame."""
    if target_col and target_col in df.columns:
        return target_col
    candidates = ["Churn", "churn", "target", "label", "class", "is_churned", "price", "sale_price", "salary"]
    for c in candidates:
        for col in df.columns:
            if col.lower() == c.lower():
                return col
    return None


def prepare_data(
    df: pd.DataFrame,
    preprocessor: ColumnTransformer | Pipeline | None = None,
    fit: bool = True,
    target_col: str | None = None,
) -> tuple[np.ndarray, np.ndarray | None, object, list[str]]:
    """Clean, engineer features, preprocess data, and extract target variable safely."""
    df_clean = clean_dataframe(df)

    found_target = find_target_col(df_clean, target_col)
    y = None
    task_type = "classification"

    if found_target and found_target in df_clean.columns:
        task_type = infer_task_type(df_clean[found_target])
        if fit:
            target_str = df_clean[found_target].astype(str).str.strip().str.lower()
            invalid_terms = {"nan", "none", "null", "n/a", "na", "inf", "-inf", "<na>", ""}
            valid_target_mask = df_clean[found_target].notna() & (~target_str.isin(invalid_terms))
            if not valid_target_mask.all():
                df_clean = df_clean[valid_target_mask].reset_index(drop=True)

        target_series = df_clean[found_target]
        if task_type == "classification":
            if target_series.dtype == object or str(target_series.dtype) in ["string", "category", "bool"]:
                str_vals = target_series.astype(str).str.strip().str.lower()
                positive_indicators = {"yes", "true", "1", "1.0", "churn", "churned", "positive", "y", "t"}
                unique_str = set(str_vals.unique())
                if any(val in positive_indicators for val in unique_str):
                    y = np.where(str_vals.isin(positive_indicators), 1, 0)
                else:
                    labels, _ = pd.factorize(target_series)
                    y = np.where(labels < 0, 0, labels)
            else:
                num_vals = pd.to_numeric(target_series, errors="coerce").fillna(0.0)
                unique_nums = set(np.unique(num_vals.values))
                if unique_nums.issubset({0, 1}):
                    y = np.where(num_vals.values > 0.5, 1, 0)
                else:
                    y = np.where(num_vals.values > np.median(num_vals.values), 1, 0)
            y = np.asarray(y, dtype=np.int64)
        else:
            # Continuous Regression
            num_vals = pd.to_numeric(target_series, errors="coerce").fillna(0.0)
            y = np.asarray(num_vals.values, dtype=np.float64)

    # Drop target and common ID columns
    drop_cols = []
    if found_target:
        drop_cols.append(found_target)
    for col in df_clean.columns:
        col_lower = col.lower()
        if (col_lower in ["customerid", "id", "index", "rownumber", "user_id"] or col_lower.endswith("_id") or (df_clean[col].dtype == object or str(df_clean[col].dtype) == "string") and df_clean[col].nunique() == len(df_clean) and len(df_clean) > 20) and col not in drop_cols:
            drop_cols.append(col)

    X_df = df_clean.drop(columns=drop_cols, errors="ignore")

    if fit:
        engineer = GenericFeatureEngineer()
        X_engineered = engineer.transform(X_df)
        num_cols = X_engineered.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = X_engineered.select_dtypes(exclude=[np.number]).columns.tolist()

        column_trans = build_preprocessor(num_cols=num_cols, cat_cols=cat_cols)
        pipeline = Pipeline([
            ("feature_engineer", GenericFeatureEngineer()),
            ("column_transformer", column_trans)
        ])
        pipeline.fit(X_df)
        X_trans = pipeline.transform(X_df)

        preprocessor = pipeline
        preprocessor.feature_cols_ = list(X_df.columns)
        preprocessor.num_cols_ = num_cols
        preprocessor.cat_cols_ = cat_cols
        preprocessor.target_col_ = found_target
        preprocessor.task_type_ = task_type
    else:
        feature_cols = getattr(preprocessor, "feature_cols_", None)
        if feature_cols:
            aligned_df = pd.DataFrame(index=X_df.index)
            for col in feature_cols:
                if col in X_df.columns:
                    aligned_df[col] = X_df[col]
                else:
                    aligned_df[col] = np.nan
            X_df = aligned_df

        if isinstance(preprocessor, Pipeline):
            X_trans = preprocessor.transform(X_df)
        elif hasattr(preprocessor, "transform"):
            engineer = GenericFeatureEngineer()
            X_engineered = engineer.transform(X_df)
            X_trans = preprocessor.transform(X_engineered)
        else:
            raise ValueError("Invalid preprocessor pipeline provided.")

    feature_names = get_feature_names(preprocessor)
    return X_trans, y, preprocessor, feature_names


def get_feature_names(preprocessor: object) -> list[str]:
    """Retrieve output feature names after transformation."""
    feature_names = []
    col_trans = preprocessor
    if isinstance(preprocessor, Pipeline):
        col_trans = preprocessor.named_steps.get("column_transformer", preprocessor)

    if hasattr(col_trans, "transformers_"):
        for name, trans, cols in col_trans.transformers_:
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


def save_preprocessor(preprocessor: object, filepath: str = "models/preprocessor.joblib"):
    """Save fitted preprocessor pipeline to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(preprocessor, filepath)


def load_preprocessor(filepath: str = "models/preprocessor.joblib") -> object:
    """Load fitted preprocessor pipeline from disk."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Preprocessor artifact not found at {filepath}")
    return joblib.load(filepath)
