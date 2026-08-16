"""
Dataset-agnostic preprocessing and leak-free feature engineering module for ChurnOps pipeline.
Supports both Classification and Regression tasks with robust missing value and infinity handling.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import re

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
    "detect_identifier_columns",
    "find_target_col",
    "get_feature_names",
    "infer_task_type",
    "is_identifier_column",
    "load_preprocessor",
    "prepare_data",
    "save_preprocessor",
]


def is_identifier_column(df: pd.DataFrame, col: str) -> bool:
    """Determine if a column is a true identifier based on column name and high uniqueness/monotonicity heuristics."""
    s = df[col].dropna()
    total_rows = len(df)
    if total_rows == 0 or len(s) == 0:
        return False

    num_unique = s.nunique()
    uniqueness_ratio = num_unique / total_rows
    col_lower = col.lower().strip()

    # Common false positive words containing "id" substring that are valid features
    false_positives = {
        "fluid", "solid", "liquid", "hybrid", "valid", "invalid", "grid",
        "pyramid", "squid", "asteroid", "android", "centroid", "humanoid", "orchid", "idea", "ideology"
    }
    if col_lower in false_positives:
        return False

    # Check if column name strongly indicates an identifier token/suffix
    known_id_names = {
        "id", "id_num", "uuid", "hash", "key", "guid", "index", "rownumber",
        "row_num", "row_id", "record_id", "customerid", "userid", "transactionid",
        "houseid", "buildingid", "orderid", "patientid", "accountid", "memberid",
        "clientid", "subjectid", "itemid", "productid", "sessionid", "sub_id"
    }

    is_id_name = (
        col_lower in known_id_names
        or col_lower.endswith(("_id", "-id", "_key", "_uuid", "_hash"))
        or col_lower.startswith(("id_", "id-", "uuid_"))
        or bool(re.search(r'(^|_|-|[a-z])(id|uuid|guid|hash|key)s?$', col_lower))
    )

    # Monotonicity check for numerical series
    is_monotonic = False
    if pd.api.types.is_numeric_dtype(s) and len(s) > 1:
        diffs = s.diff().dropna()
        if len(diffs) > 0:
            is_monotonic = bool((diffs > 0).all() or (diffs < 0).all() or (diffs == 1).all())

    # Decision heuristics:
    if is_id_name:
        if total_rows <= 10:
            return num_unique == total_rows
        # Name indicates ID: confirm with moderate/high uniqueness or monotonicity
        return (uniqueness_ratio > 0.50) or is_monotonic or (num_unique == total_rows)
    else:
        # High uniqueness/monotonicity for un-named string keys or primary key ints (>20 rows)
        if (s.dtype == object or str(s.dtype) in ["string", "category"]) and total_rows > 20:
            return uniqueness_ratio > 0.95
        if pd.api.types.is_integer_dtype(s) and total_rows > 50:
            return is_monotonic and (uniqueness_ratio > 0.90)

    return False


def detect_identifier_columns(df: pd.DataFrame, target_col: str | None = None) -> list[str]:
    """Detect all true identifier columns in DataFrame to exclude them from ML training/inference."""
    id_cols = []
    for col in df.columns:
        if target_col and col == target_col:
            continue
        if is_identifier_column(df, col):
            id_cols.append(col)
    return id_cols


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
        self.interaction_pair_ = None
        self.has_cat_cols_ = False
        self.cat_cols_ = []

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
            self.interaction_pair_ = (num_cols[0], num_cols[1]) if len(num_cols) >= 2 else None
            self.has_cat_cols_ = bool(cat_cols)
            self.cat_cols_ = cat_cols
        return self

    def transform(self, X):
        X_out = X.copy()
        if not isinstance(X_out, pd.DataFrame):
            return X_out

        # 1. Numerical interaction features (ratios between top numerical pairs)
        if hasattr(self, "interaction_pair_") and self.interaction_pair_:
            col1, col2 = self.interaction_pair_
            ratio_name = f"{col1}_per_{col2}"
            if col1 in X_out.columns and col2 in X_out.columns:
                val1 = pd.to_numeric(X_out[col1], errors="coerce").fillna(0.0)
                val2 = pd.to_numeric(X_out[col2], errors="coerce").fillna(1e-5)
                denom = np.where(val2.abs() < 1e-5, 1e-5, val2)
                ratio = val1 / denom
                ratio = np.where(np.isinf(ratio), np.nan, ratio)
                X_out[ratio_name] = ratio
            else:
                X_out[ratio_name] = np.nan
        else:
            num_cols = X_out.select_dtypes(include=[np.number]).columns.tolist()
            if len(num_cols) >= 2:
                col1, col2 = num_cols[0], num_cols[1]
                val1 = pd.to_numeric(X_out[col1], errors="coerce").fillna(0.0)
                val2 = pd.to_numeric(X_out[col2], errors="coerce").fillna(1e-5)
                denom = np.where(val2.abs() < 1e-5, 1e-5, val2)
                ratio = val1 / denom
                ratio = np.where(np.isinf(ratio), np.nan, ratio)
                X_out[f"{col1}_per_{col2}"] = ratio

        # 2. Total active non-null categorical services/features count
        has_cat = getattr(self, "has_cat_cols_", False)
        known_cats = getattr(self, "cat_cols_", [])
        cat_cols_in_x = X_out.select_dtypes(exclude=[np.number]).columns.tolist()

        if has_cat or cat_cols_in_x or known_cats:
            non_null_counts = pd.Series(0.0, index=X_out.index)
            target_cats = known_cats if known_cats else cat_cols_in_x
            for c in target_cats:
                if c in X_out.columns:
                    is_valid = X_out[c].notna() & (~X_out[c].astype(str).str.strip().str.lower().isin(["no", "none", "false", "0", "nan", ""]))
                    non_null_counts += np.where(is_valid, 1.0, 0.0)
            X_out["active_cat_features_count"] = non_null_counts

        return X_out


import __main__

__main__.GenericFeatureEngineer = GenericFeatureEngineer
__main__.ChurnFeatureEngineer = GenericFeatureEngineer
ChurnFeatureEngineer = GenericFeatureEngineer


from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import OrdinalEncoder

__all__ = [
    "ChurnFeatureEngineer",
    "GenericFeatureEngineer",
    "build_preprocessor",
    "clean_dataframe",
    "detect_identifier_columns",
    "detect_target_leakage",
    "find_target_col",
    "get_feature_names",
    "infer_task_type",
    "is_identifier_column",
    "load_preprocessor",
    "prepare_data",
    "save_preprocessor",
]


def detect_target_leakage(df_train: pd.DataFrame, y_train: np.ndarray | None, threshold: float = 0.98) -> list[str]:
    """Detect columns in training data with near-perfect correlation (> 0.98) to target variable."""
    leakage_cols = []
    if y_train is None or len(df_train) == 0:
        return leakage_cols

    y_series = pd.Series(y_train, index=df_train.index)
    for col in df_train.columns:
        if pd.api.types.is_numeric_dtype(df_train[col]):
            corr = df_train[col].corr(y_series)
            if not np.isnan(corr) and abs(corr) >= threshold:
                leakage_cols.append(col)

    return leakage_cols


def build_preprocessor(
    num_cols: list[str] | None = None,
    cat_low_cols: list[str] | None = None,
    cat_high_cols: list[str] | None = None,
    cat_cols: list[str] | None = None,
) -> ColumnTransformer:
    """Build ColumnTransformer for numerical scaling, low-cardinality one-hot encoding, and high-cardinality ordinal encoding."""
    num_cols = num_cols or []
    cat_low_cols = cat_low_cols or []
    cat_high_cols = cat_high_cols or []
    if cat_cols and not cat_low_cols and not cat_high_cols:
        cat_low_cols = cat_cols

    transformers = []
    if num_cols:
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        transformers.append(("num", num_pipeline, num_cols))

    if cat_low_cols:
        cat_low_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        transformers.append(("cat_low", cat_low_pipeline, cat_low_cols))

    if cat_high_cols:
        cat_high_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
        ])
        transformers.append(("cat_high", cat_high_pipeline, cat_high_cols))

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
    """Identify the target column in DataFrame, prioritizing explicit candidates and avoiding true identifiers."""
    if target_col and target_col in df.columns:
        return target_col

    candidates = [
        "churn", "target", "label", "class", "is_churned", "exited", "survived",
        "outcome", "response", "price", "sale_price", "saleprice", "salary", "value",
        "target_class", "target_value"
    ]
    for c in candidates:
        for col in df.columns:
            if col.lower() == c.lower():
                return col

    # Fallback: Prefer the last column that is NOT an identifier column
    non_id_cols = [c for c in df.columns if not is_identifier_column(df, c)]
    if non_id_cols:
        return non_id_cols[-1]

    return df.columns[-1] if len(df.columns) > 0 else None


def prepare_data(
    df: pd.DataFrame,
    preprocessor: ColumnTransformer | Pipeline | None = None,
    fit: bool = True,
    target_col: str | None = None,
) -> tuple[np.ndarray, np.ndarray | None, object, list[str]]:
    """Clean, engineer features, preprocess data, and extract target variable safely."""
    df_clean = clean_dataframe(df)

    if target_col and target_col in df_clean.columns:
        found_target = target_col
    elif fit:
        found_target = find_target_col(df_clean, target_col=target_col)
    else:
        found_target = None
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

    # Drop target and true identifier columns
    drop_cols = []
    if found_target:
        drop_cols.append(found_target)

    if fit:
        id_cols = detect_identifier_columns(df_clean, target_col=found_target)
        for col in id_cols:
            if col not in drop_cols:
                drop_cols.append(col)

        X_df = df_clean.drop(columns=drop_cols, errors="ignore")

        leakage_cols = detect_target_leakage(X_df, y)
        if leakage_cols:
            X_df = X_df.drop(columns=leakage_cols, errors="ignore")

        engineer = GenericFeatureEngineer()
        X_engineered = engineer.transform(X_df)
        num_cols = X_engineered.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = X_engineered.select_dtypes(exclude=[np.number]).columns.tolist()

        cat_low_cols = [c for c in cat_cols if X_engineered[c].nunique() <= 20]
        cat_high_cols = [c for c in cat_cols if X_engineered[c].nunique() > 20]

        column_trans = build_preprocessor(num_cols=num_cols, cat_low_cols=cat_low_cols, cat_high_cols=cat_high_cols)
        pipeline = Pipeline([
            ("feature_engineer", GenericFeatureEngineer()),
            ("column_transformer", column_trans),
            ("variance_selector", VarianceThreshold(threshold=0.0))
        ])
        pipeline.fit(X_df)
        X_trans = pipeline.transform(X_df)

        preprocessor = pipeline
        preprocessor.feature_cols_ = list(X_df.columns)
        preprocessor.id_cols_ = id_cols
        preprocessor.leakage_cols_ = leakage_cols
        preprocessor.num_cols_ = num_cols
        preprocessor.cat_low_cols_ = cat_low_cols
        preprocessor.cat_high_cols_ = cat_high_cols
        preprocessor.target_col_ = found_target
        preprocessor.task_type_ = task_type
        preprocessor.feature_names_ = get_feature_names(preprocessor)
    else:
        feature_cols = getattr(preprocessor, "feature_cols_", None)
        id_cols = getattr(preprocessor, "id_cols_", [])
        leakage_cols = getattr(preprocessor, "leakage_cols_", [])
        stored_target = getattr(preprocessor, "target_col_", None)

        # Drop target, ID, and leakage columns in inference mode
        for col in df_clean.columns:
            if ((found_target and col == found_target) or (stored_target and col == stored_target)) and col not in drop_cols or (col in id_cols or col in leakage_cols or is_identifier_column(df_clean, col)) and col not in drop_cols:
                drop_cols.append(col)

        X_df = df_clean.drop(columns=drop_cols, errors="ignore")

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

    feature_names = getattr(preprocessor, "feature_names_", None) or get_feature_names(preprocessor)
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
