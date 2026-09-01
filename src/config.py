"""
Centralized Configuration Module for ChurnOps.
Loads environment variables (via dotenv if present) and exposes typed settings.
"""

import os
from dataclasses import dataclass, field

# Load environment variables from .env file if available
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


@dataclass
class Settings:
    """Application Settings with environment variable overrides and sensible defaults."""

    # MLflow & Model Registry
    MLFLOW_TRACKING_URI: str = field(default_factory=lambda: os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
    MLFLOW_EXPERIMENT_NAME: str = field(
        default_factory=lambda: os.getenv("MLFLOW_EXPERIMENT_NAME", "ChurnOps_Churn_Prediction")
    )
    MODEL_NAME: str = field(default_factory=lambda: os.getenv("MODEL_NAME", "ChurnOps-Model"))

    # File System & Artifact Paths
    DATA_RAW_DIR: str = field(default_factory=lambda: os.getenv("DATA_RAW_DIR", "data/raw"))
    DEFAULT_DATA_PATH: str = field(default_factory=lambda: os.getenv("DEFAULT_DATA_PATH", "data/raw/telco_churn.csv"))
    UPLOAD_DATA_PATH: str = field(default_factory=lambda: os.getenv("UPLOAD_DATA_PATH", "data/raw/user_upload.csv"))
    MODELS_DIR: str = field(default_factory=lambda: os.getenv("MODELS_DIR", "models"))
    BEST_MODEL_PATH: str = field(default_factory=lambda: os.getenv("BEST_MODEL_PATH", "models/best_model.joblib"))
    PREPROCESSOR_PATH: str = field(default_factory=lambda: os.getenv("PREPROCESSOR_PATH", "models/preprocessor.joblib"))
    UNIFIED_PIPELINE_PATH: str = field(
        default_factory=lambda: os.getenv("UNIFIED_PIPELINE_PATH", "models/unified_pipeline.joblib")
    )
    PREDICTIONS_DB_PATH: str = field(
        default_factory=lambda: os.getenv("PREDICTIONS_DB_PATH", "monitoring/predictions.db")
    )
    REPORTS_PLOTS_DIR: str = field(default_factory=lambda: os.getenv("REPORTS_PLOTS_DIR", "reports/plots"))

    # Hyperparameters & Split Defaults
    TEST_SIZE: float = field(default_factory=lambda: float(os.getenv("TEST_SIZE", "0.2")))
    RANDOM_STATE: int = field(default_factory=lambda: int(os.getenv("RANDOM_STATE", "42")))
    DEFAULT_THRESHOLD: float = field(default_factory=lambda: float(os.getenv("DEFAULT_THRESHOLD", "0.5")))
    FAST_MODE_TRIALS: int = field(default_factory=lambda: int(os.getenv("FAST_MODE_TRIALS", "5")))
    FULL_MODE_TRIALS: int = field(default_factory=lambda: int(os.getenv("FULL_MODE_TRIALS", "15")))

    # Business Cost Optimization Weights
    COST_FN: float = field(default_factory=lambda: float(os.getenv("COST_FN", "500.0")))
    COST_FP: float = field(default_factory=lambda: float(os.getenv("COST_FP", "50.0")))

    # Monitoring & Drift Thresholds
    DRIFT_P_VALUE_THRESHOLD: float = field(default_factory=lambda: float(os.getenv("DRIFT_P_VALUE_THRESHOLD", "0.05")))
    DRIFT_PSI_THRESHOLD: float = field(default_factory=lambda: float(os.getenv("DRIFT_PSI_THRESHOLD", "0.2")))

    # FastAPI Serving Configuration
    API_HOST: str = field(default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"))
    API_PORT: int = field(default_factory=lambda: int(os.getenv("API_PORT", "8000")))
    API_KEY_HEADER: str = field(default_factory=lambda: os.getenv("API_KEY_HEADER", "X-API-Key"))
    RATE_LIMIT_PER_MINUTE: int = field(default_factory=lambda: int(os.getenv("RATE_LIMIT_PER_MINUTE", "100")))
    ENABLE_CORS: bool = field(default_factory=lambda: os.getenv("ENABLE_CORS", "true").lower() == "true")
    CORS_ORIGINS: list[str] = field(default_factory=lambda: os.getenv("CORS_ORIGINS", "*").split(","))


settings = Settings()
