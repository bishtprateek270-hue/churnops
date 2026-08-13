"""
FastAPI Serving API for ChurnOps customer churn prediction.
"""

import os
import sys

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

# Ensure workspace root directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import joblib
import mlflow
import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException, status

from api.schemas import ChurnInput, ChurnOutput, HealthResponse
from src.data_validation import DataValidationError, validate_data
from src.preprocessing import load_preprocessor, prepare_data

# Configuration
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
MODEL_NAME = "ChurnOps-Model"
DB_PATH = os.getenv("PREDICTIONS_DB_PATH", "monitoring/predictions.db")

# Global state loaded during lifespan
model_store = {
    "model": None,
    "preprocessor": None,
    "version": "unknown",
    "stage": "none"
}


def init_sqlite_db():
    """Initialize SQLite database for prediction logging and drift monitoring."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            gender TEXT,
            SeniorCitizen INTEGER,
            Partner TEXT,
            Dependents TEXT,
            tenure INTEGER,
            PhoneService TEXT,
            MultipleLines TEXT,
            InternetService TEXT,
            OnlineSecurity TEXT,
            OnlineBackup TEXT,
            DeviceProtection TEXT,
            TechSupport TEXT,
            StreamingTV TEXT,
            StreamingMovies TEXT,
            Contract TEXT,
            PaperlessBilling TEXT,
            PaymentMethod TEXT,
            MonthlyCharges REAL,
            TotalCharges REAL,
            churn_prediction INTEGER,
            churn_probability REAL,
            model_version TEXT
        )
    """)
    conn.commit()
    conn.close()


def log_prediction_to_db(input_dict: dict, prediction: int, probability: float, model_version: str):
    """Log prediction request and model result to SQLite database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        now_iso = datetime.now(timezone.utc).isoformat()
        
        cursor.execute("""
            INSERT INTO predictions (
                timestamp, gender, SeniorCitizen, Partner, Dependents, tenure,
                PhoneService, MultipleLines, InternetService, OnlineSecurity,
                OnlineBackup, DeviceProtection, TechSupport, StreamingTV,
                StreamingMovies, Contract, PaperlessBilling, PaymentMethod,
                MonthlyCharges, TotalCharges, churn_prediction, churn_probability, model_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            now_iso,
            input_dict.get("gender"),
            input_dict.get("SeniorCitizen"),
            input_dict.get("Partner"),
            input_dict.get("Dependents"),
            input_dict.get("tenure"),
            input_dict.get("PhoneService"),
            input_dict.get("MultipleLines"),
            input_dict.get("InternetService"),
            input_dict.get("OnlineSecurity"),
            input_dict.get("OnlineBackup"),
            input_dict.get("DeviceProtection"),
            input_dict.get("TechSupport"),
            input_dict.get("StreamingTV"),
            input_dict.get("StreamingMovies"),
            input_dict.get("Contract"),
            input_dict.get("PaperlessBilling"),
            input_dict.get("PaymentMethod"),
            input_dict.get("MonthlyCharges"),
            input_dict.get("TotalCharges"),
            prediction,
            probability,
            model_version
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Warning: Failed to log prediction to SQLite database: {e}")


def load_model_and_preprocessor():
    """Load model from MLflow Registry ('Production' or 'Staging') or fallback local file."""
    # 1. Load preprocessor
    try:
        model_store["preprocessor"] = load_preprocessor("models/preprocessor.joblib")
        print("Successfully loaded preprocessor artifact.")
    except Exception as e:
        print(f"Warning: Could not load preprocessor from disk: {e}")

    # 2. Load model from MLflow Registry
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = mlflow.tracking.MlflowClient()

    for stage in ["Production", "Staging"]:
        try:
            versions = client.get_latest_versions(MODEL_NAME, stages=[stage])
            if versions:
                version = versions[0].version
                model_uri = f"models:/{MODEL_NAME}/{stage}"
                print(f"Loading MLflow model '{MODEL_NAME}' stage '{stage}' version {version}...")
                model_store["model"] = mlflow.pyfunc.load_model(model_uri)
                model_store["version"] = str(version)
                model_store["stage"] = stage
                return
        except Exception as e:
            print(f"Notice: Could not load model from MLflow stage '{stage}': {e}")

    # Fallback to local joblib file
    fallback_path = "models/best_model.joblib"
    if os.path.exists(fallback_path):
        print(f"Loading fallback model from local file {fallback_path}...")
        model_store["model"] = joblib.load(fallback_path)
        model_store["version"] = "local-1.0"
        model_store["stage"] = "LocalFallback"
    else:
        print("Warning: No model found in MLflow Registry or local directory.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    init_sqlite_db()
    load_model_and_preprocessor()
    yield
    # Shutdown logic


app = FastAPI(
    title="ChurnOps Customer Churn Prediction API",
    description="Production-style FastAPI serving endpoint for Telco Customer Churn Prediction.",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="healthy" if model_store["model"] is not None else "degraded",
        model_name=MODEL_NAME,
        model_stage=model_store["stage"],
        model_version=model_store["version"],
        timestamp=datetime.now(timezone.utc).isoformat()
    )


@app.post("/predict", response_model=ChurnOutput)
def predict_churn(payload: ChurnInput):
    if model_store["model"] is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not currently loaded."
        )

    # Convert payload to DataFrame
    input_dict = payload.model_dump()
    df_input = pd.DataFrame([input_dict])

    # Validate data rules
    try:
        validate_data(df_input, is_training=False)
    except DataValidationError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))

    # Transform features
    try:
        if model_store["preprocessor"] is not None:
            X_trans, _, _, _ = prepare_data(df_input, preprocessor=model_store["preprocessor"], fit=False)
        else:
            raise RuntimeError("Preprocessor not loaded.")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Preprocessing error: {e}")

    # Predict
    try:
        model = model_store["model"]
        # Check if predict method exists on model wrapper
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_trans)
            prob = float(probs[0, 1])
            pred_class = int(prob > 0.5)
        else:
            preds = model.predict(X_trans)
            prob = float(preds[0]) if preds.ndim == 1 else float(preds[0, 1])
            pred_class = int(prob > 0.5)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Inference error: {e}")

    label = "Yes" if pred_class == 1 else "No"
    now_iso = datetime.now(timezone.utc).isoformat()

    # Log to SQLite
    log_prediction_to_db(input_dict, pred_class, prob, model_store["version"])

    return ChurnOutput(
        churn_prediction=pred_class,
        churn_label=label,
        churn_probability=round(prob, 4),
        model_version=model_store["version"],
        timestamp=now_iso
    )
