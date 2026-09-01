"""
FastAPI Serving API for ChurnOps customer churn prediction.
"""

import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

# Ensure workspace root directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import sqlite3

import joblib
import mlflow
import mlflow.pyfunc
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader

from api.schemas import BatchChurnInput, BatchChurnOutput, ChurnInput, ChurnOutput, HealthResponse
from src.config import settings
from src.data_validation import DataValidationError, validate_data
from src.preprocessing import load_preprocessor, prepare_data

# Configure structured logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configuration
MLFLOW_URI = settings.MLFLOW_TRACKING_URI
MODEL_NAME = settings.MODEL_NAME
DB_PATH = settings.PREDICTIONS_DB_PATH
API_KEY_HEADER = settings.API_KEY_HEADER
RATE_LIMIT_PER_MINUTE = settings.RATE_LIMIT_PER_MINUTE
ENABLE_CORS = settings.ENABLE_CORS
CORS_ORIGINS = settings.CORS_ORIGINS

# Global state loaded during lifespan
model_store = {"model": None, "preprocessor": None, "version": "unknown", "stage": "none", "loaded_at": None}

# Rate limiting storage (in production, use Redis)
request_counts = {}


def init_sqlite_db():
    """Initialize SQLite database for prediction logging and drift monitoring."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            request_id TEXT NOT NULL,
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
            model_version TEXT,
            processing_time_ms REAL
        )
    """)
    conn.commit()
    conn.close()
    logger.info("SQLite database initialized successfully")


def log_prediction_to_db(
    input_dict: dict,
    prediction: int,
    probability: float,
    model_version: str,
    request_id: str,
    processing_time_ms: float,
):
    """Log prediction request and model result to SQLite database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        now_iso = datetime.now(timezone.utc).isoformat()

        cursor.execute(
            """
            INSERT INTO predictions (
                request_id, timestamp, gender, SeniorCitizen, Partner, Dependents, tenure,
                PhoneService, MultipleLines, InternetService, OnlineSecurity,
                OnlineBackup, DeviceProtection, TechSupport, StreamingTV,
                StreamingMovies, Contract, PaperlessBilling, PaymentMethod,
                MonthlyCharges, TotalCharges, churn_prediction, churn_probability,
                model_version, processing_time_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                request_id,
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
                model_version,
                processing_time_ms,
            ),
        )
        conn.commit()
        conn.close()
        logger.debug(f"Prediction logged to database: {request_id}")
    except Exception as e:
        logger.warning(f"Failed to log prediction to SQLite database: {e}")


def load_model_and_preprocessor():
    """Load model from MLflow Registry ('Production' or 'Staging') or fallback local file."""
    start_time = time.time()

    # 1. Load preprocessor
    try:
        model_store["preprocessor"] = load_preprocessor(settings.PREPROCESSOR_PATH)
        logger.info("Successfully loaded preprocessor artifact.")
    except Exception as e:
        logger.warning(f"Could not load preprocessor from disk: {e}")

    # 2. Load model from MLflow Registry
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = mlflow.tracking.MlflowClient()

    for stage in ["Production", "Staging"]:
        try:
            versions = client.get_latest_versions(MODEL_NAME, stages=[stage])
            if versions:
                version = versions[0].version
                model_uri = f"models:/{MODEL_NAME}/{stage}"
                logger.info(f"Loading MLflow model '{MODEL_NAME}' stage '{stage}' version {version}...")
                model_store["model"] = mlflow.pyfunc.load_model(model_uri)
                model_store["version"] = str(version)
                model_store["stage"] = stage
                model_store["loaded_at"] = datetime.now(timezone.utc).isoformat()
                logger.info(f"Model loaded successfully in {time.time() - start_time:.2f}s")
                return
        except Exception as e:
            logger.warning(f"Could not load model from MLflow stage '{stage}': {e}")

    # Fallback to local joblib file
    fallback_path = settings.BEST_MODEL_PATH
    if os.path.exists(fallback_path):
        logger.info(f"Loading fallback model from local file {fallback_path}...")
        model_store["model"] = joblib.load(fallback_path)
        model_store["version"] = "local-1.0"
        model_store["stage"] = "LocalFallback"
        model_store["loaded_at"] = datetime.now(timezone.utc).isoformat()
    else:
        logger.error("No model found in MLflow Registry or local directory.")


def check_rate_limit(request: Request):
    """Simple in-memory rate limiting (use Redis in production)."""
    global request_counts
    client_ip = request.client.host
    now = time.time()

    # Clean old entries
    request_counts = {k: v for k, v in request_counts.items() if v > now - 60}

    if client_ip in request_counts:
        if request_counts[client_ip] >= RATE_LIMIT_PER_MINUTE:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Rate limit exceeded. Please try again later."
            )

    request_counts[client_ip] = request_counts.get(client_ip, 0) + 1


async def get_api_key(api_key: str | None = Depends(APIKeyHeader)):
    """Validate API key if authentication is enabled."""
    if os.getenv("ENABLE_AUTH", "false").lower() == "true":
        if not api_key:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="API key is required")
        # In production, validate against a secure store
        expected_key = os.getenv("API_KEY_SECRET")
        if expected_key and api_key != expected_key:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    return api_key


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    logger.info("Starting ChurnOps API...")
    init_sqlite_db()
    load_model_and_preprocessor()
    logger.info("ChurnOps API startup complete")
    yield
    # Shutdown logic
    logger.info("Shutting down ChurnOps API...")


app = FastAPI(
    title="ChurnOps Customer Churn Prediction API",
    description="Production-grade FastAPI serving endpoint for Telco Customer Churn Prediction with monitoring, logging, and model registry integration.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Add CORS middleware
if ENABLE_CORS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add request ID to all responses for tracing."""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    start_time = time.time()

    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id

    # Log request
    process_time = time.time() - start_time
    logger.info(
        f"{request.method} {request.url.path} - Status: {response.status_code} - Time: {process_time:.3f}s - RequestID: {request_id}"
    )

    return response


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint with detailed model status."""
    model_loaded = model_store["model"] is not None
    preprocessor_loaded = model_store["preprocessor"] is not None

    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_name=MODEL_NAME,
        model_stage=model_store["stage"],
        model_version=model_store["version"],
        model_loaded_at=model_store.get("loaded_at", "unknown"),
        preprocessor_loaded=preprocessor_loaded,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/metrics")
def get_metrics():
    """Prometheus-style metrics endpoint for monitoring."""
    return {
        "model_version": model_store["version"],
        "model_stage": model_store["stage"],
        "model_loaded": model_store["model"] is not None,
        "uptime_seconds": time.time()
        - (model_store.get("loaded_at", time.time()) if model_store.get("loaded_at") else time.time()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/predict", response_model=ChurnOutput)
async def predict_churn(payload: ChurnInput, request: Request):
    """Single customer churn prediction endpoint."""
    start_time = time.time()
    request_id = request.state.request_id

    # Check rate limit
    check_rate_limit(request)

    if model_store["model"] is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not currently loaded. Please check the health endpoint.",
        )

    # Convert payload to DataFrame
    input_dict = payload.model_dump()
    df_input = pd.DataFrame([input_dict])

    # Validate data rules
    try:
        validate_data(df_input, is_training=False)
    except DataValidationError as e:
        logger.warning(f"Data validation failed for request {request_id}: {e}")
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e)) from e

    # Transform features
    try:
        if model_store["preprocessor"] is not None:
            X_trans, _, _, _ = prepare_data(df_input, preprocessor=model_store["preprocessor"], fit=False)
        else:
            raise RuntimeError("Preprocessor not loaded.")
    except Exception as e:
        logger.error(f"Preprocessing error for request {request_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Preprocessing error: {str(e)}"
        ) from e

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
        logger.error(f"Inference error for request {request_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Inference error: {str(e)}"
        ) from e

    label = "Yes" if pred_class == 1 else "No"
    now_iso = datetime.now(timezone.utc).isoformat()
    processing_time_ms = (time.time() - start_time) * 1000

    # Log to SQLite
    log_prediction_to_db(input_dict, pred_class, prob, model_store["version"], request_id, processing_time_ms)

    logger.info(
        f"Prediction completed for request {request_id}: churn={label}, prob={prob:.4f}, time={processing_time_ms:.2f}ms"
    )

    return ChurnOutput(
        churn_prediction=pred_class,
        churn_label=label,
        churn_probability=round(prob, 4),
        model_version=model_store["version"],
        timestamp=now_iso,
        request_id=request_id,
        processing_time_ms=round(processing_time_ms, 2),
    )


@app.post("/predict/batch", response_model=BatchChurnOutput)
async def predict_churn_batch(payload: BatchChurnInput, request: Request):
    """Batch customer churn prediction endpoint (max 100 records)."""
    start_time = time.time()
    request_id = request.state.request_id

    if len(payload.customers) > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Batch size exceeds maximum of 100 customers"
        )

    if model_store["model"] is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model is not currently loaded.")

    results = []
    errors = []

    for idx, customer in enumerate(payload.customers):
        try:
            input_dict = customer.model_dump()
            df_input = pd.DataFrame([input_dict])

            # Validate
            validate_data(df_input, is_training=False)

            # Transform
            if model_store["preprocessor"] is not None:
                X_trans, _, _, _ = prepare_data(df_input, preprocessor=model_store["preprocessor"], fit=False)
            else:
                raise RuntimeError("Preprocessor not loaded.")

            # Predict
            model = model_store["model"]
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_trans)
                prob = float(probs[0, 1])
                pred_class = int(prob > 0.5)
            else:
                preds = model.predict(X_trans)
                prob = float(preds[0]) if preds.ndim == 1 else float(preds[0, 1])
                pred_class = int(prob > 0.5)

            label = "Yes" if pred_class == 1 else "No"
            results.append(
                {
                    "index": idx,
                    "churn_prediction": pred_class,
                    "churn_label": label,
                    "churn_probability": round(prob, 4),
                }
            )

            # Log to DB
            log_prediction_to_db(
                input_dict,
                pred_class,
                prob,
                model_store["version"],
                f"{request_id}_{idx}",
                (time.time() - start_time) * 1000,
            )

        except Exception as e:
            errors.append({"index": idx, "error": str(e)})

    processing_time_ms = (time.time() - start_time) * 1000

    return BatchChurnOutput(
        results=results,
        errors=errors,
        total_processed=len(results),
        total_errors=len(errors),
        model_version=model_store["version"],
        timestamp=datetime.now(timezone.utc).isoformat(),
        request_id=request_id,
        processing_time_ms=round(processing_time_ms, 2),
    )


@app.exception_handler(DataValidationError)
async def data_validation_exception_handler(request: Request, exc: DataValidationError):
    """Custom exception handler for data validation errors."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": str(exc), "error_type": "DataValidationError", "request_id": request.state.request_id},
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom exception handler for HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "error_type": "HTTPException", "request_id": request.state.request_id},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Custom exception handler for general exceptions."""
    logger.error(f"Unhandled exception for request {request.state.request_id}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "An internal server error occurred",
            "error_type": "InternalError",
            "request_id": request.state.request_id,
        },
    )
