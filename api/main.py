"""
FastAPI Serving API for ChurnOps customer churn prediction.
"""

import asyncio
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import httpx

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ["GIT_PYTHON_REFRESH"] = "quiet"

# Ensure workspace root directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import sqlite3
import threading

import joblib
import mlflow
import mlflow.pyfunc
import pandas as pd
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import APIKeyHeader

from api.schemas import BatchChurnInput, BatchChurnOutput, ChurnInput, ChurnOutput, HealthResponse
from src.config import settings
from src.data_validation import DataValidationError, validate_data
from src.preprocessing import find_target_col, infer_task_type, is_identifier_column, load_preprocessor, prepare_data

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
model_store: dict[str, Any] = {
    "model": None,
    "preprocessor": None,
    "version": "unknown",
    "stage": "none",
    "loaded_at": None,
}

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
    """Load model from local artifact file or MLflow Registry ('Production' or 'Staging')."""
    start_time = time.time()

    # 1. Prioritize local joblib artifact for instant loading and zero memory overhead
    fallback_path = settings.BEST_MODEL_PATH
    if os.path.exists(fallback_path):
        try:
            logger.info(f"Loading model from local file {fallback_path}...")
            model_store["model"] = joblib.load(fallback_path)
            model_store["version"] = "local-1.0"
            model_store["stage"] = "LocalPipeline"
            model_store["loaded_at"] = datetime.now(timezone.utc).isoformat()
            if os.path.exists(settings.PREPROCESSOR_PATH):
                model_store["preprocessor"] = load_preprocessor(settings.PREPROCESSOR_PATH)
            logger.info(f"Model loaded successfully from disk in {time.time() - start_time:.2f}s")
            return
        except Exception as exc:
            logger.warning(f"Could not load local joblib model: {exc}")

    # 2. Fallback to MLflow Registry
    try:
        mlflow.set_tracking_uri(MLFLOW_URI)
        client = mlflow.tracking.MlflowClient()
        import warnings

        for stage in ["Production", "Staging"]:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=FutureWarning)
                    warnings.simplefilter("ignore", category=UserWarning)
                    versions = client.get_latest_versions(MODEL_NAME, stages=[stage])  # type: ignore
                if versions:
                    version = versions[0].version
                    model_uri = f"models:/{MODEL_NAME}/{stage}"
                    logger.info(f"Loading MLflow model '{MODEL_NAME}' stage '{stage}' version {version}...")
                    model_store["model"] = mlflow.pyfunc.load_model(model_uri)
                    model_store["version"] = version
                    model_store["stage"] = stage
                    model_store["loaded_at"] = datetime.now(timezone.utc).isoformat()
                    logger.info(f"Model loaded successfully in {time.time() - start_time:.2f}s")
                    return
            except Exception as e:
                logger.warning(f"Could not load model from MLflow stage '{stage}': {e}")
    except Exception as exc:
        logger.warning(f"MLflow client unavailable: {exc}")

    logger.info("No pre-trained model found. Initializing instant baseline model...")
    create_instant_baseline_model()


def create_instant_baseline_model():
    """Initialize a fast baseline model in memory (<0.01s) for instant container startup."""
    try:
        from sklearn.linear_model import LogisticRegression

        from src.preprocessing import prepare_data

        dummy_df = pd.DataFrame({
            "tenure": [1, 12, 24, 48, 60],
            "MonthlyCharges": [20.0, 50.0, 70.0, 90.0, 100.0],
            "TotalCharges": [20.0, 600.0, 1680.0, 4320.0, 6000.0],
            "Contract": ["Month-to-month", "One year", "Two year", "Two year", "Two year"],
            "InternetService": ["DSL", "Fiber optic", "DSL", "No", "Fiber optic"],
            "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer", "Credit card", "Electronic check"],
            "Churn": ["No", "No", "No", "No", "Yes"],
        })
        X, _, preprocessor, _ = prepare_data(dummy_df.drop(columns=["Churn"]), fit=True)
        clf = LogisticRegression()
        clf.fit(X, [0, 0, 0, 0, 1])

        preprocessor.target_col_ = "Churn"  # type: ignore
        preprocessor.id_cols_ = []  # type: ignore
        preprocessor.task_type_ = "classification"  # type: ignore

        model_store["model"] = clf
        model_store["preprocessor"] = preprocessor
        model_store["version"] = "1.0-baseline"
        model_store["stage"] = "InitialBaseline"
        model_store["loaded_at"] = datetime.now(timezone.utc).isoformat()
        logger.info("Successfully initialized instant startup baseline model.")
    except Exception as exc:
        logger.error(f"Failed to create instant baseline model: {exc}")


def check_rate_limit(request: Request):
    """Simple in-memory rate limiting (use Redis in production)."""
    global request_counts
    client_ip = request.client.host if request.client else "unknown"
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


async def keep_alive_task():
    """Background task to ping health endpoint every 5 minutes, preventing Render free instance spin-down/cold-starts."""
    await asyncio.sleep(15)
    while True:
        try:
            target_url = os.getenv("RENDER_EXTERNAL_URL", f"http://127.0.0.1:{settings.API_PORT}")
            if not target_url.endswith("/health"):
                target_url = f"{target_url.rstrip('/')}/health"
            async with httpx.AsyncClient(timeout=10.0) as client:
                await client.get(target_url)
                logger.debug("Keep-alive ping sent successfully.")
        except Exception as exc:
            logger.debug(f"Keep-alive ping note: {exc}")
        await asyncio.sleep(300)  # Ping every 5 minutes (300 seconds)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    logger.info("Starting ChurnOps API...")
    init_sqlite_db()
    load_model_and_preprocessor()
    logger.info("ChurnOps API startup complete")
    ping_task = asyncio.create_task(keep_alive_task())
    yield
    # Shutdown logic
    logger.info("Shutting down ChurnOps API...")
    ping_task.cancel()


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


@app.post("/dataset/upload")
async def upload_dataset_api(file: UploadFile = File(...)):  # noqa: B008
    """Upload any CSV dataset, validate structure, and return column options and preview."""
    try:
        contents = await file.read()
        import io

        df = pd.read_csv(io.BytesIO(contents))

        os.makedirs("data/raw", exist_ok=True)
        upload_path = "data/raw/user_upload.csv"
        df.to_csv(upload_path, index=False)

        detected_target = find_target_col(df, allow_fallback=True)
        task_type = infer_task_type(df[detected_target]) if (detected_target and detected_target in df.columns) else "classification"
        target_options = [c for c in df.columns if not is_identifier_column(df, c)]
        if not target_options:
            target_options = list(df.columns)

        preview_records = df.head(10).to_dict(orient="records")

        return {
            "status": "success",
            "filename": file.filename or "uploaded.csv",
            "rows": len(df),
            "columns": list(df.columns),
            "num_columns": len(df.columns),
            "detected_target": detected_target,
            "task_type": task_type,
            "target_options": target_options,
            "preview": preview_records,
        }
    except Exception as exc:
        logger.error(f"Error processing uploaded dataset: {exc}")
        raise HTTPException(status_code=400, detail=f"Error reading uploaded CSV: {str(exc)}") from exc


# Background training state
training_job_status: dict[str, Any] = {
    "status": "idle",
    "progress": 0,
    "message": "Ready to train",
    "result": None,
    "error": None,
    "updated_at": None,
}


def update_training_progress(percent: int, message: str) -> None:
    training_job_status["progress"] = percent
    training_job_status["message"] = message
    training_job_status["updated_at"] = time.time()


def run_background_training(data_path: str, target_col: str, fast_mode: bool) -> None:
    global training_job_status
    try:
        training_job_status["status"] = "running"
        training_job_status["progress"] = 10
        training_job_status["message"] = "Initializing leak-free preprocessing & data split..."
        training_job_status["error"] = None
        training_job_status["result"] = None

        from src.train import train_and_evaluate

        result = train_and_evaluate(
            data_path=data_path,
            target_col=target_col,
            fast_mode=fast_mode,
            progress_callback=update_training_progress,
        )

        load_model_and_preprocessor()

        training_job_status["status"] = "completed"
        training_job_status["progress"] = 100
        training_job_status["message"] = "Model suite training completed successfully!"
        training_job_status["result"] = result
    except Exception as exc:
        logger.error(f"Background training failed: {exc}")
        training_job_status["status"] = "failed"
        training_job_status["progress"] = 0
        training_job_status["message"] = f"Training failed: {str(exc)}"
        training_job_status["error"] = str(exc)



@app.post("/dataset/train")
async def train_dataset_api(payload: dict):
    """Start model suite training asynchronously in the background."""
    if training_job_status["status"] == "running":
        return {
            "status": "processing",
            "message": "Training is already in progress.",
            "progress": training_job_status["progress"],
        }

    target_col = payload.get("target_col")
    fast_mode = payload.get("fast_mode", True)
    data_path = "data/raw/user_upload.csv" if os.path.exists("data/raw/user_upload.csv") else "data/raw/telco_churn.csv"

    if not os.path.exists(data_path):
        raise HTTPException(status_code=400, detail="No dataset found. Please upload a CSV first.")

    df = pd.read_csv(data_path)
    if target_col and target_col not in df.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{target_col}' not found in dataset.")

    thread = threading.Thread(
        target=run_background_training,
        args=(data_path, target_col, fast_mode),
        daemon=True,
    )
    thread.start()

    return {
        "status": "started",
        "message": "Training started in background.",
        "progress": 5,
    }


@app.get("/dataset/train/status")
def get_training_status():
    """Get real-time background training progress and metrics."""
    return training_job_status


@app.get("/dataset/preview")
def get_dataset_preview():
    """Get active uploaded dataset row preview and details."""
    data_path = "data/raw/user_upload.csv" if os.path.exists("data/raw/user_upload.csv") else "data/raw/telco_churn.csv"
    if not os.path.exists(data_path):
        return {"has_dataset": False}

    df = pd.read_csv(data_path)
    target_col = getattr(model_store.get("preprocessor"), "target_col_", None) or find_target_col(df)
    target_options = [c for c in df.columns if not is_identifier_column(df, c)]

    return {
        "has_dataset": True,
        "path": data_path,
        "rows": len(df),
        "num_columns": len(df.columns),
        "target_col": target_col,
        "columns": list(df.columns),
        "target_options": target_options,
        "preview": df.head(10).to_dict(orient="records"),
    }




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


@app.get("/", response_class=HTMLResponse)
@app.head("/")
def root():
    """Root endpoint delivering an interactive Web Application UI for ChurnOps."""
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ChurnOps - Machine Learning Studio & Monitoring</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-page: #f8fafc;
            --bg-card: #ffffff;
            --border-color: #e2e8f0;
            --border-hover: #cbd5e1;
            --text-primary: #0f172a;
            --text-secondary: #475569;
            --text-muted: #64748b;
            --primary: #4f46e5;
            --primary-hover: #4338ca;
            --primary-light: #eef2ff;
            --success-bg: #f0fdf4;
            --success-text: #166534;
            --success-border: #bbf7d0;
            --warning-bg: #fffbeb;
            --warning-text: #92400e;
            --warning-border: #fde68a;
            --danger-bg: #fef2f2;
            --danger-text: #991b1b;
            --danger-border: #fecaca;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }

        body {
            background-color: var(--bg-page);
            color: var(--text-primary);
            min-height: 100vh;
            line-height: 1.5;
        }

        .header {
            background: #ffffff;
            border-bottom: 1px solid var(--border-color);
            padding: 1rem 2rem;
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .header-content {
            max-width: 1280px;
            margin: 0 auto;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }

        .brand-icon {
            background: var(--primary);
            color: white;
            width: 36px;
            height: 36px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 800;
            font-size: 1.1rem;
        }

        .brand-text h1 {
            font-size: 1.25rem;
            font-weight: 700;
            color: var(--text-primary);
        }

        .brand-text p {
            font-size: 0.8rem;
            color: var(--text-muted);
        }

        .nav-actions {
            display: flex;
            gap: 0.75rem;
        }

        .btn-link {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.5rem 0.85rem;
            background: #ffffff;
            border: 1px solid var(--border-color);
            border-radius: 6px;
            color: var(--text-secondary);
            font-size: 0.85rem;
            font-weight: 500;
            text-decoration: none;
            transition: all 0.15s ease;
        }

        .btn-link:hover {
            border-color: var(--primary);
            color: var(--primary);
            background: var(--primary-light);
        }

        .main-container {
            max-width: 1280px;
            margin: 2rem auto;
            padding: 0 1.5rem;
        }

        .tabs-nav {
            display: flex;
            gap: 0.5rem;
            border-bottom: 1px solid var(--border-color);
            margin-bottom: 1.5rem;
            overflow-x: auto;
        }

        .tab-btn {
            padding: 0.75rem 1.25rem;
            background: transparent;
            border: none;
            border-bottom: 2px solid transparent;
            color: var(--text-muted);
            font-weight: 500;
            font-size: 0.95rem;
            cursor: pointer;
            transition: all 0.15s ease;
            white-space: nowrap;
        }

        .tab-btn:hover {
            color: var(--text-primary);
        }

        .tab-btn.active {
            color: var(--primary);
            border-bottom-color: var(--primary);
            font-weight: 600;
        }

        .tab-content {
            display: none;
        }

        .tab-content.active {
            display: block;
        }

        .card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 1.5rem;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
            margin-bottom: 1.5rem;
        }

        .card-header {
            margin-bottom: 1.25rem;
        }

        .card-title {
            font-size: 1.1rem;
            font-weight: 700;
            color: var(--text-primary);
        }

        .card-desc {
            font-size: 0.875rem;
            color: var(--text-muted);
            margin-top: 0.25rem;
        }

        .dropzone {
            border: 2px dashed #cbd5e1;
            border-radius: 8px;
            padding: 2.5rem 1.5rem;
            text-align: center;
            background: #f8fafc;
            cursor: pointer;
            transition: all 0.2s ease;
            margin-bottom: 1.25rem;
        }

        .dropzone:hover, .dropzone.dragover {
            border-color: var(--primary);
            background: var(--primary-light);
            box-shadow: 0 0 0 4px rgba(79, 70, 229, 0.12);
        }

        .dropzone input {
            display: none;
        }

        .dropzone-icon {
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }

        .dropzone-text {
            font-weight: 600;
            color: var(--text-primary);
        }

        .dropzone-sub {
            font-size: 0.85rem;
            color: var(--text-muted);
            margin-top: 0.25rem;
        }

        .btn-primary {
            background: var(--primary);
            color: white;
            border: none;
            padding: 0.65rem 1.25rem;
            border-radius: 6px;
            font-weight: 600;
            font-size: 0.9rem;
            cursor: pointer;
            transition: background 0.15s ease;
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
        }

        .btn-primary:hover {
            background: var(--primary-hover);
        }

        .btn-block {
            width: 100%;
            justify-content: center;
        }

        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1rem;
            margin-bottom: 1.25rem;
        }

        .form-group {
            display: flex;
            flex-direction: column;
            gap: 0.35rem;
        }

        .form-label {
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text-secondary);
        }

        .form-control {
            padding: 0.55rem 0.75rem;
            border: 1px solid var(--border-color);
            border-radius: 6px;
            font-size: 0.9rem;
            color: var(--text-primary);
            background: #ffffff;
            outline: none;
        }

        .form-control:focus {
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.1);
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 1rem;
            margin-top: 1rem;
        }

        .metric-tile {
            background: #f8fafc;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 1rem;
        }

        .metric-name {
            font-size: 0.75rem;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
        }

        .metric-val {
            font-size: 1.5rem;
            font-weight: 800;
            color: var(--text-primary);
            margin-top: 0.25rem;
        }

        .data-table-container {
            overflow-x: auto;
            max-height: 350px;
            border: 1px solid var(--border-color);
            border-radius: 6px;
            margin-top: 1rem;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.85rem;
            text-align: left;
        }

        th {
            background: #f1f5f9;
            color: var(--text-secondary);
            font-weight: 600;
            padding: 0.6rem 0.85rem;
            position: sticky;
            top: 0;
            border-bottom: 1px solid var(--border-color);
        }

        td {
            padding: 0.55rem 0.85rem;
            border-bottom: 1px solid #f1f5f9;
            color: var(--text-primary);
            white-space: nowrap;
        }

        tr:hover {
            background: #f8fafc;
        }

        .badge {
            display: inline-block;
            padding: 0.25rem 0.6rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
        }

        .badge-success { background: var(--success-bg); color: var(--success-text); border: 1px solid var(--success-border); }
        .badge-warning { background: var(--warning-bg); color: var(--warning-text); border: 1px solid var(--warning-border); }
        .badge-danger { background: var(--danger-bg); color: var(--danger-text); border: 1px solid var(--danger-border); }

        .json-preview {
            background: #0f172a;
            color: #38bdf8;
            padding: 1rem;
            border-radius: 6px;
            font-family: monospace;
            font-size: 0.85rem;
            max-height: 250px;
            overflow-y: auto;
        }

        .alert {
            padding: 0.85rem 1rem;
            border-radius: 6px;
            font-size: 0.9rem;
            margin-bottom: 1rem;
        }

        .alert-info { background: #eff6ff; color: #1e40af; border: 1px solid #bfdbfe; }
        .alert-success { background: var(--success-bg); color: var(--success-text); border: 1px solid var(--success-border); }
        .alert-warning { background: var(--warning-bg); color: var(--warning-text); border: 1px solid var(--warning-border); }
        .alert-danger { background: var(--danger-bg); color: var(--danger-text); border: 1px solid var(--danger-border); }
    </style>
</head>
<body>
    <header class="header">
        <div class="header-content">
            <div class="brand">
                <div class="brand-icon">&#9889;</div>
                <div class="brand-text">
                    <h1>ChurnOps</h1>
                    <p>Machine Learning Studio & Monitoring System</p>
                </div>
            </div>
            <div class="nav-actions">
                <a href="/docs" target="_blank" class="btn-link">&#128216; API Docs</a>
                <a href="/health" target="_blank" class="btn-link">&#128147; Health Probe</a>
                <a href="https://github.com/bishtprateek270-hue/churnops" target="_blank" class="btn-link">&#128191; GitHub</a>
            </div>
        </div>
    </header>

    <main class="main-container">
        <nav class="tabs-nav">
            <button class="tab-btn active" onclick="switchTab('tab-upload', event)">&#128194; 1. Dataset Upload & Train</button>
            <button class="tab-btn" onclick="switchTab('tab-row', event)">&#127919; 2. Predict Dataset Row</button>
            <button class="tab-btn" onclick="switchTab('tab-batch', event)">&#9889; 3. Batch CSV Inference</button>
            <button class="tab-btn" onclick="switchTab('tab-custom', event)">&#128221; 4. Custom Single Form</button>
            <button class="tab-btn" onclick="switchTab('tab-health', event)">&#128202; 5. Service Telemetry</button>
        </nav>

        <!-- TAB 1: UPLOAD & TRAIN -->
        <div id="tab-upload" class="tab-content active">
            <div class="card">
                <div class="card-header">
                    <h2 class="card-title">Upload Any Tabular CSV Dataset</h2>
                    <p class="card-desc">Upload classification or regression datasets (e.g. Telco Churn, Kaggle House Prices, Credit Churn). Automatic feature validation, ID filtering, and model suite evaluation.</p>
                </div>

                <div class="dropzone" id="dropzoneBox" onclick="document.getElementById('csvInput').click()">
                    <input type="file" id="csvInput" accept=".csv" onchange="handleFileInputChange(event)">
                    <div class="dropzone-icon">&#128196;</div>
                    <div class="dropzone-text" id="dropText">Click to select or drag and drop CSV file</div>
                    <div class="dropzone-sub">Supports tabular classification or regression datasets</div>
                </div>

                <div id="uploadStatus"></div>

                <div id="datasetDetails" style="display: none;">
                    <div class="form-grid">
                        <div class="form-group">
                            <label class="form-label">Total Rows</label>
                            <input type="text" id="dsRows" class="form-control" readonly>
                        </div>
                        <div class="form-group">
                            <label class="form-label">Total Columns</label>
                            <input type="text" id="dsCols" class="form-control" readonly>
                        </div>
                        <div class="form-group">
                            <label class="form-label" for="targetSelect">Select Target Column (Excluded IDs)</label>
                            <select id="targetSelect" class="form-control"></select>
                        </div>
                    </div>

                    <div style="margin-bottom: 1.25rem;">
                        <button class="btn-primary" id="trainBtn" onclick="trainModel()"><span id="trainSpinner">&#9889;</span> Train Leak-Free Model Suite</button>
                    </div>

                    <h3 style="font-size: 0.95rem; font-weight: 700; margin-bottom: 0.5rem;">Dataset Preview (First 10 Rows)</h3>
                    <div class="data-table-container">
                        <table id="previewTable"></table>
                    </div>
                </div>
            </div>

            <div id="trainingResultsCard" class="card" style="display: none;">
                <div class="card-header">
                    <h2 class="card-title" id="bestModelTitle">Training Results</h2>
                    <p class="card-desc" id="trainMetaSub">Model evaluation metrics on untouched holdout test set.</p>
                </div>
                <div class="metrics-grid" id="metricsGrid"></div>
            </div>
        </div>

        <!-- TAB 2: ROW PREDICTOR -->
        <div id="tab-row" class="tab-content">
            <div class="card">
                <div class="card-header">
                    <h2 class="card-title">Select Dataset Row for Prediction</h2>
                    <p class="card-desc">Select any row index from the active dataset. Identifier and target columns are automatically removed before running model prediction.</p>
                </div>

                <div class="form-grid" style="max-width: 400px;">
                    <div class="form-group">
                        <label class="form-label" for="rowIndexInput">Dataset Row Index</label>
                        <input type="number" id="rowIndexInput" class="form-control" value="0" min="0" onchange="loadRowPreview()">
                    </div>
                </div>

                <div style="margin-bottom: 1rem;">
                    <button class="btn-primary" id="predictRowBtn" onclick="predictRow()"><span id="predictRowSpinner">&#9889;</span> Predict Selected Row</button>
                </div>

                <div id="rowOutputContainer" style="display: none;">
                    <div class="metrics-grid" id="rowResultMetrics" style="margin-bottom: 1.25rem;"></div>

                    <h3 style="font-size: 0.95rem; font-weight: 700; margin-bottom: 0.5rem;">Clean Row Features JSON (ID & Target Stripped)</h3>
                    <pre class="json-preview" id="rowJsonPreview"></pre>
                </div>
            </div>
        </div>

        <!-- TAB 3: BATCH CSV -->
        <div id="tab-batch" class="tab-content">
            <div class="card">
                <div class="card-header">
                    <h2 class="card-title">Batch CSV Inference</h2>
                    <p class="card-desc">Generate predictions for multiple records at once and export results as CSV.</p>
                </div>

                <div class="form-grid" style="max-width: 400px;">
                    <div class="form-group">
                        <label class="form-label" for="batchLimit">Rows Count Limit</label>
                        <input type="number" id="batchLimit" class="form-control" value="50" min="5" max="500">
                    </div>
                </div>

                <button class="btn-primary" onclick="runBatchPrediction()">&#9889; Run Batch Prediction</button>

                <div id="batchResultsContainer" style="display: none; margin-top: 1.25rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                        <h3 style="font-size: 0.95rem; font-weight: 700;">Batch Predictions Output</h3>
                        <button class="btn-link" onclick="downloadBatchCSV()">&#128229; Download CSV</button>
                    </div>
                    <div class="data-table-container">
                        <table id="batchTable"></table>
                    </div>
                </div>
            </div>
        </div>

        <!-- TAB 4: CUSTOM SINGLE INPUT -->
        <div id="tab-custom" class="tab-content">
            <div class="card">
                <div class="card-header">
                    <h2 class="card-title">Custom Single Feature Input</h2>
                    <p class="card-desc">Fill quick demo profiles or manually adjust features for instant inference.</p>
                </div>

                <div style="display: flex; gap: 0.75rem; margin-bottom: 1.5rem;">
                    <button class="btn-link" onclick="fillPreset('high')">&#9888; High Risk Customer Preset</button>
                    <button class="btn-link" onclick="fillPreset('low')">&#9989; Low Risk Customer Preset</button>
                </div>

                <form id="customForm" onsubmit="submitCustomForm(event)">
                    <div class="form-grid">
                        <div class="form-group">
                            <label class="form-label">Tenure (Months)</label>
                            <input type="number" id="tenure" class="form-control" value="12">
                        </div>
                        <div class="form-group">
                            <label class="form-label">Internet Service</label>
                            <select id="InternetService" class="form-control">
                                <option value="Fiber optic">Fiber optic</option>
                                <option value="DSL">DSL</option>
                                <option value="No">No</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label class="form-label">Contract Type</label>
                            <select id="Contract" class="form-control">
                                <option value="Month-to-month">Month-to-month</option>
                                <option value="One year">One year</option>
                                <option value="Two year">Two year</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label class="form-label">Monthly Charges ($)</label>
                            <input type="number" step="0.01" id="MonthlyCharges" class="form-control" value="70.35">
                        </div>
                        <div class="form-group">
                            <label class="form-label">Total Charges ($)</label>
                            <input type="number" step="0.01" id="TotalCharges" class="form-control" value="844.20">
                        </div>
                        <div class="form-group">
                            <label class="form-label">Payment Method</label>
                            <select id="PaymentMethod" class="form-control">
                                <option value="Electronic check">Electronic check</option>
                                <option value="Bank transfer (automatic)">Bank transfer (automatic)</option>
                                <option value="Credit card (automatic)">Credit card (automatic)</option>
                                <option value="Mailed check">Mailed check</option>
                            </select>
                        </div>
                    </div>

                    <button type="submit" class="btn-primary">&#9889; Run Live Prediction</button>
                </form>

                <div id="customResult" style="display: none; margin-top: 1.25rem;">
                    <h3 style="font-size: 0.95rem; font-weight: 700; margin-bottom: 0.5rem;">Prediction Output</h3>
                    <pre class="json-preview" id="customJsonOutput"></pre>
                </div>
            </div>
        </div>

        <!-- TAB 5: TELEMETRY & HEALTH -->
        <div id="tab-health" class="tab-content">
            <div class="card">
                <div class="card-header">
                    <h2 class="card-title">Service Telemetry & Environment Health</h2>
                    <p class="card-desc">Active model store state, database logging health, and endpoint URLs.</p>
                </div>

                <div class="metrics-grid" id="healthMetrics"></div>

                <h3 style="font-size: 0.95rem; font-weight: 700; margin: 1.25rem 0 0.5rem 0;">Health Response Payload</h3>
                <pre class="json-preview" id="healthJson"></pre>
            </div>
        </div>
    </main>

    <script>
        let activeDataset = null;
        let batchResultsData = null;

        function switchTab(tabId, evt) {
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

            if (evt && evt.currentTarget) {
                evt.currentTarget.classList.add('active');
            }
            const targetEl = document.getElementById(tabId);
            if (targetEl) {
                targetEl.classList.add('active');
            }

            if (tabId === 'tab-row' && activeDataset) {
                loadRowPreview();
            }
            if (tabId === 'tab-health') {
                loadHealthTelemetry();
            }
        }

        async function initPage() {
            setupDragAndDrop();
            try {
                const res = await fetch('/dataset/preview');
                const data = await res.json();
                if (data.has_dataset) {
                    activeDataset = data;
                    renderDatasetInfo(data);
                }
            } catch(e) {}
        }

        function setupDragAndDrop() {
            const dropzone = document.getElementById('dropzoneBox');
            if (!dropzone) return;

            ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
                dropzone.addEventListener(eventName, function(e) {
                    e.preventDefault();
                    e.stopPropagation();
                }, false);
                document.body.addEventListener(eventName, function(e) {
                    e.preventDefault();
                    e.stopPropagation();
                }, false);
            });

            ['dragenter', 'dragover'].forEach(eventName => {
                dropzone.addEventListener(eventName, function() { dropzone.classList.add('dragover'); }, false);
            });

            ['dragleave', 'drop'].forEach(eventName => {
                dropzone.addEventListener(eventName, function() { dropzone.classList.remove('dragover'); }, false);
            });

            dropzone.addEventListener('drop', function(e) {
                const dt = e.dataTransfer;
                const files = dt.files;
                if (files && files.length > 0) {
                    uploadFile(files[0]);
                }
            }, false);
        }

        function handleFileInputChange(e) {
            if (e.target.files && e.target.files.length > 0) {
                uploadFile(e.target.files[0]);
            }
        }

        async function uploadFile(file) {
            if (!file) return;
            if (!file.name.toLowerCase().endsWith('.csv')) {
                document.getElementById('uploadStatus').innerHTML = '<div class="alert alert-warning">&#9888; Please select or drop a valid .csv file.</div>';
                return;
            }

            document.getElementById('dropText').innerText = 'Selected: ' + file.name;
            const formData = new FormData();
            formData.append('file', file);

            document.getElementById('uploadStatus').innerHTML = '<div class="alert alert-info">&#8987; Uploading and validating CSV...</div>';

            try {
                const res = await fetch('/dataset/upload', { method: 'POST', body: formData });
                const data = await res.json();

                if (res.ok) {
                    activeDataset = data;
                    document.getElementById('uploadStatus').innerHTML = '<div class="alert alert-success">&#9989; Loaded ' + data.rows + ' rows, ' + data.num_columns + ' columns. Detected target: <strong>' + data.detected_target + '</strong></div>';
                    renderDatasetInfo(data);
                } else {
                    document.getElementById('uploadStatus').innerHTML = '<div class="alert alert-danger">&#10060; ' + (data.detail || 'Upload failed') + '</div>';
                }
            } catch (err) {
                document.getElementById('uploadStatus').innerHTML = '<div class="alert alert-danger">&#10060; Error uploading file: ' + err.message + '</div>';
            }
        }

        function renderDatasetInfo(data) {
            document.getElementById('dsRows').value = data.rows;
            document.getElementById('dsCols').value = data.num_columns;

            const select = document.getElementById('targetSelect');
            select.innerHTML = '';
            data.target_options.forEach(opt => {
                const el = document.createElement('option');
                el.value = opt;
                el.innerText = opt;
                if (opt === data.detected_target) el.selected = true;
                select.appendChild(el);
            });

            document.getElementById('datasetDetails').style.display = 'block';
            renderTable('previewTable', data.preview);
        }

        function renderTable(tableId, records) {
            if (!records || !records.length) return;
            const table = document.getElementById(tableId);
            table.innerHTML = '';

            const cols = Object.keys(records[0]);
            let thead = '<thead><tr>' + cols.map(c => '<th>' + c + '</th>').join('') + '</tr></thead>';
            let tbody = '<tbody>' + records.map(r => '<tr>' + cols.map(c => '<td>' + (r[c] !== null ? r[c] : '') + '</td>').join('') + '</tr>').join('') + '</tbody>';

            table.innerHTML = thead + tbody;
        }

        async function pollTrainingStatus() {
            const btn = document.getElementById('trainBtn');
            const pollInterval = setInterval(async () => {
                try {
                    const statusRes = await fetch('/dataset/train/status');
                    if (!statusRes.ok) return;
                    const statusData = await statusRes.json();

                    const pct = statusData.progress || 0;
                    const msg = statusData.message || 'Training in progress...';
                    document.getElementById('trainSpinner').innerText = pct + '%';
                    document.getElementById('uploadStatus').innerHTML = '<div class="alert alert-info">&#9881;&#65039; ' + msg + ' (' + pct + '%)</div>';

                    if (statusData.status === 'completed') {
                        clearInterval(pollInterval);
                        if (btn) btn.disabled = false;
                        document.getElementById('trainSpinner').innerText = 'Train';
                        const bestName = (statusData.result && statusData.result.best_model_name) ? statusData.result.best_model_name : 'Best Model';
                        const totalTime = (statusData.result && statusData.result.total_time_seconds) ? statusData.result.total_time_seconds : '0';
                        document.getElementById('uploadStatus').innerHTML = '<div class="alert alert-success">&#127881; Training complete! Best Model: <strong>' + bestName + '</strong> in ' + totalTime + 's</div>';
                        if (statusData.result) {
                            renderResults(statusData.result);
                        }
                    } else if (statusData.status === 'failed') {
                        clearInterval(pollInterval);
                        if (btn) btn.disabled = false;
                        document.getElementById('trainSpinner').innerText = 'Train';
                        document.getElementById('uploadStatus').innerHTML = '<div class="alert alert-danger">&#10060; Training failed: ' + (statusData.error || statusData.message || 'Unknown error') + '</div>';
                    }
                } catch (e) {
                    console.error('Status poll error:', e);
                }
            }, 1200);
        }

        async function trainModel() {
            const btn = document.getElementById('trainBtn');
            const target_col = document.getElementById('targetSelect').value;
            if (btn) btn.disabled = true;
            document.getElementById('trainSpinner').innerText = '0%';
            document.getElementById('uploadStatus').innerHTML = '<div class="alert alert-info">&#8987; Starting model suite training in background...</div>';
            document.getElementById('uploadStatus').scrollIntoView({ behavior: 'smooth', block: 'center' });

            try {
                const res = await fetch('/dataset/train', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ target_col: target_col, fast_mode: true })
                });

                let data;
                try {
                    data = await res.json();
                } catch (e) {
                    const rawText = await res.text().catch(() => '');
                    data = { detail: 'Server response parsing failed (HTTP ' + res.status + ' ' + (res.statusText || 'Error') + '). ' + (rawText ? rawText.substring(0, 100) : '') };
                }

                if (res.ok && (data.status === 'started' || data.status === 'processing' || data.status === 'success')) {
                    if (data.status === 'success' && data.result) {
                        if (btn) btn.disabled = false;
                        document.getElementById('trainSpinner').innerText = 'Train';
                        renderResults(data.result);
                    } else {
                        pollTrainingStatus();
                    }
                } else {
                    if (btn) btn.disabled = false;
                    document.getElementById('trainSpinner').innerText = 'Train';
                    document.getElementById('uploadStatus').innerHTML = '<div class="alert alert-danger">&#10060; Training failed: ' + (data.detail || data.message || 'Server Error') + '</div>';
                }
            } catch (err) {
                if (btn) btn.disabled = false;
                document.getElementById('trainSpinner').innerText = 'Train';
                document.getElementById('uploadStatus').innerHTML = '<div class="alert alert-danger">&#10060; Error starting training: ' + err.message + '</div>';
            }
        }

        function renderResults(result) {
            const card = document.getElementById('trainingResultsCard');
            card.style.display = 'block';
            document.getElementById('bestModelTitle').innerText = 'Holdout Test Metrics — Best Model: ' + result.best_model_name.replace(/_/g, ' ');

            const metrics = result.best_test_metrics || {};
            const grid = document.getElementById('metricsGrid');
            grid.innerHTML = '';

            for (const [k, v] of Object.entries(metrics)) {
                let displayVal = v;
                if (typeof v === 'number') {
                    displayVal = (k.includes('accuracy') || k.includes('precision') || k.includes('recall') || k.includes('f1') || k.includes('roc_auc')) ? (v * 100).toFixed(1) + '%' : v.toFixed(4);
                }
                grid.innerHTML += '<div class="metric-tile"><div class="metric-name">' + k.replace(/_/g, ' ') + '</div><div class="metric-val">' + displayVal + '</div></div>';
            }

            card.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }

        function loadRowPreview() {
            if (!activeDataset || !activeDataset.preview) return;
            const idx = parseInt(document.getElementById('rowIndexInput').value) || 0;
            const rowData = activeDataset.preview[idx % activeDataset.preview.length] || {};

            const cleanRow = { ...rowData };
            delete cleanRow[activeDataset.target_col];
            delete cleanRow['Id'];
            delete cleanRow['customerID'];

            document.getElementById('rowJsonPreview').innerText = JSON.stringify(cleanRow, null, 2);
            document.getElementById('rowOutputContainer').style.display = 'block';
        }

        async function predictRow() {
            if (!activeDataset || !activeDataset.preview) return;
            const btn = document.getElementById('predictRowBtn');
            if (btn) btn.disabled = true;
            const idx = parseInt(document.getElementById('rowIndexInput').value) || 0;
            const rowData = activeDataset.preview[idx % activeDataset.preview.length] || {};

            document.getElementById('predictRowSpinner').innerText = '...';

            try {
                const res = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(rowData)
                });

                const data = await res.json();
                if (btn) btn.disabled = false;
                document.getElementById('predictRowSpinner').innerText = 'Predict';

                if (res.ok) {
                    const grid = document.getElementById('rowResultMetrics');
                    const actualVal = (rowData && rowData[activeDataset.target_col] !== undefined) ? rowData[activeDataset.target_col] : 'Unknown';
                    const isRegression = (activeDataset && activeDataset.task_type === 'regression');

                    if (isRegression) {
                        grid.innerHTML = `
                            <div class="metric-tile"><div class="metric-name">Predicted Value</div><div class="metric-val" style="color: #4f46e5;">${data.churn_label}</div></div>
                            <div class="metric-tile"><div class="metric-name">Actual Target Value</div><div class="metric-val">${actualVal}</div></div>
                            <div class="metric-tile"><div class="metric-name">Model Version</div><div class="metric-val">${data.model_version}</div></div>
                            <div class="metric-tile"><div class="metric-name">Inference Latency</div><div class="metric-val">${data.processing_time_ms} ms</div></div>
                        `;
                    } else {
                        grid.innerHTML = `
                            <div class="metric-tile"><div class="metric-name">Predicted Label</div><div class="metric-val" style="color: #4f46e5;">${data.churn_label} (${data.churn_prediction})</div></div>
                            <div class="metric-tile"><div class="metric-name">Confidence Probability</div><div class="metric-val">${(data.churn_probability * 100).toFixed(1)}%</div></div>
                            <div class="metric-tile"><div class="metric-name">Actual Label</div><div class="metric-val">${actualVal}</div></div>
                            <div class="metric-tile"><div class="metric-name">Inference Latency</div><div class="metric-val">${data.processing_time_ms} ms</div></div>
                        `;
                    }
                    document.getElementById('rowOutputContainer').style.display = 'block';
                    document.getElementById('rowOutputContainer').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                }
            } catch(e) {
                if (btn) btn.disabled = false;
                document.getElementById('predictRowSpinner').innerText = 'Predict';
            }
        }

        async function runBatchPrediction() {
            const limit = parseInt(document.getElementById('batchLimit').value) || 50;
            if (!activeDataset || !activeDataset.preview) return;

            const samples = activeDataset.preview.slice(0, limit);
            const res = await fetch('/predict/batch', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ customers: samples })
            });

            const data = await res.json();
            if (res.ok) {
                batchResultsData = data.results;
                renderTable('batchTable', data.results);
                document.getElementById('batchResultsContainer').style.display = 'block';
            }
        }

        function downloadBatchCSV() {
            if (!batchResultsData) return;
            const keys = Object.keys(batchResultsData[0]);
            let csv = keys.join(',') + '\\n';
            batchResultsData.forEach(r => {
                csv += keys.map(k => r[k]).join(',') + '\\n';
            });
            const blob = new Blob([csv], { type: 'text/csv' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'batch_predictions.csv';
            a.click();
        }

        function fillPreset(type) {
            if (type === 'high') {
                document.getElementById('tenure').value = 2;
                document.getElementById('InternetService').value = 'Fiber optic';
                document.getElementById('Contract').value = 'Month-to-month';
                document.getElementById('MonthlyCharges').value = 95.70;
                document.getElementById('TotalCharges').value = 191.40;
            } else {
                document.getElementById('tenure').value = 60;
                document.getElementById('InternetService').value = 'DSL';
                document.getElementById('Contract').value = 'Two year';
                document.getElementById('MonthlyCharges').value = 85.10;
                document.getElementById('TotalCharges').value = 5106.00;
            }
        }

        async function submitCustomForm(e) {
            e.preventDefault();
            const payload = {
                tenure: parseInt(document.getElementById('tenure').value),
                InternetService: document.getElementById('InternetService').value,
                Contract: document.getElementById('Contract').value,
                MonthlyCharges: parseFloat(document.getElementById('MonthlyCharges').value),
                TotalCharges: parseFloat(document.getElementById('TotalCharges').value),
                PaymentMethod: document.getElementById('PaymentMethod').value,
            };

            const res = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const data = await res.json();
            document.getElementById('customJsonOutput').innerText = JSON.stringify(data, null, 2);
            document.getElementById('customResult').style.display = 'block';
        }

        async function loadHealthTelemetry() {
            const res = await fetch('/health');
            const data = await res.json();
            document.getElementById('healthJson').innerText = JSON.stringify(data, null, 2);

            const grid = document.getElementById('healthMetrics');
            grid.innerHTML = `
                <div class="metric-tile"><div class="metric-name">Service Status</div><div class="metric-val" style="color: #166534;">${data.status.toUpperCase()}</div></div>
                <div class="metric-tile"><div class="metric-name">Active Model</div><div class="metric-val">${data.model_name}</div></div>
                <div class="metric-tile"><div class="metric-name">Stage / Version</div><div class="metric-val">${data.model_stage} (v${data.model_version})</div></div>
                <div class="metric-tile"><div class="metric-name">Preprocessor</div><div class="metric-val">${data.preprocessor_loaded ? 'Loaded' : 'None'}</div></div>
            `;
        }

        window.addEventListener('load', initPage);
    </script>
</body>
</html>"""
    return HTMLResponse(content=html_content, media_type="text/html; charset=utf-8")


@app.get("/health", response_model=HealthResponse)
@app.head("/health")
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
@app.head("/metrics")
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
        preprocessor = model_store.get("preprocessor")
        task_type = getattr(preprocessor, "task_type_", "classification")

        if task_type == "classification":
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_trans)
                prob = float(probs[0, 1]) if probs.ndim > 1 and probs.shape[1] > 1 else float(probs[0])
                pred_class = int(prob > 0.5)
            else:
                preds = model.predict(X_trans)
                pred_class = int(preds[0])
                prob = float(pred_class)
            label = "Yes" if pred_class == 1 else "No"
        else:
            preds = model.predict(X_trans)
            pred_val = float(preds[0]) if hasattr(preds, "__len__") else float(preds)
            prob = 1.0
            pred_class = round(pred_val)
            label = f"{pred_val:.2f}"
    except Exception as e:
        logger.error(f"Inference error for request {request_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Inference error: {str(e)}"
        ) from e
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
            preprocessor = model_store.get("preprocessor")
            task_type = getattr(preprocessor, "task_type_", "classification")

            if task_type == "classification":
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X_trans)
                    prob = float(probs[0, 1]) if probs.ndim > 1 and probs.shape[1] > 1 else float(probs[0])
                    pred_class = int(prob > 0.5)
                else:
                    preds = model.predict(X_trans)
                    pred_class = int(preds[0])
                    prob = float(pred_class)
                label = "Yes" if pred_class == 1 else "No"
            else:
                preds = model.predict(X_trans)
                pred_val = float(preds[0]) if hasattr(preds, "__len__") else float(preds)
                prob = 1.0
                pred_class = round(pred_val)
                label = f"{pred_val:.2f}"
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
