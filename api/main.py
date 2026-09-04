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
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles

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

# Mount reports directory for serving generated evaluation & EDA plots
os.makedirs("reports/plots", exist_ok=True)
os.makedirs("reports/eda", exist_ok=True)
app.mount("/reports", StaticFiles(directory="reports"), name="reports")


@app.get("/favicon.ico", include_in_schema=False)
@app.get("/favicon.png", include_in_schema=False)
def favicon():
    """Favicon endpoint returning the official ChurnOps favicon."""
    fav_path = os.path.join("reports", "favicon.png")
    if os.path.exists(fav_path):
        return FileResponse(fav_path, media_type="image/png")
    return Response(status_code=404)


@app.get("/logo.png", include_in_schema=False)
def logo_image():
    """Logo endpoint returning the official ChurnOps logo."""
    logo_path = os.path.join("reports", "logo_icon_opt.png")
    if not os.path.exists(logo_path):
        logo_path = os.path.join("reports", "logo_icon.png")
    if os.path.exists(logo_path):
        return FileResponse(logo_path, media_type="image/png")
    return Response(status_code=404)


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

        # Generate EDA plots immediately upon CSV upload
        if detected_target is not None:
            try:
                from src.eda_inspector import generate_eda_report

                generate_eda_report(df, target_col=detected_target, output_dir="reports/eda")
            except Exception as eda_err:
                logger.warning(f"Notice: Upload EDA generation note: {eda_err}")


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


@app.get("/dataset/graphs")
def get_dataset_graphs():
    """Return available dataset diagnostic plots and evaluation charts."""
    graphs = []

    plot_definitions = [
        (
            "confusion_matrix.png",
            "reports/plots",
            "Confusion Matrix",
            "Model Evaluation",
            "True vs Predicted classification decision matrix for holdout test set",
        ),
        (
            "roc_curve.png",
            "reports/plots",
            "ROC Curve",
            "Model Evaluation",
            "Receiver Operating Characteristic (ROC) curve and Area Under Curve (AUC)",
        ),
        (
            "pr_curve.png",
            "reports/plots",
            "Precision-Recall Curve",
            "Model Evaluation",
            "Precision vs Recall curve across decision thresholds",
        ),
        (
            "calibration_curve.png",
            "reports/plots",
            "Probability Calibration",
            "Model Reliability",
            "Reliability diagram comparing predicted vs actual probabilities",
        ),
        (
            "shap_summary.png",
            "reports/plots",
            "SHAP Feature Importance",
            "Explainability (XAI)",
            "SHAP summary chart depicting global feature contributions to predictions",
        ),
        (
            "target_distribution.png",
            "reports/eda",
            "Target Variable Distribution",
            "EDA Inspection",
            "Class frequency distribution or value density of target column",
        ),
        (
            "numerical_correlation_matrix.png",
            "reports/eda",
            "Feature Correlation Matrix",
            "EDA Inspection",
            "Pairwise Pearson correlation heatmap of numerical dataset features",
        ),
    ]

    for filename, directory, title, category, description in plot_definitions:
        filepath = os.path.join(directory, filename)
        if os.path.exists(filepath):
            mtime = int(os.path.getmtime(filepath))
            url_path = f"/{directory}/{filename}?t={mtime}"
            graphs.append({
                "id": filename.replace(".", "_"),
                "title": title,
                "category": category,
                "description": description,
                "url": url_path,
                "filename": filename,
            })

    return {"has_graphs": len(graphs) > 0, "count": len(graphs), "graphs": graphs}




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
    <title>ChurnOps — Production MLOps Studio & Model Monitoring</title>
    <link rel="icon" type="image/png" href="data:image/png;base64,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">
    <link rel="shortcut icon" href="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAYAAACqaXHeAAAetklEQVR4nMV7B3xW5bn4847zjeTLAhL2HiJWkSEOsEncuOr4J16rV2udrVetVmu9tn5Jr1ZbtVq9iqBi1boSt1agCEkEBJGIRYaMsAJkzy/fPO/4/573nJN8KKBe8d43v/Od9eac99nzAPxvDK1JOBymhYVhDlDIDzWVU2c79Cjk+Cx85vddGoEfaGitSWkl0MrKSoDKUpn+QqU1e/aNpcPWrK0b29zZPa67JzJI2npga3sHG1jQfygBTdq6o3tzQiFBCOzLCmU2DinI3lU8dXJdyfnH1zNChEqDobAwzKqryyQhRP+fIyCsNS0vqqZQUyy8a1pr9uC8t0at+nzzqbv2tszqjkZnpFJydEqADyERUoHWZh5IJYEQCoxxIJQCJQQopWBxBgEfswM+a2co5P909LDBK0+aOmbpb685f2M86b2qhIXDk3R5ebn6X0dASUUFqywt0QDEvFxrnXX/aiisXb76tMZ1S8/YuXnTmKTQ/pSQIKUE0AooAQVAFQEKQIjZCKF4YB7h7Ij5k6Cp0pQCYcA4B5/FIdNHZXZ2xqoh+bkVl5x74lvXXlhY77IaDYfD8G0QQQ4LxUkZAJQrHwAs2N4++W/LY1dtblEXNGxYPbJz/WIQXU3AGAUCWhLKNFAXSoJQU1QRLvAENDCc13veO81ghCKLa621UlqD1pQTZoHf74NQkHcPLch95+QZ4554+PaffmIjJkpKGFRWKICDiwb5PjJOKoFCKZFBBvDUyvbC1z7tvmVzKzmvsZvy5KfPgm/PCsX9fkW4j2rtgKRdYBxgEXDqLI+4XGCu7Q+4Mxf1nUGNe42ZZRBClNRKKw2cMR9kZXAYM3jAO8XHT7yn/Fcl63BSSUkJq6yslIcNASUVmlWWEvPAFTs6pjxaFb1v9c7k7L0RP8hIMwS/eE5Y7RuptrIcLe1A6LE4Ltsca/fcARAHcglOZS4SnCWa+y4S+hbsGgBz3aAS1YiSUlHK/CQn5EsdOWrQfy+a/5vfE0JigBaoplx8fwSEqziUFwutdfZNlQ1/WLgx/oudXT6fiHfrAKeK1c6hrGUDAV+mWZJGAJ11mmWSr1LaIMCzZnhuKOtwRBoCHED7rB7qBXy2A/n+HE6ASiE1s/xBGNo/Y8Psk4++5cE7L12CShJ0hUJG/M4I0MjypZWUVJbK19e1nv54VfSJNfvY+J7ubvAxJYEHGf/iFWA7PgTwhwB53qEyIgD3zqIdbKQBaVaAgCEiPMA9BDncYd5v9sxFiAt0LzL3R4L7HlQUUgHj/bKC6sRjxt7z9pM335e0JcG5nskk3xp4QsBHQN/8+p7b390gHtzSLIDrpGBUMcUzCKv/BKy1TwNYQUddmYWls2kfMEBdIF1R8ADxlF4v9V1EefrCiEw6FxDqIsbhf3PoSpsjU8brULZUkJWVQyePH/j8wqfvuJoQIo3mJUR/oycVDmuK2NJa+376wu7nn6uFB7fsjSk/pBSlhCvghCS6gW1+B9CuoWXrYzC0QsbAG/XVSyVvjxPNMc5DXwDn4P5g1ksf+Mw846uK3lOahFqckZ7uLvvTTQ1XFl35p7cWr1mTA2VlBJHAvwn48nIDfOblL9S/+84m3ynRrjbh9wFTCskiDbuzbQuARPaYY7TvjmlziemQygDYK+tmragfeqmG9HOw5N53FJsiQCguoNcj0Igswwmu2ktjYvM/Rj7chxquoKCVJpZlWYloJL55b8Z5f39j9Wy4v/zVQgDOD832lfhyevHTuyoXbPOdkoq2236LWmiDcb2IPxptBbprGRDmM8CbJRHcE+hjMEQKLkyB8WU0URqUUsrAwBBIoJRQ5rC84U4EVuMm8T8VYwyx4ciAeXg62H0s4WHQUbxICOQBDUJqOxAMBScM8n/04ymjljwfDtPqsjJ5YAQg8GXVLMBLxRXP76hYWBeYbcc6bYsRB3jPInM/sJ0fAY02AWp9YrS+K49mkoMIvOhaOqVsQTTllHKLBvw+CPp9YFk0yilrVAANfp9l/tuW2kcIG5KyUwUpyfxJoWkqZSM1gYISxj82fO6x2VcFA/HkcJ0UUviDmda0I4e8vfDpOy5Ds+jqNX1ABBSWVbOPyovFL1/b+ce/rfOXJHo6bR8DywPMkVUNYMeB7Pu0j4/d+46W9ljS3JK2kIxZAZaVFYTMAN+Rn99vZX7//h9Onjhm42nF03f/eNqRLRYjotdQEApJWwbeXrhq0PLPN4/YsrPxhJbW7jMb2rpnxFIkFE8kgYKUjDoc4QhJOhoQeKKFLVVGZoj/aGz/R2pevPs28sxvAKNIdKDgQFbA8elL5VNV+856oMZesLMpJgKWZto8D2mIfrwERS0gkUawVjwARCWNxu6z8Z7cA2pgwnmQ5GQH7AF5WW+cMPWYV+bc/6vFjJD4AVRdulL+mmbzcQLz314+8h9Lay/916bdV7Z0pSYmkogIJQlFH3s/U6htIXV2dojOOHLYH96fe3tYoi/tkP7AfoA2rA9El0HuiQ/WbVy1GwoCJIVqjRq7jkoPz5QEzf1G9tm654HwYK//7pk7pZUUirC8nCwYN2pYxTmn/Pje391S+kVaHMsKC8OkoGCjrqioUK6J/Iqa1yRcVkY2bjyKNDdvIDU1Jrjxgi1f6a8evXr9ln3hfe2JgUokDDe4FlulhKL9crKgcNroG195+OYnVWEh19XVXwuZyf7U1+yNUiJ/+uy2ea9vzrhWxTvR3WXEaCvpyDiaKykc+V//GvCdSwCsjF5FZ94utSCWn48cOnDLqbOm3PzUA7cvSproBFhJSQVUVpYcMkA51ED2ra4GWuO6tQtq1gy+d+7Ch7ftbrm0JxrTnFNhC2UN7B9KnHnCxKueuvf6Vw/mBu+HAMe/B7Vwc9txN74W/biuMQ4BrhiY0BX9BpcDkPooD5QBWzMXaOtGIMzvaF5CMNSV/mAmO3Ls8JeWvzfvRkJIFwJ+3dw1dN6EiAG6pKVFo5jB9xsmEYKI8HMKF93y2B3LP9t+X1dP0ho8IKPr7FlHXfjI3T+rmjbtOqu2dp590IeAh4CSCvZmZak887Et73y4K3g+SXQLAMkJsrwLvEZkKESkBqJsYJ88DjTWBEAtc00IKUJZIX7itKP+vPDlR+/E2N9EYhX7+99muJ7Y90SC56Wi7pD/ftujs/Z0pM4tnjr+pXtuuvALTJt5nHJIBJS4iu+DDc1TfvlK9+pdLXHqZxIVhuPFGdl3qI8cYRyYZDew1f8NRPQAIRykFCKYkcmnHzOu/MPKJ8qkAszZqbKyMnQl9XP1+2a/2528EIDaF+cGXrl86MDlyjVFcDiGif3TQl7MF36LhAjHn+Yn8g0inlvWdX1T3Md9JCK00hSpT7ToU3zuZnx1kTTcYJxfjVqY82Mmjny+6s25BnittSytrDTmZn5z29lPNEX+sTVhAzAODU1dv/z7nsbTAWBJemj9vUZlpUT9UF4NNFwE6tumxbjLQhje9ju6bMNF8Z64DhBN0QMDBB4VoJF7BN45Nl6rsh23FK8oYBNGDdq49M251xIyDznH0bZVVQaxn7R2/WyjrYQ/Fk1K24a6/ILA57FE0WWEfFiyXqP79/0RgDkpB2hVXvPt/4cWlZkYE8req/9xSzKQT2VSaSUoQeoisB7VMVmJG1oAmXJyepSBLWzonxvSF59TeB0hxC4pqehl60L3JWP9vGZkdjaPW75MCGVnCs5Zj6KTtdaByh+RVGFV1SFjkh9y8JrqaqMIPtsZOb+jh2pOpEazh+xvANceIlAZYqSGIuFaBMIk5z42YuiAl8rv/MWKwsJCXllZ2qt0aopN4gQf/6TY1RCszrBKGaiuHDvaUy2CF5y3dvsynUhcRgKBLVBVxXVRkeyNaA+XbviGQaGmWCqteVNH8gSRjBOqpHG1HaoLAGEDINXxGlLe4wzCIKUpzcsOiYsu+skDSgMpKir6mtwhIKgz/3PUkIdWTJkwY+mxE09/99gjLrzMb1+7F8jUWVvb1s/fse8iVlwsDNAmbUg0lFQwLwr8IQfBny31rcNmP1q3ZUebHfQTiUlXJ9RFBHhyj5zgcQVetwISvnyPHZXTWbNuxXtFthBoir51Ph5HfSRyzDXb21+ptzInnSF65jxy9Mhb3Vs+SojjNPzAg+PPK7Ud42PSChIZU5o6CtBQ2d1rpLphf9cVxlggFdeZOQNg+CDr5Voh0ClB7+xrCEAqvvBl3bQNXVHObRvAsiAjGIQJGYHW4VlZ67TWx125tu7xxVbOL85euWtmpDke7BEs55rlDQvnzRx0cxlApMzJ/uofDAFSkOkxG5W7UIDmzwCPesDR/r3nadeFtFkgI6hOOCpr5dsAuqgIVE2a9kXFhjrgN8s+u7yqPfpCW2cncM6BUQYpqWBMgEdWtLWdRAhZHwC4+jebmjrndPtui3XHQEkBrT7/FTcuqWt98rRxv37pr1v8WuvUD4EEij/7WntypBRAjcJDJeeZQAHUtQCG7Y1YpNAKYD6DZPYf1nXnzRfswWegw3OgF2xp6xq+NxLTkXgq1dUT0+1dXbq9u1s0CZ1V19BRgBySACC+eHKTHyVP61hQi2TDvka5PU7ORvO87ZYJSQS+MFxlzDY6bniM+8PCAR0qFZR2EjC5i0mMPvb3LAHqAQchrmusiVbEsny7IOfoTkfXHVhfBSyW8jNKNKaHjNPKgIOknGhtBS0bAUNALpk4YNGyjpat6/Jyx+t4EiYFk3JPyjfxuNfq9vzHhzsfffzUkY8RQhpJuXnsYfEbejkgeHR2FmRg8G47gBpb79p/zwVO0wNaSXQTQdjJDsuwZbg3D/vVoZQyaWgcOBMjSpNP0uhSGKsHp40ZQ8dnZNS/eHywsKRAXnPWILjhsVOGT79rku+s/pb8eFmn/64TXt2184aFO/+otR7xxr8aT7j11S1XPfiPbacEuIf4/5nF4GaRQyxFR1sgdyZNZtdRdKbO0qsIjQ+AjpBnHUwVj6eloA8xvNRG71r3D8T3vfeeoWhBqKABAJ7F47+594IAi56v65wx97PY3aua+V0nPVZ7S2drNKM14YdMzmD2/cvm310cuGFebS3Mux7l9rvpCW5eklSaH5sFcjUBnbSd5IaJ/13nxxMDlxuolgRz7RxIHrc4iFT5ftn4r7FYeq7DTZSmm/iysjKeOvVnZ++N2HkBCv72mIywgE+PzaQdFxWO+nA6IasDAD9Z1tQ989KnGhbWNSVllpWSbVGb+CH759OmTXto3nSyCabNtUrGVKjKtH6EbxoUf4YxbvGBASBTs0AnUkbESK8ooOyjaNhApHNdKUGUSIAtUsPjSTvHrdge8AVu5tcB2s17e5nblFaGAI/U1l/7Tpv11qIGOf+DRpjzcTv9+5IdyZfe3yE+qFm54yTUEYmSCt/xA7NXjM4QG32hPJYU2mKcWU0xBj955OPX5lfXnR5Ye73tAk8wvDelMANj4UG7SbghkI9vYLEkkGn5QOu6QO/sBoK1bqP5UQEqU/BwuAERIgkRSZ1IWP2f+cdn4wCgtuxgwuCxQFpB1PhLhgOcECCp5IT2SEom4jHbVpprWxLbVqKL51ma+sdWlpbW3PTBFr5kUoXv7B/1+1WCJp6sbwkcmUfjnZOHk+WffNlW/GCL9c8pv62qOmlU8N65N89c6nGBz2KQsmtU+UEiJI4/QwL+rSGdgG7QxH/qUEi93A2oiTGfjtU1D/g+UyiBEykj8RRf+NHaqRhKVJeVHdATTC9qmzwsPtPlgN5BIMWxKQSLEKA51g4oJdrHKM3P8OXhlHWfjJcbyyeIWwFWWgBTUloHs/0kvhbjMq2zrnhyzWVrt8R/+15tZMmYS+fXzhiX++dttYuPa+tJnZiTwTdcd1HRA1dcUbrDS4fvh4CzM7K3zrGb5V4BjPQPan7WKGK/sd4Fp0/2PScIjxkoEu3pgYZG6xKLkadryg/uBiPIToocdQvma9NebmyoNpbCq3A5VQ9FhBDQHEk045zqMoCfj1hzYWOPHh1p64yXPFyVdfRdy7tH9OeYcqt48cbjntJaz7v6oYWXrdpEf71o+drXIvX1YNtoveTMx+PxE7XWMwghya8x6NBs/54CShqAMyAJqcnYfsDOGAfaTgAxAZBrHnvDY+MTUG3H9Z7mzpkvfbByNEC5PpCcOTqgT/L73AUNAvqyVU4J3GEHt3KGJhRae1LoZ8C973xRtGx76s0VW6MPb2xjTy7dof+0vtGes3q3fvmOl9aei44RVrLm3zH7xYbKa48NRHfsisdTSYvKeDKRSLZH4kcvXVE7FJ+cvk4K4TBnhETzLWuZLxA09NWxFNCjBgI/cyJokQJIpUyXDnG8QNdLtIkFtuzoigWerqi+FaUcs7UH4wKnAuwUQp2gDzmgLw3gVXk9fnGQgnJBzSRbkMHRWFKlol3JVLxHJKMRYUe7El3dPbK1I5FbU14spl03ho476yZ/RyRBRg7OeScvN8cvFQtmZef4czPY26fMnLYLYU7PFtFwWZnh3anZoQX9OCW2Vk6fEgYHk4YAPW8KaKyLx2LGOmiN1sDdtGSpWLfaun3vNc+9uXwsJiAPxAVupS/tHN2N9GmuafRcBLfW7ZTFHRvCCdhYPiRATCshIYQTApwS7JlihpVCgyN628LHkEKw+NW/3jZr2tifjxyc9+CsaUf9Ys3iv1+Cma+vKmruKa5bRw9e9Mq+5q4GSrN9QmlNKYG4DWTsQGAlM0Et+ATU7iagfmx48BSiIhaRqqkjGpzz0vtP+Sx2enl5NULz1ZYNt9rrNEqYLX2Km4DpLas53UT7lckVYSb5a8rnpu6DyEFFj611Ks2tIhobt9zS13N4ZT1eff6PvcyWvjRajhMrNGOENE8KBRf5M0O4HmmmYbU2boPOywRWUgzsxEmghQ06FnUDJnSaJAMRE1t27Dlt9lXhXwPUiGnXXdfL25JoJZQSUmkhtBYSiJBKCbyWYg51Ndbw0u9rLYTEOXjZVQlCamELIbHQq7SQUgkpnX2vr21G2IEJQE8qCfug0OkqPVgzCMefihKAUgC4aGC/uat6mkt3akUDuHiljWtM3XoWnTUFYPhAUCtqAXbscjLGFgNONYvHY2Lt+u0PXffrB3bMe/i3bw7LPAJbRUSIW9lZOUGe0JozyzJqICUEZGdlQ6bWWFAAqpU/KzeHE6k4Q3EQGpJJybNzcyDTb5s5jFOelZPDU0pz5H2JzZVC86xQCPzcdOhBzcYWSqBcPL14y5TsTBIpPWn8Nsws1RzCM+T4U4otI+EwPX/QgOpndjat3B3MOFH3xJDEzG1YMxlhnUgADCwA9pPZBgG6di2o3TsAUgni81msuaVJLVgWf/nO+5+57E93XfMGemBTB+W8UZBUdYnMfMUsyqlSJKlADMgMQk4osBbf39/vf+LcArkmkWUpEIqCJCAE1aGQIqMHZKzAOaP7Zy4590fkyngU1ZAiEiRQYqncDL81Kkehl0Os10tTt7y8qfy+91rvCXAVv/ftjTf8/oJJL+jr5low7/oDVoeId1ChNUNEPFffcM6929ver29ql36gDKQGItAh0oDHSB2jNZgFkJKg6+tBf7kJyJ4dQDqakHfJwIJ8mDl59A0Vzz80107v6v0BR6afQsmjtX9dsCF1c0trl2kMGF6QCRdOy7ztsSsmP6IKqzhUF2Ef7qGKoxXs3X/7N3nBsn9Vv9spClmkRxKFPo8DOCICkeAUilARIv/6nE6QSARg726Ahr06uXubziOSHjE4e+4zF4z8/YSr725Bv/ysD+7jIwq61eZIRBdhEIRZYEI0tuJUY3/xAUa1Owc9uKKyapMAOWJIFtmdaKYL3wxK7Elub28fcfnT255ZvjV5eizag7VCbDeFhK31gNwQPW2S/3ev3XTcfQIw0VqiDloeD5u2V6I+a++ZfNWaujXrWyMkExQGf05/tuECrA47nrGp59gud5jOBsvco4mkFj1RRbs72bDczF0XzDry8QfOP+oJQjD5YzDNCiflkyKoVuXlmEn65hDW6T6vpNh87nWfYxP2717fdsO7tW33bG6SBWD3CE4csXZa4TSGFSo3N5edMIo88s+7jr8tKYGEw5qUlx+sQUJrVkmIvOfzbb9/ujHxh67WNptramG6EARmixwD4HACZjXc/IxBECom5AwnAlCKS2ErlpeRCWNCesMJI3Pm/eX/TXqNE9K0n1bCyK05nxQiW6QNrFlAzUYNUPnVdvvMq/+86PTNyfy7NrfAjLbOCASYkgS0w61eNdv4G6bdRgSzsvm0IfqFJfechG1ywm0AUwc0DZjQRNYrrVm3YEGnOpN0dQtQmvfKv4sAc27cAUSEqyPczUkrKqBKa9smmD3g2cEgDMnQbUNzfO8fP65fzYXH9Ft8zLD+ezkxEcYBB/J8IMBh7fb2/OcWbZy5au22k1u6RMlea+TwCIQA7Ki0qEkyGO8CEWCob3Iajm9h3G4JtpWZYx03nP2z6vfHXUoIaTeB0YFeaoqM5WX4nAGnLv7XqhWtyTH+eBzRyhB4j/LaKEeH+uZcfUVh4j1cEKYaJbaMaG1rxnxWELI4hYCI9OSHfLsCjGzICbA9tpD1QZqK4Bqa2hJWvyzfxGjcHh7pSYxt604MTwjWrytqQypnBLDMkGIqhboO++CctlwTrLmFK439ME5Th8fn6HsQXxafPox+fseZQ8+5YPH8xoPm0byq7SdNHcfe8ln90trG7twMkdJKEYrUNk6l1G7NxEXAflzgcolBQB8yiNIYS0mRkkQpbAWlwAkm13DByiwaAZKYoRcCpJ0CgfGInQTiC0paMEYTX5CCSlHHI3QbJQ3ru860yVs43TTYYJ/epKmUTMjgoMAZYxLPfHD3zGsPGrwg8OGqKn78wLzP7xzfr2R6fsiOKkap43/2SqRJdLqR3AGx6basGn/CrM1wHeeMMD9V2q9TioiokIkeYccjIhHtMZtIRIW244LKhPDLhLJyCzQdMYlBMMi1FubDCYoC4jZSm623c46ar01wj7kFp73QrE/aEAhMyIsnJw/PqtgvOD3YCFdpXl5MxPt19ec8sKGt8pOGnmBQpATBHgBjFj2R0EYkPPk3e1c3eOe9itMoUifZYlpwMOki07pQvCKMEKafQPcbBiqrv1Oh7u1YcSn/teM+TjDsr700nhaSZ/AJA+jeW0/P/+k1Zx7xkWkDhm8xvCrPqr3NJ4c/a3zx4+bESBnpFpYmDLPl2PdpgHHFwgBszKQDsHPumdF0ZMjeParJXtb1CrKhXND5I0BjA7ZI9bK7I+dusGQ+0nGANWGDJwbmmhEplRRKY9/S9OH80z+cl39J4bGjdpSUaFZZSZy8/HdBgtY9g0o/2D53WXPq/PbWTghojHeAefJvaitpesBBgCujrh4wJtMoS5eaws08G8AlaH8GkAGDQYfynOvIISaT5Ml3GtU9re9uTrM1NutpLYWUtmZ8cD8/zBjGHn3r1il3oS+S3pVC4DuMiooKVlpaKoME4I5lm+94f3vXf23ukH4VjSgfGiFTSXMthIsIBLJPBHD9LicYf8FEXKBtl+KWH1RuPkDeAKCYB0ERMCklD9i+NLPXv+RdM+dIdWzYUUqlbM0ys7Jh0gC16+Ljsm6/++yxr+MyPPvvwUTgOw5jIvGgvFzVNrYf86eVe/5rTVPs/IYuG1Q0hojQWGAFaWJcB0Cs+bkiYRbrcgARjsOCFNfZeaBCuSbGIAJrlJ5JcxufXUob6psNkYJIdBEhEXAtbQncF8iAUdl2qvjI7PlzLh+Dn8y0mn6Div3dYBz/4wYEj40CBOAvtTtmv7ep/a4vW5MnN3bZIGNRYEoLjokJpU05QduOwsO4QVMOmgdABzJAB0MAgUzHirgI20+peVTv/bwAdTkqNoNMxIkStgKpGQsEAjAkmEhOHRF85dazB/xl1oh+X+B/HqoRi8Bh+mQu28/hwRV1Zyze2vYfW1qSZzbIgK+rJwkqFsP+VUmZpcHnp8QKEOULYse4kxoyouIWYpHa6Jya1KHH5irtmjLpY4VdTAJ77inDL8Xy/AyGZCT2HTXEV3Ftce7c00fnf2lyZAeh+mFDgDe8PkM8xkTKNq2PmLOu6+J1jckLtrXbx7TxbH+3TSCZsEEnUwDJhGb4eReWjHFp0nEOXKo6itwrR+KXBViMljiVEbT/Qb8fsrmCPBJrmzQo+Omxw3yv/u6Mwe8SQjrcBbHwpBKdLusHGwQO43AQsUEjR+B5iAHUCT3u2Q3RmRua4ic3dIvpzVExvEexfjEIgi0JZn4AWdiTbWfvfEDH8ZNZCuDTCjJ0HEIk1TYk29o7MGRVTRkaXHHbrH4fBShpSnr0Lazi4aIi9W0A/0EQkC4a1dXVtKa42Mlaui8K+Qh0J1Xem9u7xq1vTY3qSarx3Qk1JGWrAilVRlfCdPlDBtHgZyrhY76m3CBrzrDolmOH+OounJCzNSfIOroT6fCFaUlFGakogUOy+v/ZCGtN0Zs03xseZGDEh4m/9O2QrR/hKo5u+uHoIvv/sB1y7q0T6MYAAAAASUVORK5CYII=">
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
            --accent: #2563eb;
            --accent-hover: #1d4ed8;
            --accent-light: #eff6ff;
            --accent-border: #bfdbfe;
            --dark-btn: #0f172a;
            --dark-btn-hover: #1e293b;
            --success-bg: #f0fdf4;
            --success-text: #166534;
            --success-border: #bbf7d0;
            --warning-bg: #fffbeb;
            --warning-text: #92400e;
            --warning-border: #fde68a;
            --danger-bg: #fef2f2;
            --danger-text: #991b1b;
            --danger-border: #fecaca;
            --radius-sm: 6px;
            --radius-md: 8px;
            --radius-lg: 12px;
            --shadow-sm: 0 1px 2px 0 rgba(15, 23, 42, 0.05);
            --shadow-card: 0 1px 3px 0 rgba(15, 23, 42, 0.04), 0 1px 2px -1px rgba(15, 23, 42, 0.04);
            --shadow-hover: 0 6px 16px -4px rgba(15, 23, 42, 0.08), 0 2px 6px -2px rgba(15, 23, 42, 0.04);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            -webkit-font-smoothing: antialiased;
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
            padding: 0.85rem 2rem;
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .header-content {
            max-width: 1440px;
            margin: 0 auto;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 0.85rem;
        }

        .brand-logo-img {
            height: 44px;
            width: auto;
            object-fit: contain;
            display: block;
        }

        .brand-title {
            font-size: 1.45rem;
            font-weight: 800;
            line-height: 1.1;
            letter-spacing: -0.03em;
            display: flex;
            align-items: center;
        }

        .brand-churn {
            color: #0b1b3d;
        }

        .brand-ops {
            color: #0066ff;
        }

        .brand-subtitle {
            font-size: 0.775rem;
            font-weight: 500;
            color: #64748b;
            letter-spacing: 0.02em;
            margin-top: 0.15rem;
        }

        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.15rem 0.55rem;
            border-radius: 9999px;
            font-size: 0.725rem;
            font-weight: 600;
            background: #f0fdf4;
            color: #166534;
            border: 1px solid #bbf7d0;
        }

        .status-dot {
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: #22c55e;
        }

        .brand-text p {
            font-size: 0.775rem;
            color: var(--text-muted);
        }

        .nav-actions {
            display: flex;
            gap: 0.6rem;
        }

        .btn-link {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.45rem 0.85rem;
            background: #ffffff;
            border: 1px solid var(--border-color);
            border-radius: var(--radius-md);
            color: var(--text-secondary);
            font-size: 0.825rem;
            font-weight: 500;
            text-decoration: none;
            transition: all 0.15s ease;
        }

        .btn-link:hover {
            border-color: var(--border-hover);
            color: var(--text-primary);
            background: #f8fafc;
        }

        .main-container {
            max-width: 1440px;
            margin: 1.75rem auto;
            padding: 0 1.75rem;
        }

        .tabs-nav {
            display: flex;
            gap: 0.35rem;
            border-bottom: 1px solid var(--border-color);
            margin-bottom: 1.75rem;
            overflow-x: auto;
            padding-bottom: 2px;
        }

        .tab-btn {
            padding: 0.65rem 1.1rem;
            background: transparent;
            border: none;
            border-bottom: 2px solid transparent;
            color: var(--text-muted);
            font-weight: 500;
            font-size: 0.875rem;
            cursor: pointer;
            transition: all 0.15s ease;
            white-space: nowrap;
            border-radius: var(--radius-sm) var(--radius-sm) 0 0;
            display: flex;
            align-items: center;
            gap: 0.4rem;
        }

        .tab-btn:hover {
            color: var(--text-primary);
            background: rgba(241, 245, 249, 0.6);
        }

        .tab-btn.active {
            color: var(--accent);
            border-bottom-color: var(--accent);
            font-weight: 600;
            background: transparent;
        }

        .tab-content {
            display: none;
        }

        .tab-content.active {
            display: block;
            animation: fadeInSlideUp 0.22s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        }

        @keyframes fadeInSlideUp {
            from {
                opacity: 0;
                transform: translateY(6px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-lg);
            padding: 1.75rem;
            box-shadow: var(--shadow-card);
            margin-bottom: 1.75rem;
            transition: border-color 0.15s ease;
        }

        .card-header {
            margin-bottom: 1.35rem;
        }

        .card-title {
            font-size: 1.1rem;
            font-weight: 700;
            color: var(--text-primary);
            letter-spacing: -0.015em;
        }

        .card-desc {
            font-size: 0.85rem;
            color: var(--text-muted);
            margin-top: 0.25rem;
        }

        .dropzone {
            border: 2px dashed #cbd5e1;
            border-radius: var(--radius-lg);
            padding: 2.75rem 1.5rem;
            text-align: center;
            background: #fafafa;
            cursor: pointer;
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
            margin-bottom: 1.35rem;
        }

        .dropzone:hover, .dropzone.dragover {
            border-color: var(--accent);
            background: var(--accent-light);
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
        }

        .dropzone input {
            display: none;
        }

        .dropzone-icon {
            font-size: 2.2rem;
            margin-bottom: 0.5rem;
            color: var(--text-secondary);
        }

        .dropzone-text {
            font-weight: 600;
            font-size: 0.95rem;
            color: var(--text-primary);
        }

        .dropzone-sub {
            font-size: 0.825rem;
            color: var(--text-muted);
            margin-top: 0.25rem;
        }

        .btn-primary {
            background: var(--dark-btn);
            color: #ffffff;
            border: 1px solid var(--dark-btn);
            padding: 0.65rem 1.35rem;
            border-radius: var(--radius-md);
            font-weight: 600;
            font-size: 0.875rem;
            cursor: pointer;
            transition: all 0.15s cubic-bezier(0.4, 0, 0.2, 1);
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            box-shadow: var(--shadow-sm);
        }

        .btn-primary:hover {
            background: var(--dark-btn-hover);
            border-color: var(--dark-btn-hover);
        }

        .btn-primary:active {
            transform: scale(0.985);
        }

        .btn-primary:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }

        .btn-block {
            width: 100%;
            justify-content: center;
        }

        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 1.1rem;
            margin-bottom: 1.35rem;
        }

        .form-group {
            display: flex;
            flex-direction: column;
            gap: 0.35rem;
        }

        .form-label {
            font-size: 0.825rem;
            font-weight: 600;
            color: var(--text-secondary);
        }

        .form-control {
            padding: 0.6rem 0.85rem;
            border: 1px solid var(--border-color);
            border-radius: var(--radius-md);
            font-size: 0.875rem;
            color: var(--text-primary);
            background: #ffffff;
            outline: none;
            transition: all 0.15s ease;
        }

        .form-control:focus {
            border-color: var(--accent);
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.12);
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 1rem;
            margin-top: 1rem;
        }

        .metric-tile {
            background: #f8fafc;
            border: 1px solid var(--border-color);
            border-radius: var(--radius-md);
            padding: 1.1rem 1.25rem;
            transition: all 0.15s ease;
        }

        .metric-tile:hover {
            border-color: var(--border-hover);
            background: #ffffff;
            box-shadow: var(--shadow-sm);
        }

        .metric-name {
            font-size: 0.725rem;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .metric-val {
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--text-primary);
            margin-top: 0.35rem;
            letter-spacing: -0.02em;
        }

        .data-table-container {
            overflow-x: auto;
            max-height: 380px;
            border: 1px solid var(--border-color);
            border-radius: var(--radius-md);
            margin-top: 1rem;
            background: #ffffff;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.85rem;
            text-align: left;
        }

        th {
            background: #f8fafc;
            color: var(--text-secondary);
            font-weight: 600;
            font-size: 0.775rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            padding: 0.75rem 1rem;
            position: sticky;
            top: 0;
            border-bottom: 1px solid var(--border-color);
            z-index: 10;
        }

        td {
            padding: 0.65rem 1rem;
            border-bottom: 1px solid #f1f5f9;
            color: var(--text-primary);
            white-space: nowrap;
            font-variant-numeric: tabular-nums;
        }

        tr:hover td {
            background: #f8fafc;
        }

        .badge {
            display: inline-flex;
            align-items: center;
            padding: 0.2rem 0.55rem;
            border-radius: var(--radius-sm);
            font-size: 0.75rem;
            font-weight: 600;
            line-height: 1;
        }

        .badge-success { background: var(--success-bg); color: var(--success-text); border: 1px solid var(--success-border); }
        .badge-warning { background: var(--warning-bg); color: var(--warning-text); border: 1px solid var(--warning-border); }
        .badge-danger { background: var(--danger-bg); color: var(--danger-text); border: 1px solid var(--danger-border); }
        .badge-info { background: var(--accent-light); color: #1e40af; border: 1px solid var(--accent-border); }

        .json-preview {
            background: #0f172a;
            color: #38bdf8;
            padding: 1.1rem;
            border-radius: var(--radius-md);
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            font-size: 0.825rem;
            line-height: 1.6;
            max-height: 280px;
            overflow-y: auto;
        }

        .alert {
            padding: 0.85rem 1.1rem;
            border-radius: var(--radius-md);
            font-size: 0.875rem;
            margin-bottom: 1.1rem;
            font-weight: 500;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .alert-info { background: var(--accent-light); color: #1e40af; border: 1px solid var(--accent-border); }
        .alert-success { background: var(--success-bg); color: var(--success-text); border: 1px solid var(--success-border); }
        .alert-warning { background: var(--warning-bg); color: var(--warning-text); border: 1px solid var(--warning-border); }
        .alert-danger { background: var(--danger-bg); color: var(--danger-text); border: 1px solid var(--danger-border); }

        .graph-card {
            background: #ffffff;
            border: 1px solid var(--border-color);
            border-radius: var(--radius-lg);
            padding: 1.1rem;
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
            cursor: pointer;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }

        .graph-card:hover {
            border-color: var(--accent);
            box-shadow: var(--shadow-hover);
            transform: translateY(-2px);
        }

        @keyframes modalScaleIn {
            from {
                opacity: 0;
                transform: scale(0.96);
            }
            to {
                opacity: 1;
                transform: scale(1);
            }
        }
    </style>
</head>
<body>
    <header class="header">
        <div class="header-content">
            <div class="brand">
                <img src="data:image/png;base64,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" alt="ChurnOps Logo" class="brand-logo-img">
                <div class="brand-text">
                    <h1 class="brand-title"><span class="brand-churn">Churn</span><span class="brand-ops">Ops</span></h1>
                    <p class="brand-subtitle">Predict &bull; Monitor &bull; Optimize</p>
                </div>
            </div>
        </div>
    </header>

    <main class="main-container">
        <nav class="tabs-nav">
            <button class="tab-btn active" onclick="switchTab('tab-upload', event)">📁 1. Dataset Upload & Train</button>
            <button class="tab-btn" onclick="switchTab('tab-graphs', event)">📊 2. Diagnostic & EDA Graphs <span id="graphBadge" class="badge badge-warning" style="display:none; margin-left:0.3rem;">0</span></button>
            <button class="tab-btn" onclick="switchTab('tab-row', event)">🎯 3. Predict Dataset Row</button>
            <button class="tab-btn" onclick="switchTab('tab-batch', event)">⚡ 4. Batch CSV Inference</button>
            <button class="tab-btn" onclick="switchTab('tab-custom', event)">📝 5. Custom Single Form</button>
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

                    <div style="margin-bottom: 1.35rem;">
                        <button class="btn-primary" id="trainBtn" onclick="trainModel()"><span id="trainSpinner">&#9889;</span> Train Leak-Free Model Suite</button>
                    </div>

                    <h3 style="font-size: 0.95rem; font-weight: 700; margin-bottom: 0.5rem; color: var(--text-primary);">Dataset Preview (First 10 Rows)</h3>
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

                <div id="inlineGraphsContainer" style="display: none; margin-top: 2rem; border-top: 1px solid var(--border-color); padding-top: 1.5rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                        <h3 style="font-size: 1.05rem; font-weight: 700; color: var(--text-primary);">📊 Evaluation & Interpretability Charts</h3>
                        <button class="btn-link" onclick="switchTab('tab-graphs')">View Full Gallery →</button>
                    </div>
                    <div id="inlineGraphsGrid" style="display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 1.25rem;"></div>
                </div>
            </div>
        </div>

        <!-- TAB 2: DIAGNOSTIC & EDA GRAPHS -->
        <div id="tab-graphs" class="tab-content">
            <div class="card">
                <div class="card-header" style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 1rem;">
                    <div>
                        <h2 class="card-title">Dataset & Model Diagnostic Graphs</h2>
                        <p class="card-desc">Visual evaluation charts, EDA distributions, feature correlations, and SHAP explainability plots for the trained dataset.</p>
                    </div>
                    <button class="btn-link" onclick="loadDatasetGraphs()">🔄 Refresh Graphs</button>
                </div>

                <div id="graphsEmptyState" class="alert alert-warning" style="display: none; margin-top: 1rem;">
                    ⚠️ <strong>No diagnostic graphs generated yet.</strong> Upload a CSV dataset and click <em>Train Leak-Free Model Suite</em> to generate EDA distributions, Confusion Matrices, ROC Curves, and SHAP explainability charts.
                </div>

                <div id="graphsGalleryGrid" style="display: grid; grid-template-columns: repeat(auto-fill, minmax(350px, 1fr)); gap: 1.5rem; margin-top: 1.25rem;"></div>
            </div>
        </div>

        <!-- TAB 3: ROW PREDICTOR -->
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

                <div style="margin-bottom: 1.25rem;">
                    <button class="btn-primary" id="predictRowBtn" onclick="predictRow()"><span id="predictRowSpinner">&#9889;</span> Predict Selected Row</button>
                </div>

                <div id="rowOutputContainer" style="display: none;">
                    <div class="metrics-grid" id="rowResultMetrics" style="margin-bottom: 1.35rem;"></div>

                    <h3 style="font-size: 0.95rem; font-weight: 700; margin-bottom: 0.5rem; color: var(--text-primary);">Clean Row Features JSON (ID & Target Stripped)</h3>
                    <pre class="json-preview" id="rowJsonPreview"></pre>
                </div>
            </div>
        </div>

        <!-- TAB 4: BATCH CSV -->
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

                <button class="btn-primary" id="runBatchBtn" onclick="runBatchPrediction()">&#9889; Run Batch Prediction</button>
                <div id="batchStatus" style="margin-top: 1rem;"></div>

                <div id="batchResultsContainer" style="display: none; margin-top: 1.35rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                        <h3 style="font-size: 0.95rem; font-weight: 700; color: var(--text-primary);">Batch Predictions Output</h3>
                        <button class="btn-link" onclick="downloadBatchCSV()">&#128229; Download CSV</button>
                    </div>
                    <div class="data-table-container">
                        <table id="batchTable"></table>
                    </div>
                </div>
            </div>
        </div>

        <!-- TAB 5: CUSTOM SINGLE INPUT -->
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

                <div id="customResult" style="display: none; margin-top: 1.35rem;">
                    <h3 style="font-size: 0.95rem; font-weight: 700; margin-bottom: 0.5rem; color: var(--text-primary);">Prediction Output</h3>
                    <pre class="json-preview" id="customJsonOutput"></pre>
                </div>
            </div>
        </div>
    </main>

    <!-- Graph Modal Lightbox -->
    <div id="graphModal" onclick="closeGraphModal(event)" style="display: none; position: fixed; inset: 0; background: rgba(15, 23, 42, 0.75); z-index: 1000; align-items: center; justify-content: center; padding: 1.5rem; backdrop-filter: blur(6px);">
        <div style="background: #ffffff; border-radius: 14px; max-width: 940px; width: 100%; max-height: 90vh; overflow: auto; padding: 1.75rem; position: relative; box-shadow: 0 25px 50px -12px rgba(15, 23, 42, 0.25); border: 1px solid var(--border-color);">
            <button onclick="closeGraphModalDirect()" style="position: absolute; top: 1.1rem; right: 1.1rem; background: #f1f5f9; border: 1px solid #e2e8f0; font-size: 1.2rem; font-weight: bold; width: 34px; height: 34px; border-radius: 50%; cursor: pointer; color: var(--text-secondary); display: flex; align-items: center; justify-content: center; transition: all 0.15s ease;">&times;</button>
            <h3 id="modalGraphTitle" style="font-size: 1.2rem; font-weight: 700; color: var(--text-primary); margin-bottom: 0.25rem;"></h3>
            <p id="modalGraphDesc" style="font-size: 0.85rem; color: var(--text-muted); margin-bottom: 1.1rem;"></p>
            <div style="text-align: center; background: #f8fafc; border-radius: 10px; padding: 1rem; border: 1px solid var(--border-color);">
                <img id="modalGraphImg" src="" alt="Graph View" style="max-width: 100%; max-height: 65vh; object-fit: contain; border-radius: 6px;">
            </div>
        </div>
    </div>

    <script>
        let activeDataset = null;
        let batchResultsData = null;

        async function ensureActiveDataset() {
            if (activeDataset && activeDataset.preview) return activeDataset;
            try {
                const res = await fetch('/dataset/preview');
                const data = await res.json();
                if (data.has_dataset) {
                    activeDataset = data;
                    renderDatasetInfo(data);
                    return activeDataset;
                }
            } catch(e) {
                console.error('Error fetching active dataset:', e);
            }
            return null;
        }

        async function switchTab(tabId, evt) {
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

            if (evt && evt.currentTarget) {
                evt.currentTarget.classList.add('active');
            }
            const targetEl = document.getElementById(tabId);
            if (targetEl) {
                targetEl.classList.add('active');
            }

            if (tabId === 'tab-row') {
                await ensureActiveDataset();
                loadRowPreview();
            }
            if (tabId === 'tab-batch') {
                await ensureActiveDataset();
            }
            if (tabId === 'tab-graphs') {
                loadDatasetGraphs();
            }
            if (tabId === 'tab-health') {
                loadHealthTelemetry();
            }
        }

        async function initPage() {
            setupDragAndDrop();
            await ensureActiveDataset();
            loadDatasetGraphs();
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
                    loadDatasetGraphs();
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

            loadDatasetGraphs();
            card.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }

        async function loadDatasetGraphs() {
            try {
                const res = await fetch('/dataset/graphs');
                const data = await res.json();
                const gallery = document.getElementById('graphsGalleryGrid');
                const inlineGrid = document.getElementById('inlineGraphsGrid');
                const inlineContainer = document.getElementById('inlineGraphsContainer');
                const emptyState = document.getElementById('graphsEmptyState');
                const badge = document.getElementById('graphBadge');

                if (!data.has_graphs || !data.graphs || !data.graphs.length) {
                    if (emptyState) emptyState.style.display = 'block';
                    if (gallery) gallery.innerHTML = '';
                    if (inlineGrid) inlineGrid.innerHTML = '';
                    if (inlineContainer) inlineContainer.style.display = 'none';
                    if (badge) badge.style.display = 'none';
                    return;
                }

                if (emptyState) emptyState.style.display = 'none';
                if (inlineContainer) inlineContainer.style.display = 'block';
                if (badge) {
                    badge.innerText = data.count;
                    badge.style.display = 'inline-block';
                }

                let html = '';
                data.graphs.forEach(g => {
                    const badgeClass = g.category === 'Explainability (XAI)' ? 'badge-warning' : (g.category === 'EDA Inspection' ? 'badge-info' : 'badge-success');
                    const cleanDesc = g.description ? g.description.replace(/'/g, "\\'") : '';
                    html += `
                        <div class="graph-card" onclick="openGraphModal('${g.title}', '${cleanDesc}', '${g.url}')">
                            <div>
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                                    <span class="badge ${badgeClass}">${g.category}</span>
                                    <span style="font-size: 0.75rem; color: var(--text-muted); font-weight: 500;">🔍 Expand</span>
                                </div>
                                <h4 style="font-size: 0.95rem; font-weight: 700; color: var(--text-primary); margin-bottom: 0.25rem;">${g.title}</h4>
                                <p style="font-size: 0.8rem; color: var(--text-muted); margin-bottom: 0.75rem; min-height: 2.4em; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;">${g.description}</p>
                            </div>
                            <div style="background: #f8fafc; border-radius: 8px; border: 1px solid var(--border-color); overflow: hidden; height: 220px; display: flex; align-items: center; justify-content: center; padding: 0.5rem;">
                                <img src="${g.url}" alt="${g.title}" style="max-width: 100%; max-height: 100%; object-fit: contain; border-radius: 4px;">
                            </div>
                        </div>
                    `;
                });

                if (gallery) gallery.innerHTML = html;
                if (inlineGrid) inlineGrid.innerHTML = html;
            } catch(e) {
                console.error('Error loading graphs:', e);
            }
        }

        function openGraphModal(title, desc, url) {
            document.getElementById('modalGraphTitle').innerText = title;
            document.getElementById('modalGraphDesc').innerText = desc;
            document.getElementById('modalGraphImg').src = url;
            document.getElementById('graphModal').style.display = 'flex';
        }

        function closeGraphModal(e) {
            if (e.target && e.target.id === 'graphModal') {
                document.getElementById('graphModal').style.display = 'none';
            }
        }

        function closeGraphModalDirect() {
            document.getElementById('graphModal').style.display = 'none';
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
                            <div class="metric-tile"><div class="metric-name">Predicted Value</div><div class="metric-val" style="color: #2563eb;">${data.churn_label}</div></div>
                            <div class="metric-tile"><div class="metric-name">Actual Target Value</div><div class="metric-val">${actualVal}</div></div>
                            <div class="metric-tile"><div class="metric-name">Model Version</div><div class="metric-val">${data.model_version}</div></div>
                            <div class="metric-tile"><div class="metric-name">Inference Latency</div><div class="metric-val">${data.processing_time_ms} ms</div></div>
                        `;
                    } else {
                        grid.innerHTML = `
                            <div class="metric-tile"><div class="metric-name">Predicted Label</div><div class="metric-val" style="color: #2563eb;">${data.churn_label} (${data.churn_prediction})</div></div>
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
            const btn = document.getElementById('runBatchBtn');
            const statusDiv = document.getElementById('batchStatus');

            if (btn) btn.disabled = true;
            if (statusDiv) statusDiv.innerHTML = '<div class="alert alert-info">&#8987; Fetching dataset and generating batch predictions...</div>';

            const ds = await ensureActiveDataset();
            if (!ds || !ds.preview || !ds.preview.length) {
                if (btn) btn.disabled = false;
                if (statusDiv) statusDiv.innerHTML = '<div class="alert alert-warning">&#9888; No active dataset loaded. Please upload a CSV dataset on Tab 1 (Dataset Upload & Train) first.</div>';
                return;
            }

            const limit = parseInt(document.getElementById('batchLimit').value) || 50;
            const samples = ds.preview.slice(0, limit);

            try {
                const res = await fetch('/predict/batch', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ customers: samples })
                });

                const data = await res.json();
                if (btn) btn.disabled = false;

                if (res.ok) {
                    if (statusDiv) statusDiv.innerHTML = '<div class="alert alert-success">&#9989; Batch predictions completed for ' + (data.total_processed || samples.length) + ' records in ' + (data.processing_time_ms || 0) + ' ms.</div>';
                    batchResultsData = data.results;
                    renderTable('batchTable', data.results);
                    document.getElementById('batchResultsContainer').style.display = 'block';
                    document.getElementById('batchResultsContainer').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                } else {
                    if (statusDiv) statusDiv.innerHTML = '<div class="alert alert-danger">&#10060; Batch prediction failed: ' + (data.detail || 'Server error') + '</div>';
                }
            } catch (err) {
                if (btn) btn.disabled = false;
                if (statusDiv) statusDiv.innerHTML = '<div class="alert alert-danger">&#10060; Error executing batch prediction: ' + err.message + '</div>';
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
