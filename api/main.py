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

import httpx

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
from fastapi.responses import HTMLResponse, JSONResponse
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
        logger.info("No pre-trained model found. Initializing initial baseline model...")
        try:
            from src.train import train_and_evaluate

            train_and_evaluate(fast_mode=True)
            if os.path.exists(fallback_path):
                model_store["model"] = joblib.load(fallback_path)
                model_store["preprocessor"] = load_preprocessor(settings.PREPROCESSOR_PATH)
                model_store["version"] = "local-auto-1.0"
                model_store["stage"] = "AutoTrained"
                model_store["loaded_at"] = datetime.now(timezone.utc).isoformat()
                logger.info("Successfully trained and loaded initial baseline model.")
        except Exception as exc:
            logger.error(f"Failed to auto-train initial baseline model: {exc}")


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
    <title>ChurnOps - Customer Churn Prediction System</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-dark: #0f172a;
            --card-bg: rgba(30, 41, 59, 0.7);
            --border-color: rgba(255, 255, 255, 0.1);
            --primary-accent: #6366f1;
            --primary-gradient: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #d946ef 100%);
            --danger-gradient: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
            --success-gradient: linear-gradient(135deg, #10b981 0%, #059669 100%);
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        body {
            background-color: var(--bg-dark);
            background-image:
                radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.15) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(217, 70, 239, 0.15) 0px, transparent 50%);
            color: var(--text-main);
            min-height: 100vh;
            padding: 2rem 1rem;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
        }

        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 2rem;
            margin-bottom: 2rem;
            border-bottom: 1px solid var(--border-color);
            flex-wrap: wrap;
            gap: 1rem;
        }

        .logo-group {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }

        .logo-badge {
            background: var(--primary-gradient);
            padding: 0.5rem 0.85rem;
            border-radius: 12px;
            font-weight: 800;
            font-size: 1.25rem;
            box-shadow: 0 10px 25px -5px rgba(99, 102, 241, 0.4);
        }

        .logo-title h1 {
            font-size: 1.5rem;
            font-weight: 700;
            letter-spacing: -0.02em;
        }

        .logo-title p {
            color: var(--text-muted);
            font-size: 0.85rem;
        }

        .nav-links {
            display: flex;
            gap: 0.75rem;
            flex-wrap: wrap;
        }

        .nav-btn {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border-color);
            color: var(--text-main);
            padding: 0.5rem 1rem;
            border-radius: 8px;
            text-decoration: none;
            font-size: 0.875rem;
            font-weight: 500;
            transition: all 0.2s ease;
        }

        .nav-btn:hover {
            background: rgba(255, 255, 255, 0.1);
            border-color: var(--primary-accent);
            transform: translateY(-1px);
        }

        .preset-bar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: var(--card-bg);
            backdrop-filter: blur(12px);
            border: 1px solid var(--border-color);
            padding: 1rem 1.5rem;
            border-radius: 16px;
            margin-bottom: 2rem;
            flex-wrap: wrap;
            gap: 1rem;
        }

        .preset-title {
            font-weight: 600;
            font-size: 0.95rem;
            color: var(--text-muted);
        }

        .preset-btns {
            display: flex;
            gap: 0.75rem;
        }

        .btn-secondary {
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid var(--border-color);
            color: var(--text-main);
            padding: 0.4rem 0.85rem;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 500;
            font-size: 0.85rem;
            transition: all 0.2s ease;
        }

        .btn-secondary:hover {
            background: rgba(255, 255, 255, 0.15);
        }

        .main-grid {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 2rem;
        }

        @media (max-width: 900px) {
            .main-grid {
                grid-template-columns: 1fr;
            }
        }

        .form-card {
            background: var(--card-bg);
            backdrop-filter: blur(16px);
            border: 1px solid var(--border-color);
            border-radius: 20px;
            padding: 2rem;
            box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.5);
        }

        .section-header {
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 1.25rem;
            color: var(--primary-accent);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .form-section {
            margin-bottom: 2rem;
        }

        .fields-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.25rem;
        }

        .field-group {
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
        }

        label {
            font-size: 0.825rem;
            font-weight: 500;
            color: var(--text-muted);
        }

        select, input[type="number"] {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 0.6rem 0.8rem;
            color: var(--text-main);
            font-size: 0.9rem;
            outline: none;
            transition: border-color 0.2s ease;
        }

        select:focus, input[type="number"]:focus {
            border-color: var(--primary-accent);
        }

        .btn-submit {
            background: var(--primary-gradient);
            color: white;
            border: none;
            width: 100%;
            padding: 1rem;
            border-radius: 12px;
            font-size: 1.05rem;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 10px 25px -5px rgba(99, 102, 241, 0.4);
            margin-top: 1rem;
        }

        .btn-submit:hover {
            opacity: 0.95;
            transform: translateY(-2px);
            box-shadow: 0 15px 30px -5px rgba(99, 102, 241, 0.6);
        }

        .result-card {
            background: var(--card-bg);
            backdrop-filter: blur(16px);
            border: 1px solid var(--border-color);
            border-radius: 20px;
            padding: 2rem;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            position: sticky;
            top: 2rem;
            height: fit-content;
            box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.5);
        }

        .gauge-container {
            width: 160px;
            height: 160px;
            border-radius: 50%;
            background: conic-gradient(var(--primary-accent) 0%, rgba(255, 255, 255, 0.1) 0%);
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 1.5rem 0;
            position: relative;
            transition: background 0.8s ease;
        }

        .gauge-inner {
            width: 130px;
            height: 130px;
            background: var(--bg-dark);
            border-radius: 50%;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }

        .gauge-value {
            font-size: 2rem;
            font-weight: 800;
        }

        .gauge-label {
            font-size: 0.75rem;
            color: var(--text-muted);
            text-transform: uppercase;
        }

        .status-badge {
            padding: 0.5rem 1.25rem;
            border-radius: 50px;
            font-weight: 700;
            font-size: 0.9rem;
            margin-bottom: 1.5rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .badge-danger {
            background: var(--danger-gradient);
            color: white;
            box-shadow: 0 10px 20px -5px rgba(239, 68, 68, 0.5);
        }

        .badge-success {
            background: var(--success-gradient);
            color: white;
            box-shadow: 0 10px 20px -5px rgba(16, 185, 129, 0.5);
        }

        .meta-list {
            width: 100%;
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
            border-top: 1px solid var(--border-color);
            padding-top: 1.25rem;
            font-size: 0.85rem;
        }

        .meta-item {
            display: flex;
            justify-content: space-between;
            color: var(--text-muted);
        }

        .meta-item span:last-child {
            color: var(--text-main);
            font-weight: 600;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="logo-group">
                <div class="logo-badge">⚡</div>
                <div class="logo-title">
                    <h1>ChurnOps</h1>
                    <p>Enterprise Customer Churn Prediction Engine</p>
                </div>
            </div>
            <div class="nav-links">
                <a href="/docs" target="_blank" class="nav-btn">🚀 Swagger API Docs</a>
                <a href="/health" target="_blank" class="nav-btn">💓 Health Probe</a>
                <a href="/metrics" target="_blank" class="nav-btn">📊 Prometheus Metrics</a>
                <a href="https://github.com/bishtprateek270-hue/churnops" target="_blank" class="nav-btn">🐙 GitHub Repository</a>
            </div>
        </header>

        <div class="preset-bar">
            <span class="preset-title">Fill Quick Demo Profiles:</span>
            <div class="preset-btns">
                <button type="button" class="btn-secondary" onclick="loadPreset('high')">⚠️ High Churn Risk Customer</button>
                <button type="button" class="btn-secondary" onclick="loadPreset('low')">✅ Low Churn Risk Customer</button>
            </div>
        </div>

        <div class="main-grid">
            <div class="form-card">
                <form id="churnForm">
                    <div class="form-section">
                        <div class="section-header">👤 Customer Profile & Demographics</div>
                        <div class="fields-grid">
                            <div class="field-group">
                                <label for="gender">Gender</label>
                                <select id="gender">
                                    <option value="Female">Female</option>
                                    <option value="Male">Male</option>
                                </select>
                            </div>
                            <div class="field-group">
                                <label for="SeniorCitizen">Senior Citizen</label>
                                <select id="SeniorCitizen">
                                    <option value="0">No (0)</option>
                                    <option value="1">Yes (1)</option>
                                </select>
                            </div>
                            <div class="field-group">
                                <label for="Partner">Partner</label>
                                <select id="Partner">
                                    <option value="Yes">Yes</option>
                                    <option value="No">No</option>
                                </select>
                            </div>
                            <div class="field-group">
                                <label for="Dependents">Dependents</label>
                                <select id="Dependents">
                                    <option value="No">No</option>
                                    <option value="Yes">Yes</option>
                                </select>
                            </div>
                        </div>
                    </div>

                    <div class="form-section">
                        <div class="section-header">📶 Subscribed Services</div>
                        <div class="fields-grid">
                            <div class="field-group">
                                <label for="tenure">Tenure (Months)</label>
                                <input type="number" id="tenure" value="12" min="0" max="100">
                            </div>
                            <div class="field-group">
                                <label for="PhoneService">Phone Service</label>
                                <select id="PhoneService">
                                    <option value="Yes">Yes</option>
                                    <option value="No">No</option>
                                </select>
                            </div>
                            <div class="field-group">
                                <label for="MultipleLines">Multiple Lines</label>
                                <select id="MultipleLines">
                                    <option value="No">No</option>
                                    <option value="Yes">Yes</option>
                                    <option value="No phone service">No phone service</option>
                                </select>
                            </div>
                            <div class="field-group">
                                <label for="InternetService">Internet Service</label>
                                <select id="InternetService">
                                    <option value="Fiber optic">Fiber optic</option>
                                    <option value="DSL">DSL</option>
                                    <option value="No">No</option>
                                </select>
                            </div>
                            <div class="field-group">
                                <label for="OnlineSecurity">Online Security</label>
                                <select id="OnlineSecurity">
                                    <option value="No">No</option>
                                    <option value="Yes">Yes</option>
                                    <option value="No internet service">No internet service</option>
                                </select>
                            </div>
                            <div class="field-group">
                                <label for="OnlineBackup">Online Backup</label>
                                <select id="OnlineBackup">
                                    <option value="No">No</option>
                                    <option value="Yes">Yes</option>
                                    <option value="No internet service">No internet service</option>
                                </select>
                            </div>
                            <div class="field-group">
                                <label for="DeviceProtection">Device Protection</label>
                                <select id="DeviceProtection">
                                    <option value="No">No</option>
                                    <option value="Yes">Yes</option>
                                    <option value="No internet service">No internet service</option>
                                </select>
                            </div>
                            <div class="field-group">
                                <label for="TechSupport">Tech Support</label>
                                <select id="TechSupport">
                                    <option value="No">No</option>
                                    <option value="Yes">Yes</option>
                                    <option value="No internet service">No internet service</option>
                                </select>
                            </div>
                            <div class="field-group">
                                <label for="StreamingTV">Streaming TV</label>
                                <select id="StreamingTV">
                                    <option value="No">No</option>
                                    <option value="Yes">Yes</option>
                                    <option value="No internet service">No internet service</option>
                                </select>
                            </div>
                            <div class="field-group">
                                <label for="StreamingMovies">Streaming Movies</label>
                                <select id="StreamingMovies">
                                    <option value="No">No</option>
                                    <option value="Yes">Yes</option>
                                    <option value="No internet service">No internet service</option>
                                </select>
                            </div>
                        </div>
                    </div>

                    <div class="form-section">
                        <div class="section-header">💳 Billing & Contract Information</div>
                        <div class="fields-grid">
                            <div class="field-group">
                                <label for="Contract">Contract Type</label>
                                <select id="Contract">
                                    <option value="Month-to-month">Month-to-month</option>
                                    <option value="One year">One year</option>
                                    <option value="Two year">Two year</option>
                                </select>
                            </div>
                            <div class="field-group">
                                <label for="PaperlessBilling">Paperless Billing</label>
                                <select id="PaperlessBilling">
                                    <option value="Yes">Yes</option>
                                    <option value="No">No</option>
                                </select>
                            </div>
                            <div class="field-group">
                                <label for="PaymentMethod">Payment Method</label>
                                <select id="PaymentMethod">
                                    <option value="Electronic check">Electronic check</option>
                                    <option value="Mailed check">Mailed check</option>
                                    <option value="Bank transfer (automatic)">Bank transfer (automatic)</option>
                                    <option value="Credit card (automatic)">Credit card (automatic)</option>
                                </select>
                            </div>
                            <div class="field-group">
                                <label for="MonthlyCharges">Monthly Charges ($)</label>
                                <input type="number" id="MonthlyCharges" value="70.35" step="0.01">
                            </div>
                            <div class="field-group">
                                <label for="TotalCharges">Total Charges ($)</label>
                                <input type="number" id="TotalCharges" value="844.20" step="0.01">
                            </div>
                        </div>
                    </div>

                    <button type="submit" class="btn-submit">⚡ Run Live Churn Prediction</button>
                </form>
            </div>

            <div class="result-card">
                <h3>Prediction Analytics</h3>

                <div class="gauge-container" id="gauge">
                    <div class="gauge-inner">
                        <div class="gauge-value" id="probValue">--%</div>
                        <div class="gauge-label">Probability</div>
                    </div>
                </div>

                <div class="status-badge badge-success" id="statusBadge">Awaiting Input</div>

                <div class="meta-list">
                    <div class="meta-item">
                        <span>Prediction Outcome:</span>
                        <span id="labelValue">Pending</span>
                    </div>
                    <div class="meta-item">
                        <span>Inference Latency:</span>
                        <span id="latencyValue">-- ms</span>
                    </div>
                    <div class="meta-item">
                        <span>Active Model Stage:</span>
                        <span id="stageValue">Production</span>
                    </div>
                    <div class="meta-item">
                        <span>Model Version:</span>
                        <span id="versionValue">v1.0</span>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const highRisk = {
            gender: "Female", SeniorCitizen: 1, Partner: "No", Dependents: "No", tenure: 2,
            PhoneService: "Yes", MultipleLines: "No", InternetService: "Fiber optic",
            OnlineSecurity: "No", OnlineBackup: "No", DeviceProtection: "No", TechSupport: "No",
            StreamingTV: "Yes", StreamingMovies: "Yes", Contract: "Month-to-month",
            PaperlessBilling: "Yes", PaymentMethod: "Electronic check", MonthlyCharges: 95.70, TotalCharges: 191.40
        };

        const lowRisk = {
            gender: "Male", SeniorCitizen: 0, Partner: "Yes", Dependents: "Yes", tenure: 60,
            PhoneService: "Yes", MultipleLines: "Yes", InternetService: "DSL",
            OnlineSecurity: "Yes", OnlineBackup: "Yes", DeviceProtection: "Yes", TechSupport: "Yes",
            StreamingTV: "Yes", StreamingMovies: "Yes", Contract: "Two year",
            PaperlessBilling: "No", PaymentMethod: "Credit card (automatic)", MonthlyCharges: 85.10, TotalCharges: 5106.00
        };

        function loadPreset(type) {
            const data = type === 'high' ? highRisk : lowRisk;
            for (const key in data) {
                const el = document.getElementById(key);
                if (el) el.value = data[key];
            }
            document.getElementById('churnForm').dispatchEvent(new Event('submit'));
        }

        document.getElementById('churnForm').addEventListener('submit', async (e) => {
            e.preventDefault();

            const payload = {
                gender: document.getElementById('gender').value,
                SeniorCitizen: parseInt(document.getElementById('SeniorCitizen').value),
                Partner: document.getElementById('Partner').value,
                Dependents: document.getElementById('Dependents').value,
                tenure: parseInt(document.getElementById('tenure').value),
                PhoneService: document.getElementById('PhoneService').value,
                MultipleLines: document.getElementById('MultipleLines').value,
                InternetService: document.getElementById('InternetService').value,
                OnlineSecurity: document.getElementById('OnlineSecurity').value,
                OnlineBackup: document.getElementById('OnlineBackup').value,
                DeviceProtection: document.getElementById('DeviceProtection').value,
                TechSupport: document.getElementById('TechSupport').value,
                StreamingTV: document.getElementById('StreamingTV').value,
                StreamingMovies: document.getElementById('StreamingMovies').value,
                Contract: document.getElementById('Contract').value,
                PaperlessBilling: document.getElementById('PaperlessBilling').value,
                PaymentMethod: document.getElementById('PaymentMethod').value,
                MonthlyCharges: parseFloat(document.getElementById('MonthlyCharges').value),
                TotalCharges: parseFloat(document.getElementById('TotalCharges').value)
            };

            try {
                const startTime = performance.now();
                const res = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                const data = await res.json();
                const endTime = performance.now();
                const latency = Math.round(endTime - startTime);

                if (res.ok) {
                    const prob = Math.round(data.churn_probability * 100);
                    document.getElementById('probValue').innerText = prob + '%';
                    document.getElementById('labelValue').innerText = data.churn_label === 'Yes' ? 'HIGH CHURN RISK' : 'LOW CHURN RISK';
                    document.getElementById('latencyValue').innerText = (data.processing_time_ms || latency) + ' ms';
                    document.getElementById('versionValue').innerText = data.model_version || '1.0';

                    const gauge = document.getElementById('gauge');
                    const badge = document.getElementById('statusBadge');

                    if (data.churn_prediction === 1) {
                        gauge.style.background = `conic-gradient(#ef4444 ${prob}%, rgba(255, 255, 255, 0.1) 0%)`;
                        badge.className = 'status-badge badge-danger';
                        badge.innerText = 'HIGH CHURN RISK';
                    } else {
                        gauge.style.background = `conic-gradient(#10b981 ${prob}%, rgba(255, 255, 255, 0.1) 0%)`;
                        badge.className = 'status-badge badge-success';
                        badge.innerText = 'LOW CHURN RISK';
                    }
                } else {
                    alert('Prediction failed: ' + (data.detail || 'Unknown error'));
                }
            } catch (err) {
                alert('Error connecting to API: ' + err.message);
            }
        });

        // Initial prediction on load
        window.addEventListener('load', () => {
            document.getElementById('churnForm').dispatchEvent(new Event('submit'));
        });
    </script>
</body>
</html>"""
    return HTMLResponse(content=html_content)


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
