# 🚀 ChurnOps Cloud Deployment Guide

This guide provides step-by-step instructions to deploy the **ChurnOps MLOps Suite** (FastAPI Serving API, Streamlit Monitoring Dashboard, MLflow Registry) to production cloud platforms.

---

## 📋 Table of Contents
1. [Option 1: Deploy on Render (Recommended - Free Tier Available)](#1-deploy-on-render-recommended)
2. [Option 2: Deploy on Railway](#2-deploy-on-railway)
3. [Option 3: Deploy on Google Cloud Run / AWS ECS (Docker Container)](#3-deploy-on-google-cloud-run--aws-ecs)
4. [Option 4: Local Docker & Multi-Container Deployment](#4-local-docker--multi-container-deployment)
5. [Environment Variables Reference](#environment-variables-reference)

---

## 1. Deploy on Render (Recommended)

Render supports automated deployment using the included [`render.yaml`](file:///c:/Users/Bisht/OneDrive/Desktop/churnops/render.yaml) Blueprint file.

### One-Click Blueprint Deployment:
1. Push your repository to **GitHub**:
   ```bash
   git push origin main
   ```
2. Log in to [Render Dashboard](https://dashboard.render.com/).
3. Click **New +** → **Blueprint**.
4. Connect your GitHub repository (`bishtprateek270-hue/churnops`).
5. Render will automatically detect [`render.yaml`](file:///c:/Users/Bisht/OneDrive/Desktop/churnops/render.yaml) and provision two services:
   - **`churnops-api`**: FastAPI Serving Endpoint (`https://churnops-api.onrender.com`)
   - **`churnops-dashboard`**: Streamlit Analytics Dashboard (`https://churnops-dashboard.onrender.com`)
6. Click **Apply**. Deployment will complete in ~2-3 minutes.

### Manual Web Service Creation on Render:
- **FastAPI API**:
  - Environment: `Docker`
  - Dockerfile Path: `api/Dockerfile`
  - Health Check Path: `/health`
- **Streamlit Dashboard**:
  - Environment: `Docker`
  - Dockerfile Path: `api/Dockerfile`
  - Start Command: `streamlit run monitoring/dashboard.py --server.port $PORT --server.address 0.0.0.0`

---

## 2. Deploy on Railway

1. Go to [Railway.app](https://railway.app/) and create a **New Project**.
2. Select **Deploy from GitHub Repo** → Choose `churnops`.
3. Railway will automatically build using `api/Dockerfile`.
4. To deploy both API and Dashboard services:
   - Duplicate the service card in Railway.
   - For Service 1 (API): Set Custom Start Command to `uvicorn api.main:app --host 0.0.0.0 --port $PORT`.
   - For Service 2 (Dashboard): Set Custom Start Command to `streamlit run monitoring/dashboard.py --server.port $PORT --server.address 0.0.0.0`.
5. Under **Variables**, add optional parameters (e.g. `ENABLE_CORS=true`, `RATE_LIMIT_PER_MINUTE=100`).

---

## 3. Deploy on Google Cloud Run / AWS ECS

### Step 1: Build & Push Docker Image
```bash
# Build production Docker image
docker build -f api/Dockerfile -t churnops-app:latest .

# Tag for Google Container Registry (GCR) or AWS ECR
docker tag churnops-app:latest gcr.io/YOUR_PROJECT_ID/churnops-app:latest

# Push to Container Registry
docker push gcr.io/YOUR_PROJECT_ID/churnops-app:latest
```

### Step 2: Deploy to GCP Cloud Run
```bash
gcloud run deploy churnops-api \
  --image gcr.io/YOUR_PROJECT_ID/churnops-app:latest \
  --platform managed \
  --allow-unauthenticated \
  --port 8000 \
  --memory 2Gi
```

---

## 4. Local Docker & Multi-Container Deployment

To test full production orchestration locally (API, Streamlit, and MLflow UI concurrently):

```bash
# Build and start containers in detached mode
docker-compose up --build -d

# Verify container health
docker-compose ps
```

### Accessible Ports:
- **FastAPI API**: [http://localhost:8000](http://localhost:8000)
- **FastAPI Interactive Docs (Swagger)**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Streamlit Dashboard**: [http://localhost:8501](http://localhost:8501)
- **MLflow Tracking Server**: [http://localhost:5000](http://localhost:5000)

---

## Environment Variables Reference

| Variable Name | Default Value | Description |
| :--- | :--- | :--- |
| `MLFLOW_TRACKING_URI` | `file:./mlruns` | MLflow tracking backend location |
| `PREDICTIONS_DB_PATH` | `monitoring/predictions.db` | SQLite inference logging database path |
| `RATE_LIMIT_PER_MINUTE` | `100` | FastAPI request rate limit per client IP |
| `ENABLE_CORS` | `true` | Enable Cross-Origin Resource Sharing |
| `CORS_ORIGINS` | `*` | Allowed CORS origins (comma-separated) |
| `COST_FN` | `500.0` | False Negative cost weight for decision optimization |
| `COST_FP` | `50.0` | False Positive cost weight for decision optimization |
