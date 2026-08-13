# ⚡ ChurnOps: Production MLOps Pipeline for Customer Churn Prediction

**ChurnOps** is an end-to-end, production-grade MLOps system built to automate data validation, model training, experiment tracking, model registry management, REST API serving, continuous integration/retraining, and data drift monitoring for customer churn prediction.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    subgraph Data & Pipeline Layer
        A[Telco Churn Dataset] --> B[Data Validation src/data_validation.py]
        B --> C[Preprocessing Pipeline src/preprocessing.py]
    end

    subgraph Training & MLflow Tracking
        C --> D[Multi-Model Trainer src/train.py]
        D -->|Train Log Params & Metrics| E[(MLflow Tracking Store)]
        D -->|Evaluate Val F1 Score| F{Best Model Selector}
        F -->|Register & Stage| G[(MLflow Model Registry - Staging)]
    end

    subgraph Evaluation & Promotion
        G --> H[Evaluator src/evaluate.py]
        H -->|Compare vs Prod Test Set| I{Candidate F1 > Prod F1?}
        I -->|Yes| J[(MLflow Model Registry - Production)]
        I -->|No| K[Keep Current Production Model]
    end

    subgraph Serving & Monitoring
        J --> L[FastAPI App api/main.py]
        L -->|Post Requests| M[Inference /predict Endpoint]
        M -->|Log Request Payload| N[(SQLite DB predictions.db)]
        N --> O[PSI Drift Monitor monitoring/drift_check.py]
        N --> P[Streamlit Dashboard monitoring/dashboard.py]
    end

    subgraph CI/CD Automation
        Q[GitHub Actions push/PR] --> R[Lint & Pytest]
        R --> S[Retrain Pipeline retrain_pipeline.py]
        S --> T[Docker Build api/Dockerfile]
    end
```

---

## 📁 Repository Structure

```
churnops/
├── data/
│   ├── raw/                       # Raw input datasets (tracked via DVC)
│   ├── processed/                 # Processed test sets for candidate evaluation
│   └── generate_dataset.py        # Synthetic Telco Customer Churn dataset generator
├── src/
│   ├── data_validation.py         # Schema, null, and numerical range validators
│   ├── preprocessing.py           # Reproducible sklearn ColumnTransformer pipeline
│   ├── train.py                   # Multi-model training, MLflow logging, Staging promotion
│   └── evaluate.py                # Candidate vs Production evaluation and promotion
├── api/
│   ├── main.py                    # FastAPI application loading Production MLflow model
│   ├── schemas.py                 # Pydantic input/output validation models
│   └── Dockerfile                 # Multi-stage production container configuration
├── monitoring/
│   ├── drift_check.py             # PSI & KS-test data drift detector
│   └── dashboard.py               # Streamlit real-time monitoring dashboard
├── pipelines/
│   └── retrain_pipeline.py        # End-to-end retraining & evaluation pipeline
├── tests/
│   ├── test_data_validation.py    # Unit tests for data schema & boundary validation
│   ├── test_preprocessing.py     # Unit tests for feature transformer pipeline
│   └── test_api.py                # FastAPI endpoint unit tests with TestClient
├── .github/workflows/
│   └── ci-cd.yml                  # GitHub Actions workflow (lint, test, retrain, docker)
├── dvc.yaml                       # DVC pipeline declaration
├── requirements.txt               # Pinned Python dependencies
└── README.md                      # Documentation
```

---

## 🚀 Quickstart & Setup Guide

### 1. Prerequisites & Installation

- Python 3.11+
- Git & Docker (optional for container serving)

Clone the repository and install dependencies:
```bash
# Install dependencies
pip install -r requirements.txt
```

---

## 🛠️ Step-by-Step Execution Workflow

### Phase 1: Generate Dataset & Train Models with MLflow

1. **Generate Dataset**:
   ```bash
   python data/generate_dataset.py
   ```

2. **Train Models & Log to MLflow**:
   ```bash
   python src/train.py
   ```
   *Trains Logistic Regression, Random Forest, and XGBoost. Logs parameters, metrics (F1, ROC-AUC, Precision, Recall), confusion matrices, and model artifacts to MLflow, and registers the top model to Stage `"Staging"`.*

3. **View MLflow Tracking UI**:
   ```bash
   mlflow ui --port 5000
   ```
   Open `http://localhost:5000` to inspect experiment runs, metrics, and registered models.

---

### Phase 2: Model Evaluation & Stage Promotion

Run evaluation comparing the Staging candidate model against current Production model:
```bash
python src/evaluate.py
```
*If candidate F1 score exceeds the current Production model F1 score, the model is automatically promoted to `"Production"` in MLflow Registry.*

---

### Phase 3: Serve Predictions via FastAPI API

Launch the production REST API:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

- **Health Check**: `GET http://localhost:8000/health`
- **Interactive Swagger Docs**: `http://localhost:8000/docs`
- **Prediction Request Example**:
  ```bash
  curl -X POST "http://localhost:8000/predict" \
       -H "Content-Type: application/json" \
       -d '{
         "gender": "Female",
         "SeniorCitizen": 0,
         "Partner": "Yes",
         "Dependents": "No",
         "tenure": 12,
         "PhoneService": "Yes",
         "MultipleLines": "No",
         "InternetService": "DSL",
         "OnlineSecurity": "No",
         "OnlineBackup": "Yes",
         "DeviceProtection": "No",
         "TechSupport": "No",
         "StreamingTV": "No",
         "StreamingMovies": "No",
         "Contract": "Month-to-month",
         "PaperlessBilling": "Yes",
         "PaymentMethod": "Electronic check",
         "MonthlyCharges": 65.50,
         "TotalCharges": 786.00
       }'
  ```

---

### Phase 4: Containerized Deployment (Docker)

Build and run the API using Docker:
```bash
# Build Docker image
docker build -t churnops-api:latest -f api/Dockerfile .

# Run Docker container
docker run -d -p 8000:8000 --name churnops-service churnops-api:latest
```

---

### Phase 5: Automated Retraining Pipeline

Execute the full automated retraining cycle:
```bash
python pipelines/retrain_pipeline.py
```

---

### Phase 6: Monitoring & Streamlit Dashboard

1. **Run Population Stability Index (PSI) Data Drift Check**:
   ```bash
   python monitoring/drift_check.py
   ```

2. **Launch Streamlit Monitoring Dashboard**:
   ```bash
   streamlit run monitoring/dashboard.py
   ```
   Open `http://localhost:8501` to view prediction volume, live churn rate, active model version, PSI drift alerts, and feature distribution comparison plots.

---

### Phase 7: Automated Unit Testing

Run the comprehensive pytest suite:
```bash
pytest tests/ -v
```

---

## 🔄 CI/CD Retrain-and-Promote Flow

The GitHub Actions workflow (`.github/workflows/ci-cd.yml`) automates quality control and model lifecycle management:

1. **Linting & Testing**: Runs `ruff check .` and `pytest` on every push/PR.
2. **Retraining & Promotion**: On push to `main`, executes `pipelines/retrain_pipeline.py` to train new candidates on incoming data, evaluates candidate metrics against the active Production model on a held-out test set, and only promotes to `"Production"` if F1 performance is superior.
3. **Docker Build**: Builds and validates the container image for seamless deployment.
