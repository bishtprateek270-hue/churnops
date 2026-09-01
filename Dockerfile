FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create application directories
RUN mkdir -p /app/src /app/api /app/models /app/data /app/mlruns /app/monitoring /app/pipelines

# Copy application files
COPY src/ /app/src/
COPY api/ /app/api/
COPY data/ /app/data/
COPY models/ /app/models/
COPY mlruns/ /app/mlruns/
COPY monitoring/ /app/monitoring/
COPY pipelines/ /app/pipelines/

EXPOSE 8000 8501 5000

ENV PYTHONPATH=/app
ENV MLFLOW_TRACKING_URI=file:/app/mlruns

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
