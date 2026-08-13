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

# Copy application files
COPY src/ /app/src/
COPY api/ /app/api/
COPY models/ /app/models/
COPY data/ /app/data/
COPY mlruns/ /app/mlruns/

EXPOSE 8000

ENV PYTHONPATH=/app
ENV MLFLOW_TRACKING_URI=file:/app/mlruns

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
