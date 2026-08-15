"""
Automated retraining and model promotion pipeline.
"""

import os
import sys

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

# Ensure workspace root directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd

from data.generate_dataset import generate_telco_churn_data
from src.data_validation import validate_data
from src.evaluate import compare_and_promote
from src.train import train_and_evaluate


def run_retraining_pipeline():
    print("=== Starting ChurnOps Automated Retraining Pipeline ===")
    
    # 1. Fetch / Generate Data
    data_path = "data/raw/telco_churn.csv"
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}. Generating dataset...")
        df = generate_telco_churn_data()
        os.makedirs("data/raw", exist_ok=True)
        df.to_csv(data_path, index=False)
    else:
        df = pd.read_csv(data_path)

    # 1.5 Validate Data
    print("Validating dataset...")
    validate_data(df, is_training=True)
    print("Data validation successful.")

    # 2. Train and register candidate model in Staging
    print("\n--- Training Candidate Models ---")
    train_result = train_and_evaluate(data_path=data_path)
    print(f"Training completed. Best Staging Version: {train_result['version']} (Val F1: {train_result['best_f1']:.4f})")

    # 3. Evaluate against current Production model and promote if superior
    print("\n--- Evaluating Candidate Model against Production ---")
    promoted, report = compare_and_promote(promote=True)
    
    print("\n=== Retraining Pipeline Summary ===")
    if "error" in report:
        print(f"Promotion Status: {report['error']}")
    else:
        print(f"Candidate Model Version: {report.get('candidate_version', 'N/A')}")
        if report.get("candidate_metrics"):
            print(f"Candidate F1 Score: {report['candidate_metrics'].get('f1_score', 0.0):.4f}")
        if report.get("production_version"):
            print(f"Production Model Version: {report['production_version']}")
            if report.get("production_metrics"):
                print(f"Production F1 Score: {report['production_metrics'].get('f1_score', 0.0):.4f}")
        else:
            print("Production Model: None (First deployment)")
        print(f"Model Promoted to Production: {'YES' if promoted else 'NO'}")
    return report


if __name__ == "__main__":
    try:
        run_retraining_pipeline()
        sys.exit(0)
    except Exception as e:
        print(f"Pipeline failed with error: {e}")
        sys.exit(1)
