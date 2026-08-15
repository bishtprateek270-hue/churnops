"""
Streamlit home page for ChurnOps dashboard.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st

st.set_page_config(
    page_title="ChurnOps | Home",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("⚡ ChurnOps MLOps Platform")
st.markdown(
    """
Welcome to **ChurnOps** — train churn models on your own data and monitor live predictions.

Use the sidebar to navigate:

- **Monitoring** — live inference metrics, drift detection, and model metadata
- **Train and Predict** — upload a CSV, train a model, and run custom predictions

### Quick links
- FastAPI docs: [http://localhost:8000/docs](http://localhost:8000/docs)
- Health check: [http://localhost:8000/health](http://localhost:8000/health)
"""
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Monitoring")
    st.markdown(
        "Track prediction volume, churn rate, PSI drift alerts, and recent API requests."
    )

with col2:
    st.subheader("🎯 Train & Predict")
    st.markdown(
        "Upload your Telco churn CSV, train Logistic Regression / Random Forest / XGBoost, "
        "then predict churn for any customer row."
    )

if os.path.exists("data/raw/telco_churn.csv"):
    st.success("Sample Kaggle dataset is available at `data/raw/telco_churn.csv`.")
if os.path.exists("models/best_model.joblib"):
    st.info("A trained model is ready. Open **Train and Predict** to run inference.")
