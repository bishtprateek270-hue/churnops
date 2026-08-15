"""
Streamlit page: upload dataset, train model, and run custom predictions.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd
import streamlit as st

from monitoring.predict_utils import (
    UPLOAD_PATH,
    load_trained_artifacts,
    predict_customers,
    predict_single_row,
    run_training,
    save_uploaded_dataset,
    validate_upload,
)
from src.data_validation import REQUIRED_COLUMNS

st.set_page_config(page_title="ChurnOps | Train & Predict", page_icon="🎯", layout="wide")

st.title("🎯 Train & Predict on Your Dataset")
st.markdown(
    "Upload a Telco-style churn CSV, train a model on your data, then inspect predictions "
    "for any customer row you choose."
)

if "uploaded_df" not in st.session_state:
    st.session_state.uploaded_df = None
if "train_result" not in st.session_state:
    st.session_state.train_result = None
if "model_ready" not in st.session_state:
    st.session_state.model_ready = os.path.exists("models/best_model.joblib")

st.subheader("1. Upload Dataset")
uploaded_file = st.file_uploader(
    "Upload CSV (Telco Customer Churn format)",
    type=["csv"],
    help="CSV must include feature columns like gender, tenure, Contract, MonthlyCharges, etc. "
    "Include a Churn column for supervised training.",
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.session_state.uploaded_df = df
    st.success(f"Loaded **{len(df):,}** rows and **{len(df.columns)}** columns.")
    st.dataframe(df.head(10), use_container_width=True)

    is_valid, message = validate_upload(df)
    if is_valid:
        st.info(message)
    else:
        st.error(message)

    if st.button("💾 Save Dataset", use_container_width=False):
        path = save_uploaded_dataset(df)
        st.success(f"Saved to `{path}`")

st.divider()
st.subheader("2. Train Model")

if st.session_state.uploaded_df is None and not os.path.exists(UPLOAD_PATH):
    st.warning("Upload a dataset first, or use the sample Kaggle dataset at `data/raw/telco_churn.csv`.")
else:
    train_source = UPLOAD_PATH if os.path.exists(UPLOAD_PATH) else "data/raw/telco_churn.csv"
    st.caption(f"Training will use: `{train_source}`")

    if st.button("🚀 Train Model", type="primary", use_container_width=True):
        if st.session_state.uploaded_df is not None:
            if "Churn" not in st.session_state.uploaded_df.columns:
                st.error("Training requires a `Churn` column (values: Yes/No).")
                st.stop()
            save_uploaded_dataset(st.session_state.uploaded_df)
            train_source = UPLOAD_PATH

        with st.spinner("Training Logistic Regression, Random Forest, and XGBoost..."):
            try:
                result = run_training(data_path=train_source)
                st.session_state.train_result = result
                st.session_state.model_ready = True
                st.success("Training complete!")
            except Exception as exc:
                st.error(f"Training failed: {exc}")

if st.session_state.train_result:
    result = st.session_state.train_result
    col1, col2, col3 = st.columns(3)
    col1.metric("Best Model", result["best_model_name"].replace("_", " "))
    col2.metric("Validation F1", f"{result['best_f1']:.4f}")
    col3.metric("MLflow Version", result["version"])

st.divider()
st.subheader("3. Run Predictions")

if not st.session_state.model_ready:
    st.info("Train a model first to unlock predictions.")
    st.stop()

try:
    model, preprocessor = load_trained_artifacts()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

predict_tab_single, predict_tab_batch, predict_tab_form = st.tabs(
    ["Pick from Dataset", "Batch Predict", "Manual Input"]
)

source_df = st.session_state.uploaded_df
if source_df is None and os.path.exists(UPLOAD_PATH):
    source_df = pd.read_csv(UPLOAD_PATH)
elif source_df is None and os.path.exists("data/raw/telco_churn.csv"):
    source_df = pd.read_csv("data/raw/telco_churn.csv")

with predict_tab_single:
    if source_df is None:
        st.warning("Upload a dataset to pick customers from it.")
    else:
        row_idx = st.number_input(
            "Select customer row index",
            min_value=0,
            max_value=len(source_df) - 1,
            value=0,
            step=1,
        )
        selected = source_df.iloc[int(row_idx)]
        st.json(selected.drop(labels=["Churn"], errors="ignore").to_dict())

        if st.button("Predict Selected Customer", key="predict_single"):
            prediction = predict_single_row(selected, model=model, preprocessor=preprocessor)
            actual = selected.get("Churn", "Unknown")

            c1, c2, c3 = st.columns(3)
            c1.metric("Predicted Churn", prediction["churn_label"])
            c2.metric("Churn Probability", f"{prediction['churn_probability']:.2%}")
            if actual != "Unknown":
                c3.metric("Actual Churn", actual)

            if prediction["churn_label"] == "Yes":
                st.error("⚠️ High churn risk — consider retention offer.")
            else:
                st.success("✅ Customer likely to stay.")

with predict_tab_batch:
    if source_df is None:
        st.warning("Upload a dataset for batch predictions.")
    else:
        sample_size = st.slider("Rows to predict", min_value=5, max_value=min(500, len(source_df)), value=min(50, len(source_df)))
        if st.button("Run Batch Prediction", key="predict_batch"):
            batch_results = predict_customers(source_df.head(sample_size), model=model, preprocessor=preprocessor)
            st.dataframe(batch_results, use_container_width=True)

            if "prediction_correct" in batch_results.columns:
                accuracy = batch_results["prediction_correct"].mean()
                st.metric("Prediction Accuracy on Sample", f"{accuracy:.2%}")

            churn_rate = (batch_results["churn_prediction"] == 1).mean()
            st.metric("Predicted Churn Rate", f"{churn_rate:.2%}")

with predict_tab_form:
    st.markdown("Enter customer details manually:")
    with st.form("manual_predict_form"):
        col_a, col_b, col_c = st.columns(3)
        gender = col_a.selectbox("Gender", ["Female", "Male"])
        senior = col_b.selectbox("Senior Citizen", [0, 1])
        partner = col_c.selectbox("Partner", ["Yes", "No"])
        dependents = col_a.selectbox("Dependents", ["Yes", "No"])
        tenure = col_b.number_input("Tenure (months)", min_value=0, max_value=120, value=12)
        phone = col_c.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = col_a.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet = col_b.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = col_c.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup = col_a.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device_protection = col_b.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_support = col_c.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        streaming_tv = col_a.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = col_b.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        contract = col_c.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = col_a.selectbox("Paperless Billing", ["Yes", "No"])
        payment = col_b.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        )
        monthly = col_c.number_input("Monthly Charges", min_value=0.0, max_value=300.0, value=65.0)
        total = col_a.number_input("Total Charges", min_value=0.0, value=780.0)

        submitted = st.form_submit_button("Predict", use_container_width=True)

    if submitted:
        manual_row = pd.Series({
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple_lines,
            "InternetService": internet,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
        })
        prediction = predict_single_row(manual_row, model=model, preprocessor=preprocessor)
        st.json(prediction)

st.caption(f"Required feature columns: {', '.join(REQUIRED_COLUMNS)}")
