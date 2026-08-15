"""
Streamlit page: production monitoring dashboard.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import sqlite3

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from monitoring.drift_check import DB_PATH, TRAIN_DATA_PATH, run_drift_analysis

st.set_page_config(
    page_title="ChurnOps | Monitoring",
    page_icon="📊",
    layout="wide",
)

st.markdown("""
<style>
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        margin-bottom: 15px;
    }
    .metric-title {
        font-size: 0.95rem;
        color: #8b9bb4;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #ffffff;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)


def load_prediction_logs() -> pd.DataFrame:
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query("SELECT * FROM predictions ORDER BY timestamp DESC", conn)
        conn.close()
        return df
    except Exception:
        conn.close()
        return pd.DataFrame()


def load_training_data() -> pd.DataFrame:
    if os.path.exists(TRAIN_DATA_PATH):
        return pd.read_csv(TRAIN_DATA_PATH)
    return pd.DataFrame()


st.title("📊 Production Monitoring Dashboard")
st.markdown("Real-time inference tracking, model performance metrics, and data drift detection.")

st.sidebar.header("⚙️ Dashboard Controls")
st.sidebar.button("🔄 Refresh Data", use_container_width=True)

df_logs = load_prediction_logs()
df_ref = load_training_data()

if df_logs.empty:
    st.warning(
        "No prediction logs found in SQLite (`monitoring/predictions.db`). "
        "Use the FastAPI `/predict` endpoint or **Train and Predict** page to generate predictions."
    )
    st.stop()

df_logs["timestamp"] = pd.to_datetime(df_logs["timestamp"])

col1, col2, col3, col4 = st.columns(4)

total_reqs = len(df_logs)
churn_count = (df_logs["churn_prediction"] == 1).sum()
churn_rate = (churn_count / total_reqs * 100) if total_reqs > 0 else 0.0
avg_prob = df_logs["churn_probability"].mean() if total_reqs > 0 else 0.0
active_version = df_logs["model_version"].iloc[0] if not df_logs.empty else "N/A"

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Total Inferences</div>
        <div class="metric-value">{total_reqs:,}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Live Churn Rate</div>
        <div class="metric-value">{churn_rate:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Avg Churn Probability</div>
        <div class="metric-value">{avg_prob:.3f}</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Active Model Version</div>
        <div class="metric-value" style="color: #6366f1;">v{active_version}</div>
    </div>
    """, unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📈 Prediction Analytics", "🧪 Data Drift & PSI", "🤖 Model Metadata"])

with tab1:
    st.subheader("Inference Request Timeline")

    col_chart1, col_chart2 = st.columns([2, 1])

    with col_chart1:
        df_ts = df_logs.set_index("timestamp").resample("1h")["id"].count().reset_index()
        df_ts.columns = ["Timestamp", "Requests"]
        st.line_chart(df_ts, x="Timestamp", y="Requests", color="#6366f1")

    with col_chart2:
        st.subheader("Prediction Class Breakdown")
        churn_labels = df_logs["churn_prediction"].map({1: "Churn (Yes)", 0: "Stay (No)"}).value_counts()
        st.bar_chart(churn_labels, color="#ec4899")

    st.subheader("Recent Prediction Requests (Latest 10)")
    st.dataframe(
        df_logs[
            [
                "timestamp",
                "customerID" if "customerID" in df_logs else "id",
                "Contract",
                "InternetService",
                "MonthlyCharges",
                "churn_prediction",
                "churn_probability",
                "model_version",
            ]
        ].head(10),
        use_container_width=True,
    )

with tab2:
    st.subheader("Population Stability Index (PSI) Data Drift Monitor")

    drift_report = run_drift_analysis()

    if drift_report.get("status") == "success":
        max_psi = drift_report["max_psi"]
        alert = drift_report["drift_alert"]

        st.markdown(f"**Max Feature PSI Score:** `{max_psi:.4f}`")
        if alert:
            st.error("🚨 **CRITICAL DRIFT ALERT TRIGGERED** (PSI > 0.25). Model retraining recommended!")
        elif max_psi > 0.10:
            st.warning("⚠️ **MODERATE DATA DRIFT DETECTED** (0.10 <= PSI < 0.25).")
        else:
            st.success("✅ **DATA DISTRIBUTION STABLE** (PSI < 0.10).")

        st.subheader("Feature Drift Metrics Breakdown")
        metrics_df = []
        for feat, info in drift_report["feature_metrics"].items():
            metrics_df.append({
                "Feature": feat,
                "Type": info["type"],
                "PSI Score": info["psi"],
                "KS Statistic": info["ks_stat"] if info["ks_stat"] is not None else "-",
                "Status": info["status"],
            })
        st.table(pd.DataFrame(metrics_df))

        st.subheader("Feature Distribution Comparison")
        selected_feat = st.selectbox(
            "Select Feature to Inspect:",
            ["MonthlyCharges", "tenure", "TotalCharges", "Contract", "InternetService"],
        )

        if selected_feat in df_ref.columns and selected_feat in df_logs.columns:
            fig, ax = plt.subplots(figsize=(8, 4))
            if selected_feat in ["MonthlyCharges", "tenure", "TotalCharges"]:
                ref_vals = pd.to_numeric(df_ref[selected_feat], errors="coerce").dropna()
                log_vals = pd.to_numeric(df_logs[selected_feat], errors="coerce").dropna()
                sns.kdeplot(ref_vals, label="Reference (Training)", color="#3b82f6", fill=True, alpha=0.3, ax=ax)
                sns.kdeplot(log_vals, label="Actual (Production)", color="#f43f5e", fill=True, alpha=0.3, ax=ax)
                ax.set_title(f"Distribution Comparison - {selected_feat}")
                ax.legend()
            else:
                df_comp = pd.DataFrame({
                    "Reference": df_ref[selected_feat].value_counts(normalize=True),
                    "Production": df_logs[selected_feat].value_counts(normalize=True),
                }).fillna(0)
                df_comp.plot(kind="bar", ax=ax, color=["#3b82f6", "#f43f5e"])
                ax.set_title(f"Categorical Proportions - {selected_feat}")
                ax.set_ylabel("Proportion")
            st.pyplot(fig)

with tab3:
    st.subheader("MLflow Model Registry & Deployment Stage Status")
    st.json({
        "Model Name": "ChurnOps-Model",
        "Active Stage": "Production / Staging",
        "MLflow Tracking URI": os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"),
        "SQLite Log Database": DB_PATH,
        "Reference Dataset": TRAIN_DATA_PATH,
    })
