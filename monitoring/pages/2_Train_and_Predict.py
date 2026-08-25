import importlib
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import pandas as pd
import streamlit as st

import monitoring.predict_utils
from monitoring.predict_utils import (
    UPLOAD_PATH,
    load_trained_artifacts,
    predict_customers,
    predict_single_row,
    run_training,
    save_uploaded_dataset,
    validate_upload,
)
from src.preprocessing import find_target_col, infer_task_type, is_identifier_column

importlib.reload(monitoring.predict_utils)

st.set_page_config(page_title="ChurnOps | Train & Predict", page_icon="🎯", layout="wide")

st.title("🎯 Dataset-Agnostic Train & Predict Studio")
st.markdown(
    "Upload **any** classification or regression CSV dataset. Select your target column, train a leak-free model suite "
    "with Optuna tuning, and run custom or batch predictions."
)

if "uploaded_df" not in st.session_state:
    st.session_state.uploaded_df = None
if "target_col" not in st.session_state:
    st.session_state.target_col = None
if "train_result" not in st.session_state:
    st.session_state.train_result = None
if "model_ready" not in st.session_state:
    st.session_state.model_ready = os.path.exists("models/unified_pipeline.joblib") or os.path.exists("models/best_model.joblib")

st.subheader("1. Upload Dataset & Target Selection")
uploaded_file = st.file_uploader(
    "Upload CSV Dataset",
    type=["csv"],
    help="Upload any dataset CSV with feature columns and a target column (classification or regression).",
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.session_state.uploaded_df = df
        st.success(f"Loaded **{len(df):,}** rows and **{len(df.columns)}** columns.")
        st.dataframe(df.head(10), use_container_width=True)

        detected_target = find_target_col(df)
        target_options = [c for c in df.columns if not is_identifier_column(df, c)]
        if not target_options:
            target_options = list(df.columns)
        default_idx = target_options.index(detected_target) if detected_target in target_options else 0

        st.session_state.target_col = st.selectbox(
            "Select Target Column for Supervised Training:",
            options=target_options,
            index=default_idx,
            help="Select the column containing the target variable to predict.",
        )

        inferred_task = infer_task_type(df[st.session_state.target_col])
        is_id_target = is_identifier_column(df, st.session_state.target_col)
        if is_id_target:
            st.warning(f"⚠️ Column `{st.session_state.target_col}` appears to be a unique row identifier, not a predictable target variable.")
            allow_id_target = st.checkbox(
                "I understand and want to train on this identifier column anyway",
                value=False,
                key="id_target_chk"
            )
        else:
            allow_id_target = False
            st.info(f"💡 Auto-detected Task Type: **{inferred_task.upper()}** for target `{st.session_state.target_col}`.")

        is_valid, message = validate_upload(df, target_col=st.session_state.target_col)
        if is_valid:
            st.success(message)
        else:
            st.error(message)

        if st.button("💾 Save Dataset", use_container_width=False):
            path = save_uploaded_dataset(df)
            st.success(f"Saved to `{path}`")
    except Exception as exc:
        st.error(f"⚠️ Error reading uploaded CSV: {exc}")

st.divider()
st.subheader("2. Train Model Suite")

if st.session_state.uploaded_df is None and not os.path.exists(UPLOAD_PATH):
    st.warning("Upload a dataset first, or use the sample dataset at `data/raw/telco_churn.csv`.")
else:
    train_source = UPLOAD_PATH if os.path.exists(UPLOAD_PATH) else "data/raw/telco_churn.csv"
    st.caption(f"Training will use: `{train_source}`")

    mode_selection = st.radio(
        "Select Training Speed & Engine Mode:",
        options=["🚀 Fast Training Mode (Recommended, <15s)", "🔬 Advanced Optimization (Optuna HPO)"],
        index=0,
        horizontal=True,
        help="Fast Mode evaluates candidate baseline models in parallel (<15 seconds). Advanced Mode runs Optuna HPO."
    )
    is_fast_mode = "Fast" in mode_selection

    if st.button("🚀 Train Model Pipeline", type="primary", use_container_width=True):
        selected_target = st.session_state.target_col
        if st.session_state.uploaded_df is not None:
            if selected_target not in st.session_state.uploaded_df.columns:
                st.error(f"Training requires target column '{selected_target}'.")
                st.stop()
            save_uploaded_dataset(st.session_state.uploaded_df)
            train_source = UPLOAD_PATH
        else:
            if selected_target is None and os.path.exists(train_source):
                sample_df = pd.read_csv(train_source)
                selected_target = find_target_col(sample_df)

        allow_id = st.session_state.get("id_target_chk", False)
        if selected_target and is_identifier_column(pd.read_csv(train_source), selected_target) and not allow_id:
            st.error(f"⚠️ Training blocked: Column '{selected_target}' is a unique row identifier. Please select a valid target or check the override box.")
            st.stop()

        status_box = st.status("Initializing Training Pipeline...", expanded=True)
        prog_bar = st.progress(0)

        def streamlit_progress(percent: int, msg: str):
            prog_bar.progress(percent)
            status_box.update(label=msg)

        try:
            result = run_training(
                data_path=train_source,
                target_col=selected_target,
                fast_mode=is_fast_mode,
                allow_id_target=allow_id,
                progress_callback=streamlit_progress,
            )
            st.session_state.train_result = result
            st.session_state.model_ready = True
            status_box.update(label=f"🎉 Training Complete in {result.get('total_time_seconds', 0)}s!", state="complete", expanded=False)
            st.success(f"Training complete in **{result.get('total_time_seconds', 0)}s**! Pipeline saved to `models/unified_pipeline.joblib`.")
        except Exception as exc:
            st.session_state.train_result = None
            status_box.update(label="⚠️ Training Failed", state="error")
            st.error(f"⚠️ Training failed: {exc}")

if st.session_state.train_result:
    result = st.session_state.train_result
    task_type = result.get("task_type", "classification")
    test_m = result.get("best_test_metrics", result.get("best_val_metrics", {}))

    def fmt_pct(val):
        if val is None or not isinstance(val, (int, float, np.number)) or np.isnan(val):
            return "N/A"
        return f"{float(val):.2%}"

    def fmt_num(val):
        if val is None or not isinstance(val, (int, float, np.number)) or np.isnan(val):
            return "N/A"
        return f"{float(val):.4f}"

    warnings = result.get("warnings", [])
    if warnings:
        for w in warnings:
            st.warning(f"⚠️ {w}")

    st.markdown(f"### 📊 Training Results (Holdout Test Set): Best Model — **{result['best_model_name'].replace('_', ' ')}**")
    if result.get("total_time_seconds"):
        st.caption(
            f"⏱️ Runtime: **{result['total_time_seconds']}s** | Baseline Score: **{fmt_num(result.get('baseline_score'))}** | "
            f"CV Score: **{fmt_num(result.get('cv_score'))}**"
        )

    if task_type == "classification":
        m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
        m1.metric("Accuracy", fmt_pct(test_m.get("accuracy")))
        m2.metric("Precision", fmt_pct(test_m.get("precision")))
        m3.metric("Recall", fmt_pct(test_m.get("recall")))
        m4.metric("F1 Score", fmt_num(test_m.get("f1_score")))
        m5.metric("ROC-AUC", fmt_num(test_m.get("roc_auc")))
        m6.metric("PR-AUC", fmt_num(test_m.get("pr_auc")))
        m7.metric("Business Cost", f"${result.get('business_cost', 0.0):,.2f}")

        tab_eval_plots, tab_shap, tab_compare = st.tabs(
            ["📈 Confusion Matrix & Curves", "🔍 SHAP & Feature Importance", "🏆 Model Suite Comparison"]
        )

        with tab_eval_plots:
            col_cm, col_roc = st.columns(2)
            with col_cm:
                if os.path.exists("reports/plots/confusion_matrix.png"):
                    st.image("reports/plots/confusion_matrix.png", caption="Confusion Matrix", use_container_width=True)
                elif os.path.exists("reports/plots/calibration_curve.png"):
                    st.image("reports/plots/calibration_curve.png", caption="Calibration Curve", use_container_width=True)
            with col_roc:
                if os.path.exists("reports/plots/roc_curve.png"):
                    st.image("reports/plots/roc_curve.png", caption="ROC Curve", use_container_width=True)
                elif os.path.exists("reports/plots/pr_curve.png"):
                    st.image("reports/plots/pr_curve.png", caption="Precision-Recall Curve", use_container_width=True)

        with tab_shap:
            if os.path.exists("reports/plots/shap_summary.png"):
                st.image("reports/plots/shap_summary.png", caption="SHAP Summary Feature Importance", use_container_width=True)
            else:
                st.warning("⚠️ SHAP feature importance plot unavailable for this run.")

        with tab_compare:
            if result.get("model_results"):
                comp_df = pd.DataFrame(result["model_results"])
                st.dataframe(comp_df, use_container_width=True)
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("MAE", fmt_num(test_m.get("mae")))
        m2.metric("RMSE", fmt_num(test_m.get("rmse")))
        m3.metric("R² Score", fmt_num(test_m.get("r2_score")))
        m4.metric("MLflow Version", result.get("version", "1"))

        tab_shap, tab_compare = st.tabs(
            ["🔍 SHAP & Feature Importance", "🏆 Model Suite Comparison"]
        )

        with tab_shap:
            if os.path.exists("reports/plots/shap_summary.png"):
                st.image("reports/plots/shap_summary.png", caption="SHAP Summary Feature Importance", use_container_width=True)
            else:
                st.info("SHAP plot was generated during training and logged to MLflow artifacts.")

        with tab_compare:
            if result.get("model_results"):
                comp_df = pd.DataFrame(result["model_results"])
                st.dataframe(comp_df, use_container_width=True)

st.divider()
st.subheader("3. Run Predictions")

model, preprocessor, opt_th, status_msg = load_trained_artifacts()
if model is None or preprocessor is None:
    st.warning(f"⚠️ {status_msg}")
    st.stop()

task_type = getattr(preprocessor, "task_type_", "classification")

predict_tab_single, predict_tab_batch, predict_tab_form = st.tabs(
    ["Pick from Dataset", "Batch Predict", "Manual Input Form"]
)

source_df = st.session_state.uploaded_df
if source_df is None and os.path.exists(UPLOAD_PATH):
    source_df = pd.read_csv(UPLOAD_PATH)
elif source_df is None and os.path.exists("data/raw/telco_churn.csv"):
    source_df = pd.read_csv("data/raw/telco_churn.csv")

target_name = getattr(preprocessor, "target_col_", None) or (st.session_state.target_col if st.session_state.target_col else "Churn")

with predict_tab_single:
    if source_df is None:
        st.warning("Upload a dataset to pick rows from it.")
    else:
        try:
            row_idx = st.number_input(
                "Select dataset row index",
                min_value=0,
                max_value=len(source_df) - 1,
                value=0,
                step=1,
            )
            selected = source_df.iloc[int(row_idx)]
            st.json(selected.drop(labels=[target_name], errors="ignore").to_dict())

            if st.button("Predict Selected Row", key="predict_single"):
                prediction = predict_single_row(selected, model=model, preprocessor=preprocessor, threshold=opt_th)
                actual = selected.get(target_name, "Unknown")

                c1, c2 = st.columns(2)
                if task_type == "classification":
                    c1.metric("Predicted Label", prediction.get("churn_label", prediction.get("churn_prediction")))
                    c2.metric("Probability", f"{prediction.get('churn_probability', 0.0):.2%}")
                else:
                    c1.metric("Predicted Value", f"{prediction.get('predicted_value', 0.0):,.4f}")
                    if actual != "Unknown":
                        c2.metric("Actual Value", f"{actual}")
        except Exception as exc:
            st.error(f"⚠️ Prediction error: {exc}")

with predict_tab_batch:
    if source_df is None:
        st.warning("Upload a dataset for batch predictions.")
    else:
        try:
            sample_size = st.slider("Rows to predict", min_value=5, max_value=min(500, len(source_df)), value=min(50, len(source_df)))
            if st.button("Run Batch Prediction", key="predict_batch"):
                batch_results = predict_customers(source_df.head(sample_size), model=model, preprocessor=preprocessor, threshold=opt_th)
                st.dataframe(batch_results, use_container_width=True)
        except Exception as exc:
            st.error(f"⚠️ Batch prediction error: {exc}")

with predict_tab_form:
    st.markdown("Enter feature values manually:")

    feature_cols = getattr(preprocessor, "feature_cols_", [])

    with st.form("manual_predict_form"):
        form_inputs = {}
        cols = st.columns(3)

        for col_idx, col_name in enumerate(feature_cols):
            with cols[col_idx % 3]:
                is_num = False
                if source_df is not None and col_name in source_df.columns:
                    is_num = pd.api.types.is_numeric_dtype(source_df[col_name])
                else:
                    is_num = col_name in getattr(preprocessor, "num_cols_", [])

                if is_num:
                    default_val = 0.0
                    if source_df is not None and col_name in source_df.columns:
                        try:
                            default_val = float(source_df[col_name].dropna().median())
                        except Exception:
                            default_val = 0.0
                    form_inputs[col_name] = st.number_input(f"{col_name} (numeric)", value=default_val)
                else:
                    options = []
                    if source_df is not None and col_name in source_df.columns:
                        options = [str(x) for x in source_df[col_name].dropna().unique().tolist()]
                    if not options:
                        options = ["Yes", "No"]
                    form_inputs[col_name] = st.selectbox(f"{col_name} (categorical)", options=options)

        submitted = st.form_submit_button("Predict", use_container_width=True)

    if submitted:
        try:
            manual_row = pd.Series(form_inputs)
            prediction = predict_single_row(manual_row, model=model, preprocessor=preprocessor, threshold=opt_th)
            st.json(prediction)
        except Exception as exc:
            st.error(f"⚠️ Manual prediction error: {exc}")

feature_list = getattr(preprocessor, "feature_cols_", [])
st.caption(f"Trained feature columns ({len(feature_list)}): {', '.join(feature_list)}")
