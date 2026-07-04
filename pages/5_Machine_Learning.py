"""Machine Learning page."""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
from aot_stock_network.prediction import PredictionPipeline
from aot_stock_network.feature_engineering import FeatureEngineer
from aot_stock_network.dashboard.utils import get_data, inject_css, download_figure, download_button

st.set_page_config(page_title="Machine Learning", page_icon="🤖", layout="wide")
inject_css()

st.title("🤖 Machine Learning")
st.markdown("Train models, compare performance, and interpret predictions.")

df = get_data()
if df is None or df.empty:
    st.warning("No data available.")
    st.stop()

models = st.multiselect(
    "Select models",
    ["lr", "rf", "xgb", "lgb", "cat", "arima", "prophet"],
    default=["lr", "rf", "xgb"],
)
tune = st.checkbox("Hyperparameter tuning (slow)", False)
shap = st.checkbox("SHAP analysis", False)

if st.button("🚀 Run Training", type="primary"):
    if not models:
        st.warning("Select at least one model.")
        st.stop()

    fe = FeatureEngineer(df=df)
    features = fe.build_all_features()

    pipe = PredictionPipeline(features, target_col="aot_close", test_months=12, val_months=6)
    with st.spinner("Training models..."):
        results = pipe.run(models=models, tune=tune, calc_shap=shap)

    st.success(f"Trained {len(results.results)} model(s)")

    col1, col2 = st.columns(2)
    with col1:
        fig, _ = results.plot_model_comparison()
        st.pyplot(fig)
        download_figure(fig, "model_comparison.png")
    with col2:
        fig, _ = results.plot_predictions()
        st.pyplot(fig)
        download_figure(fig, "predictions.png")

    col3, col4 = st.columns(2)
    with col3:
        fig, _ = results.plot_residuals()
        st.pyplot(fig)
        download_figure(fig, "residuals.png")
    with col4:
        fig = results.plot_feature_importance()
        if fig:
            st.pyplot(fig)
            download_figure(fig, "feature_importance.png")
        else:
            st.info("Feature importance not available for this model.")

    if shap:
        fig = results.plot_shap()
        if fig:
            st.pyplot(fig)
            download_figure(fig, "shap_summary.png")

    st.subheader("Metrics Summary")
    st.dataframe(results.metrics_dataframe(), use_container_width=True)
