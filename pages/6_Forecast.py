"""Forecast page."""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
from aot_stock_network.prediction import PredictionPipeline
from aot_stock_network.feature_engineering import FeatureEngineer
from aot_stock_network.dashboard.utils import get_data, inject_css, download_figure, download_button

st.set_page_config(page_title="Forecast", page_icon="🔮", layout="wide")
inject_css()

st.title("🔮 Forecast")
st.markdown("Future AOT closing price predictions.")

df = get_data()
if df is None or df.empty:
    st.warning("No data available.")
    st.stop()

horizon = st.slider("Forecast horizon (months)", 1, 24, 12, 1)

if st.button("🔮 Generate Forecast", type="primary"):
    fe = FeatureEngineer(df=df)
    features = fe.build_all_features()

    pipe = PredictionPipeline(features, target_col="aot_close", test_months=12, val_months=6)
    with st.spinner("Training best model (RF)..."):
        results = pipe.run(models=["rf", "xgb", "lr"], tune=False, calc_shap=False)

    best = results.best()
    if best is None:
        st.error("No model trained successfully.")
        st.stop()

    st.success(f"Best model: {best.label} (val RMSE={best.val_metrics.rmse:.4f})")

    st.subheader("Model Predictions (Test Set)")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(best.y_test_true, label="Actual", marker="o", linestyle="-")
    ax.plot(best.y_test_pred, label="Predicted", marker="s", linestyle="--")
    ax.set_title(f"{best.label} — Actual vs Predicted")
    ax.set_xlabel("Test sample")
    ax.set_ylabel("AOT Close (THB)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    download_figure(fig, "forecast_test.png")

    st.subheader("Metrics Table")
    st.dataframe(results.metrics_dataframe(), use_container_width=True)
