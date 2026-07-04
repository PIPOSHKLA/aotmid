"""Correlation analysis page."""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from aot_stock_network.visualization import plot_correlation_heatmap, plot_feature_correlation_matrix, plot_scatter
from aot_stock_network.dashboard.utils import get_data, inject_css, download_figure, download_button

st.set_page_config(page_title="Correlation", page_icon="🔗", layout="wide")
inject_css()

st.title("🔗 Correlation Analysis")
st.markdown("Pairwise relationships between factors.")

df = get_data()
if df is None or df.empty:
    st.warning("No data available.")
    st.stop()

numeric = df.select_dtypes(include=[np.number])
if numeric.empty:
    st.warning("No numeric columns found.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["Heatmap", "Scatter Matrix", "Pairwise Scatter"])

with tab1:
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_correlation_heatmap(numeric, ax=ax)
    st.pyplot(fig)
    download_figure(fig, "correlation_heatmap.png")

with tab2:
    fig = plot_feature_correlation_matrix(numeric)
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    cols = st.multiselect("Select columns", numeric.columns.tolist(), default=numeric.columns[:4].tolist())
    if len(cols) >= 2:
        x = st.selectbox("X", cols, index=0, key="cx")
        y = st.selectbox("Y", cols, index=min(1, len(cols) - 1), key="cy")
        fig, ax = plt.subplots(figsize=(8, 6))
        plot_scatter(numeric, x_col=x, y_col=y, ax=ax)
        st.pyplot(fig)
        download_figure(fig, f"scatter_{x}_vs_{y}.png")
