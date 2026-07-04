"""EDA page."""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from aot_stock_network.visualization import EDAVisualizer
from aot_stock_network.dashboard.utils import get_data, inject_css, download_figure

st.set_page_config(page_title="EDA", page_icon="📈", layout="wide")
inject_css()

st.title("📈 Exploratory Data Analysis")
st.markdown("Time series, distributions, trends, and decomposition.")

df = get_data()
if df is None or df.empty:
    st.warning("No data available.")
    st.stop()

viz = EDAVisualizer(df=df)

plot_type = st.selectbox("Plot type", [
    "Time Series", "Distribution", "Monthly Trend", "Yearly Trend",
    "Outlier Box", "Outlier Time Series", "Decomposition",
])

if plot_type == "Time Series":
    col = st.selectbox("Column", [c for c in df.columns if df[c].dtype.kind in "ifc"])
    fig = viz.plot_time_series(column=col)
    st.pyplot(fig)
    download_figure(fig, f"timeseries_{col}.png")

elif plot_type == "Distribution":
    col = st.selectbox("Column", [c for c in df.columns if df[c].dtype.kind in "ifc"])
    fig = viz.plot_distribution(column=col)
    st.pyplot(fig)
    download_figure(fig, f"distribution_{col}.png")

elif plot_type == "Monthly Trend":
    col = st.selectbox("Column", [c for c in df.columns if df[c].dtype.kind in "ifc"])
    fig = viz.plot_monthly_trend(column=col)
    st.pyplot(fig)
    download_figure(fig, f"monthly_{col}.png")

elif plot_type == "Yearly Trend":
    col = st.selectbox("Column", [c for c in df.columns if df[c].dtype.kind in "ifc"])
    fig = viz.plot_yearly_trend(column=col)
    st.pyplot(fig)
    download_figure(fig, f"yearly_{col}.png")

elif plot_type == "Outlier Box":
    fig = viz.plot_outlier_box()
    st.pyplot(fig)
    download_figure(fig, "outlier_box.png")

elif plot_type == "Outlier Time Series":
    col = st.selectbox("Column", [c for c in df.columns if df[c].dtype.kind in "ifc"])
    fig = viz.plot_outlier_timeseries(column=col)
    st.pyplot(fig)
    download_figure(fig, f"outlier_ts_{col}.png")

elif plot_type == "Decomposition":
    fig = viz.plot_decomposition()
    st.pyplot(fig)
    download_figure(fig, "decomposition.png")
