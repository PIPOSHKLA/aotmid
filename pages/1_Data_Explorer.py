"""Data Explorer page."""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from aot_stock_network.dashboard.utils import get_data, inject_css, download_button

st.set_page_config(page_title="Data Explorer", page_icon="📊", layout="wide")
inject_css()

st.title("📊 Data Explorer")
st.markdown("Browse, filter, and explore the combined monthly dataset.")

df = get_data()
if df is None or df.empty:
    st.warning("No data available. Generate sample data from Home page first.")
    st.stop()

st.metric("Rows", df.shape[0], help="Number of monthly observations")
st.metric("Columns", df.shape[1], help="Number of variables + features")

tab1, tab2, tab3 = st.tabs(["Table", "Summary", "Column Filter"])

with tab1:
    col, val = st.columns([1, 3])
    with col:
        n = st.slider("Rows to show", 5, min(100, len(df)), 20)
    with val:
        st.dataframe(df.head(n), use_container_width=True)
    csv = df.head(1000).to_csv(index=True).encode()
    st.download_button("⬇ Download CSV", csv, "aot_data.csv", "text/csv")

with tab2:
    st.dataframe(df.describe(), use_container_width=True)

with tab3:
    cols = st.multiselect("Select columns", df.columns.tolist(), default=df.columns[:4].tolist())
    if cols:
        st.dataframe(df[cols].head(50), use_container_width=True)
        if len(cols) >= 2:
            x = st.selectbox("X axis", cols, index=0)
            y = st.selectbox("Y axis", cols, index=min(1, len(cols) - 1))
            fig = px.scatter(df, x=x, y=y, title=f"{y} vs {x}", trendline="ols")
            st.plotly_chart(fig, use_container_width=True)
