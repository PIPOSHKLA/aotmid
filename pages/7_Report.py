"""Report page."""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import streamlit as st
import numpy as np
import pandas as pd
from aot_stock_network.dashboard.utils import get_data, inject_css

st.set_page_config(page_title="Report", page_icon="📄", layout="wide")
inject_css()

st.title("📄 Research Report")
st.markdown("Structured findings from every module.")

df = get_data()
if df is None or df.empty:
    st.warning("No data available.")
    st.stop()

sections = st.selectbox("Jump to section", [
    "Overview", "Data Summary", "Network Analysis", "Machine Learning", "Conclusion"
])

if sections == "Overview":
    st.header("1. Overview")
    st.markdown("""
    This research investigates how macroeconomic and market factors collectively influence
    the stock price of Airports of Thailand (AOT). Using Social Network Analysis (SNA),
    we model factor interdependencies as a weighted graph. Monthly data (2015–2024) was
    collected from four official Thai sources: SET, MOTS, BOT, and NESDC.

    **Key variables:**
    - AOT closing price (target)
    - Trading volume
    - SET Index
    - Tourist arrivals
    - USD/THB exchange rate
    - Policy rate
    - CPI (headline)
    - GDP
    """)

elif sections == "Data Summary":
    st.header("2. Data Summary")
    st.dataframe(df.describe(), use_container_width=True)
    st.markdown(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
    st.markdown(f"**Date range:** {df.index[0]} to {df.index[-1]}")

elif sections == "Network Analysis":
    st.header("3. Network Analysis")
    st.markdown("""
    The factor network is built using Pearson/Spearman/Mutual Information as edge weights.
    Five centrality metrics are computed: degree, betweenness, closeness, eigenvector,
    and PageRank. Communities are detected using the Louvain algorithm.

    **Interpretation:**
    - Betweenness centrality identifies bridge variables (e.g., USD/THB)
    - Communities separate market factors from macroeconomic indicators
    - Degree centrality shows the most connected variables
    """)

    from aot_stock_network.network_analysis import NetworkBuilder, NetworkAnalyzer
    builder = NetworkBuilder(df)
    try:
        G = builder.build_graph(method="pearson", threshold=0.3)
        analyzer = NetworkAnalyzer(G)
        m = analyzer.compute_all()
        st.metric("Nodes", G.number_of_nodes())
        st.metric("Edges", G.number_of_edges())
        st.metric("Density", f"{m.network_density:.3f}")
        st.metric("Communities", m.n_communities)
        st.dataframe(m.to_dataframe(), use_container_width=True)
    except Exception as e:
        st.warning(f"Network analysis error: {e}")

elif sections == "Machine Learning":
    st.header("4. Machine Learning")
    st.markdown("""
    Eight model families are compared: Linear Regression, Random Forest, XGBoost,
    LightGBM, CatBoost, ARIMA, Prophet, and LSTM. The best model is selected by
    validation RMSE. SHAP analysis identifies the most influential features.

    **Evaluation metrics:** RMSE, MAE, MAPE, R²
    """)

elif sections == "Conclusion":
    st.header("5. Conclusion")
    st.markdown("""
    This study demonstrates that a network-analytic framework can reveal structural
    properties of financial factor interdependencies that conventional analysis cannot.

    **Key findings:**
    - Louvain community structure separates market from macroeconomic variables
    - USD/THB exchange rate is the primary bridge between domains
    - Random Forest provides the most accurate and interpretable predictions
    - Tourist arrivals and foreign exchange are the dominant drivers of AOT stock price

    **Future work:**
    - Incorporate higher-frequency data (daily, weekly)
    - Add external factors (oil prices, geopolitical risk)
    - Real-time dashboard deployment
    """)
