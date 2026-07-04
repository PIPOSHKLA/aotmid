"""Home page: project overview, key metrics, quick navigation."""

import numpy as np
import streamlit as st

from aot_stock_network.dashboard.utils import (
    get_data,
    init_session_state,
    inject_css,
    metric_card,
)


def show() -> None:
    inject_css(st.session_state.get("dark_mode", False))
    init_session_state()
    df = get_data()

    st.title("Airports of Thailand (AOT) — Stock Price Influence Analysis")
    st.markdown(
        "A graduate-level DDAS research project applying **Social Network "
        "Analysis (SNA)** and **Machine Learning** to quantify how macroeconomic "
        "and market factors influence AOT stock price."
    )

    # ── Key metrics row ───────────────────────────────────
    cols = st.columns(4)
    with cols[0]:
        st.markdown(metric_card("Data Span", f"{len(df)} months"), unsafe_allow_html=True)
    with cols[1]:
        n_vars = len(df.select_dtypes(include=[np.number]).columns)
        st.markdown(metric_card("Variables", str(n_vars)), unsafe_allow_html=True)
    with cols[2]:
        if "aot_close" in df.columns:
            price = df["aot_close"].iloc[-1]
            change = df["aot_close"].pct_change().iloc[-1] * 100
            st.markdown(
                metric_card("AOT Close", f"{price:.2f} THB", f"{change:+.2f}% (last month)"),
                unsafe_allow_html=True,
            )
    with cols[3]:
        st.markdown(metric_card("Analysis Modules", "8 ML Models"), unsafe_allow_html=True)

    st.divider()

    # ── Overview sections ─────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("#### 🎯 Research Objective")
        st.markdown(
            "Identify and quantify the relative influence of key macroeconomic "
            "and market factors on AOT's stock price using social network "
            "analysis and machine learning."
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("#### 🔬 Methodology")
        st.markdown("""
        - **Data Collection**: Official Thai sources (SET, MOTS, BOT, data.go.th)
        - **Feature Engineering**: 11+ predictive feature types
        - **Social Network Graph**: Pearson/Spearman/MI edges, centrality metrics
        - **Machine Learning**: 8 model families with hyperparameter tuning
        - **SHAP Analysis**: Model-agnostic feature importance
        """)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("#### 📊 Data Sources")
        st.markdown("""
        | Source | Data |
        |--------|------|
        | SET (OAQ API) | AOT stock price, volume, SET index |
        | MOTS (Ministry of Tourism) | Tourist arrivals & revenue |
        | Bank of Thailand | USD/THB, policy rate, CPI |
        | NESDC (data.go.th) | GDP (quarterly) |
        """)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("#### 🧭 Dashboard Navigation")
        st.markdown("""
        Use the **sidebar** to explore:
        - **Data Explorer**: Browse and filter raw data
        - **EDA**: Distribution, trends, seasonality plots
        - **Correlation**: Pairwise relationships
        - **Social Network**: Interactive graph with metrics
        - **Machine Learning**: Model comparison & SHAP
        - **Forecast**: Future price predictions
        - **Report**: Generated research findings
        - **Download Center**: Export data & figures
        """)
        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    # ── Data preview ──────────────────────────────────────
    with st.expander("📋 Data Preview", expanded=False):
        tab1, tab2 = st.tabs(["Raw Data", "Summary Statistics"])
        with tab1:
            st.dataframe(df.head(10), use_container_width=True)
        with tab2:
            st.dataframe(df.describe(), use_container_width=True)

    st.caption("2026 — AOT Stock Network Research Project")
