"""
Home.py — AOT Stock Network Analysis Dashboard
================================================

Entry point for the Streamlit multipage application.

Usage:
    streamlit run Home.py

This file uses Streamlit's navigation API (st.Page + st.navigation) to
define the page structure and sidebar. Each page is backed by a show()
function in the dashboard package.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure src/ is on the path
_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import streamlit as st
import matplotlib
matplotlib.use("Agg")

from aot_stock_network.dashboard import (
    show_home,
    show_data_explorer,
    show_eda,
    show_correlation,
    show_social_network,
    show_ml,
    show_forecast,
    show_report,
    show_download,
)

# ── Page configuration ─────────────────────────────────────
st.set_page_config(
    page_title="AOT Stock Network",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Navigation ─────────────────────────────────────────────
pages = {
    "Overview": [
        st.Page(show_home, title="Home", icon="🏠"),
    ],
    "Analysis": [
        st.Page(show_data_explorer, title="Data Explorer", icon="📊"),
        st.Page(show_eda, title="EDA", icon="📈"),
        st.Page(show_correlation, title="Correlation", icon="🔗"),
    ],
    "Network": [
        st.Page(show_social_network, title="Social Network Analysis", icon="🕸️"),
    ],
    "Prediction": [
        st.Page(show_ml, title="Machine Learning", icon="🤖"),
        st.Page(show_forecast, title="Forecast", icon="🔮"),
    ],
    "Output": [
        st.Page(show_report, title="Report", icon="📄"),
        st.Page(show_download, title="Download Center", icon="⬇️"),
    ],
}

# ── Render navigation + current page ──────────────────────
pg = st.navigation(pages)
pg.run()
