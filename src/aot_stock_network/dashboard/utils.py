"""Shared dashboard utilities: CSS, dark mode, data loading, sidebar."""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st

from aot_stock_network.feature_engineering import FeatureEngineer

logger = logging.getLogger("dashboard")

# ── paths ─────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# ── page config ───────────────────────────────────────────

PAGE_ICONS = {
    "Home": "🏠",
    "Data Explorer": "📊",
    "EDA": "📈",
    "Correlation": "🔗",
    "Social Network Analysis": "🕸️",
    "Machine Learning": "🤖",
    "Forecast": "🔮",
    "Report": "📄",
    "Download Center": "⬇️",
}

SIDEBAR_SECTIONS = {
    "Overview": ["Home"],
    "Analysis": ["Data Explorer", "EDA", "Correlation"],
    "Network": ["Social Network Analysis"],
    "Prediction": ["Machine Learning", "Forecast"],
    "Output": ["Report", "Download Center"],
}


# ── CSS ───────────────────────────────────────────────────


def inject_css(dark_mode: bool = False) -> None:
    """Inject custom CSS for dark/light mode."""
    bg = "#0E1117" if dark_mode else "#FFFFFF"
    bg2 = "#1A1D24" if dark_mode else "#F8F9FA"
    text = "#E8E8E8" if dark_mode else "#212121"
    muted = "#9E9E9E" if dark_mode else "#757575"
    border = "#333333" if dark_mode else "#E0E0E0"
    card = "#1E2029" if dark_mode else "#FFFFFF"
    accent = "#4C72B0"

    st.markdown(
        f"""
    <style>
        .stApp .main .block-container {{ padding: 1.5rem 2rem; }}
        h1, h2, h3 {{ color: {text} !important; }}
        .metric-card {{
            background: {card};
            border: 1px solid {border};
            border-radius: 12px;
            padding: 1.2rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
            text-align: center;
        }}
        .metric-card .label {{ font-size: 0.8rem; color: {muted}; margin-bottom: 0.3rem; }}
        .metric-card .value {{ font-size: 1.6rem; font-weight: 700; color: {accent}; }}
        .metric-card .delta {{ font-size: 0.75rem; color: {muted}; }}
        .info-box {{
            background: {card};
            border: 1px solid {border};
            border-radius: 8px;
            padding: 1rem;
            margin: 0.5rem 0;
        }}
        .info-box h4 {{ margin: 0 0 0.4rem; color: {text}; }}
        .info-box p {{ margin: 0; font-size: 0.9rem; color: {muted}; }}
        .section-title {{
            font-size: 1.1rem; font-weight: 600;
            color: {accent}; margin: 1.5rem 0 0.8rem;
            border-bottom: 2px solid {accent}; padding-bottom: 0.3rem;
        }}
        .stDataFrame, .stTable {{ font-size: 0.85rem; }}
        div[data-testid="stSidebar"] .stRadio label {{ font-size: 0.9rem; }}
        div[data-testid="stSidebar"] hr {{ margin: 0.5rem 0; }}
        @media (max-width: 768px) {{
            .stApp .main .block-container {{ padding: 1rem; }}
            .metric-card .value {{ font-size: 1.2rem; }}
        }}
    </style>
    """,
        unsafe_allow_html=True,
    )


# ── data loading ──────────────────────────────────────────


@st.cache_data(ttl=300)
def load_aligned_data() -> Optional[pd.DataFrame]:
    """Load the aligned monthly dataset."""
    paths = [
        PROCESSED_DIR / "aligned_monthly.csv",
        PROCESSED_DIR / "feature_dataset.csv",
        ROOT / "data" / "processed" / "aligned_monthly.csv",
        ROOT / "data" / "processed" / "feature_dataset.csv",
    ]
    for p in paths:
        if p.exists():
            df = pd.read_csv(p, index_col=0, parse_dates=True)
            logger.info("Loaded %s: %d rows x %d cols", p.name, *df.shape)
            return df
    return None


@st.cache_data(ttl=300)
def load_feature_dataset() -> Optional[pd.DataFrame]:
    """Load the feature-engineered dataset."""
    paths = [
        PROCESSED_DIR / "feature_dataset.csv",
        ROOT / "data" / "processed" / "feature_dataset.csv",
    ]
    for p in paths:
        if p.exists():
            df = pd.read_csv(p, index_col=0, parse_dates=True)
            logger.info("Loaded %s: %d rows x %d cols", p.name, *df.shape)
            return df
    # Fallback: generate from aligned
    aligned = load_aligned_data()
    if aligned is not None:
        eng = FeatureEngineer(aligned)
        return eng.build_all_features()
    return None


@st.cache_data(ttl=300)
def load_or_generate_sample_data() -> pd.DataFrame:
    """Generate sample data if no real data exists (for demo purposes)."""
    rng = np.random.default_rng(42)
    dates = pd.period_range("2015-01", "2024-12", freq="M")
    n = len(dates)
    df = pd.DataFrame(
        {
            "aot_close": 50 + np.cumsum(rng.normal(0.3, 1.0, n)),
            "aot_volume": 2e7 + rng.integers(-5e6, 5e6, n),
            "set_close": 1500 + np.cumsum(rng.normal(0, 15, n)),
            "tourists_total_arrivals": 2e6 + rng.integers(-3e5, 3e5, n),
            "tourism_revenue": 5e9 + rng.integers(-1e9, 1e9, n),
            "fx_usdthb_rate": 32 + rng.normal(0, 0.3, n),
            "policy_policy_rate": 1.5 + rng.normal(0, 0.1, n),
            "cpi_cpi_headline": 100 + np.cumsum(rng.normal(0.15, 0.1, n)),
            "gdp_gdp": 500 + np.cumsum(rng.normal(2, 0.5, n)),
        },
        index=dates,
    )
    logger.info("Generated sample data: %d rows x %d cols", *df.shape)
    return df


# ── session state helpers ─────────────────────────────────


def init_session_state() -> None:
    """Ensure all session state keys exist."""
    defaults: Dict[str, Any] = {
        "dark_mode": False,
        "df_raw": None,
        "df_features": None,
        "network_graph": None,
        "network_metrics": None,
        "network_viz": None,
        "ml_results": None,
        "ml_pipeline": None,
        "forecast_horizon": 12,
        "selected_model": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def get_data() -> pd.DataFrame:
    """Get the best available DataFrame."""
    if st.session_state.df_features is not None:
        return st.session_state.df_features
    df = load_feature_dataset()
    if df is not None:
        st.session_state.df_features = df
        return df
    df = load_or_generate_sample_data()
    st.session_state.df_features = df
    return df


# ── sidebar ───────────────────────────────────────────────


def render_sidebar() -> None:
    """Render the sidebar with navigation, data info, and controls."""
    with st.sidebar:
        st.markdown("### AOT Stock Network")
        st.markdown(
            "Social Network Analysis of Factors Influencing Airports of Thailand Stock Price"
        )
        st.divider()

        # Data info
        df = st.session_state.get("df_features")
        if df is not None:
            st.caption(f"Data: {len(df)} months × {len(df.columns)} features")
            st.caption(f"Range: {df.index[0]} – {df.index[-1]}")
        else:
            st.caption("No data loaded")

        st.divider()

        # Dark mode toggle
        dark = st.toggle("Dark Mode", value=st.session_state.dark_mode)
        if dark != st.session_state.dark_mode:
            st.session_state.dark_mode = dark
            st.rerun()

        st.divider()
        st.caption("v1.0 | AOT Stock Network Project")


# ── helper components ─────────────────────────────────────


def metric_card(label: str, value: str, delta: str = "") -> str:
    """Return HTML for a metric card."""
    return f"""
    <div class="metric-card">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
        {f'<div class="delta">{delta}</div>' if delta else ""}
    </div>
    """


def download_button(df: pd.DataFrame, filename: str, label: str = "Download CSV") -> None:
    """Render a download button for a DataFrame."""
    buf = io.BytesIO()
    df.to_csv(buf, encoding="utf-8-sig")
    buf.seek(0)
    st.download_button(
        label=label,
        data=buf,
        file_name=filename,
        mime="text/csv",
        use_container_width=True,
    )


def download_figure(fig, filename: str, label: str = "Download SVG") -> None:
    """Render a download button for a matplotlib figure."""
    buf = io.BytesIO()
    fig.savefig(buf, format="svg", dpi=300, bbox_inches="tight")
    buf.seek(0)
    st.download_button(
        label=label,
        data=buf,
        file_name=filename,
        mime="image/svg+xml",
        use_container_width=True,
    )
