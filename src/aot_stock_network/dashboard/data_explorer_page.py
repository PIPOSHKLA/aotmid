"""Data Explorer: browse, filter, search, and summarise the dataset."""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from aot_stock_network.dashboard.utils import (
    download_button,
    get_data,
    inject_css,
)


def show() -> None:
    inject_css(st.session_state.get("dark_mode", False))
    df = get_data()

    st.title("Data Explorer")
    st.markdown("Browse, filter, and explore the feature dataset.")

    # ── Filters ───────────────────────────────────────────
    with st.expander("🔍 Filters & Options", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            all_cols = df.columns.tolist()
            selected_cols = st.multiselect(
                "Columns to display",
                all_cols,
                default=all_cols[: min(8, len(all_cols))],
            )
        with col2:
            if isinstance(df.index, pd.PeriodIndex):
                dates = [str(d) for d in df.index]
            else:
                dates = df.index.astype(str).tolist()
            date_range = st.select_slider(
                "Date range",
                options=dates,
                value=(dates[0], dates[-1]),
            )
        with col3:
            search = st.text_input("Search (any column)", placeholder="e.g., 2020")

    # Apply date filter
    if date_range:
        start, end = date_range
        mask = df.index.astype(str).between(start, end)
        filtered = df.loc[mask]
    else:
        filtered = df

    # Apply search
    if search:
        search_lower = search.lower()
        mask = filtered.astype(str).apply(
            lambda row: row.str.contains(search_lower, case=False).any(),
            axis=1,
        )
        filtered = filtered.loc[mask]

    # Apply column selection
    cols_to_show = [c for c in selected_cols if c in filtered.columns]
    if not cols_to_show:
        cols_to_show = filtered.columns[:6].tolist()
    display_df = filtered[cols_to_show]

    # ── Stats row ─────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", len(display_df))
    c2.metric("Columns", len(cols_to_show))
    c3.metric("Missing Values", int(display_df.isna().sum().sum()))
    c4.metric("Numeric Features", len(display_df.select_dtypes(include=[np.number]).columns))

    # ── Data table ────────────────────────────────────────
    st.dataframe(display_df, use_container_width=True, height=400)

    # ── Quick time series preview ─────────────────────────
    st.markdown("#### Quick Time Series Preview")
    col = st.selectbox("Select column to plot", numeric_cols, index=0)
    fig = px.line(
        x=df.index.astype(str),
        y=df[col].values,
        labels={"x": "Date", "y": col},
        title=f"{col} Over Time",
    )
    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=40, r=20, t=40, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Download ──────────────────────────────────────────
    with st.expander("⬇️ Export"):
        download_button(display_df, "aot_data_explorer.csv", "Download filtered data as CSV")
