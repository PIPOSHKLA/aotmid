"""Correlation page: heatmaps, scatter matrix, pairwise exploration."""

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from aot_stock_network.dashboard.utils import (
    download_button,
    download_figure,
    get_data,
    inject_css,
)
from aot_stock_network.visualization import (
    plot_correlation_heatmap,
    plot_feature_correlation_matrix,
    plot_scatter,
)


def show() -> None:
    inject_css(st.session_state.get("dark_mode", False))
    df = get_data()

    st.title("Correlation Analysis")
    st.markdown("Explore pairwise relationships between variables.")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    target = "aot_close" if "aot_close" in df.columns else numeric_cols[0]

    tab1, tab2, tab3 = st.tabs(
        [
            "🔥 Heatmap",
            "🎯 Target Correlation",
            "🔬 Pairwise Scatter",
        ]
    )

    # ── Tab 1: Heatmap ────────────────────────────────────
    with tab1:
        method = st.radio(
            "Correlation method", ["pearson", "spearman"], horizontal=True, key="corr_method"
        )
        with st.spinner(f"Computing {method} correlation..."):
            corr_df = df[numeric_cols].corr(method=method)
        fig, _ = plot_correlation_heatmap(
            df[numeric_cols],
            method=method,
            title=f"{method.title()} Correlation Matrix",
        )
        st.pyplot(fig, use_container_width=True)
        download_figure(fig, f"correlation_heatmap_{method}.svg")
        plt.close(fig)

        with st.expander("📋 Correlation Table"):
            styled = corr_df.style.background_gradient(cmap="RdBu_r", vmin=-1, vmax=1)
            st.dataframe(styled, use_container_width=True)
            download_button(
                corr_df.reset_index(),
                f"correlation_matrix_{method}.csv",
            )

    # ── Tab 2: Target Correlation ─────────────────────────
    with tab2:
        col = st.selectbox(
            "Target variable", numeric_cols, index=numeric_cols.index(target), key="corr_target"
        )
        fig, _ = plot_feature_correlation_matrix(df, target_col=col)
        st.pyplot(fig, use_container_width=True)
        download_figure(fig, f"target_correlation_{col}.svg")
        plt.close(fig)

        # Interactive plotly bar chart of correlations
        corr_series = df[numeric_cols].corrwith(df[col]).drop(col).sort_values()
        colors = ["#C44E52" if v < 0 else "#4C72B0" for v in corr_series.values]
        fig_bar = go.Figure(
            go.Bar(
                x=corr_series.values,
                y=corr_series.index,
                orientation="h",
                marker_color=colors,
                text=[f"{v:.3f}" for v in corr_series.values],
                textposition="outside",
            )
        )
        fig_bar.update_layout(
            title=f"Correlation with {col}",
            xaxis_title="Correlation",
            template="plotly_white",
            height=max(300, len(corr_series) * 35),
            margin=dict(l=20, r=60, t=40, b=20),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Tab 3: Pairwise Scatter ───────────────────────────
    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            x_col = st.selectbox("X-axis", numeric_cols, index=0, key="corr_x")
        with c2:
            y_col = st.selectbox(
                "Y-axis", numeric_cols, index=min(1, len(numeric_cols) - 1), key="corr_y"
            )
        hue = st.selectbox("Color by", ["None"] + numeric_cols, key="corr_hue")
        add_reg = st.checkbox("Show regression line", value=True)

        fig, _ = plot_scatter(
            df,
            x_col,
            y_col,
            hue_col=None if hue == "None" else hue,
            add_regression=add_reg,
            title=f"{x_col} vs {y_col}",
        )
        st.pyplot(fig, use_container_width=True)
        download_figure(fig, f"scatter_{x_col}_vs_{y_col}.svg")
        plt.close(fig)

        st.markdown("#### Interactive Scatter Matrix")
        matrix_cols = st.multiselect(
            "Columns for matrix",
            numeric_cols,
            default=numeric_cols[: min(4, len(numeric_cols))],
            key="corr_matrix_cols",
        )
        if len(matrix_cols) >= 2:
            df_sub = df[matrix_cols].dropna()
            fig_mat = px.scatter_matrix(
                df_sub,
                dimensions=matrix_cols,
                title="Scatter Matrix",
                opacity=0.6,
            )
            fig_mat.update_traces(marker=dict(size=4, line=dict(width=0.5, color="white")))
            fig_mat.update_layout(
                height=800,
                template="plotly_white",
                margin=dict(l=40, r=40, t=60, b=40),
            )
            st.plotly_chart(fig_mat, use_container_width=True)
        else:
            st.info("Select at least 2 columns for the scatter matrix.")
