"""EDA page: distribution, time series, trends, outliers, decomposition."""

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from aot_stock_network.dashboard.utils import (
    download_figure,
    get_data,
    inject_css,
)
from aot_stock_network.visualization import (
    EDAVisualizer,
    plot_decomposition,
    plot_distribution,
    plot_monthly_trend,
    plot_outlier_box,
    plot_outlier_timeseries,
    plot_time_series,
    plot_yearly_trend,
)


def show() -> None:
    inject_css(st.session_state.get("dark_mode", False))
    df = get_data()

    st.title("Exploratory Data Analysis")
    st.markdown("Distribution analysis, time series, seasonality, and outlier detection.")

    target = "aot_close" if "aot_close" in df.columns else df.columns[0]
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📈 Distribution",
            "📉 Time Series",
            "📅 Seasonality",
            "🔍 Outliers",
        ]
    )

    # ── Tab 1: Distribution ───────────────────────────────
    with tab1:
        col = st.selectbox(
            "Variable",
            numeric_cols,
            index=numeric_cols.index(target) if target in numeric_cols else 0,
            key="eda_dist",
        )
        fig, ax = plot_distribution(df, col)
        st.pyplot(fig, use_container_width=True)
        download_figure(fig, f"distribution_{col}.svg")
        plt.close(fig)

        st.markdown("#### Distribution Grid")
        with st.spinner("Generating distribution grid..."):
            fig_g, _ = EDAVisualizer(df).plot_distribution_grid()
            st.pyplot(fig_g, use_container_width=True)
            download_figure(fig_g, "distribution_grid.svg")
            plt.close(fig_g)

    # ── Tab 2: Time Series ────────────────────────────────
    with tab2:
        col = st.selectbox("Variable", numeric_cols, key="eda_ts")
        highlight = st.checkbox("Highlight COVID period (2020-03 to 2021-12)", value=True)
        periods = [("2020-03", "2021-12")] if highlight else None
        fig, ax = plot_time_series(df, col, highlight_periods=periods)
        st.pyplot(fig, use_container_width=True)
        download_figure(fig, f"timeseries_{col}.svg")
        plt.close(fig)

        st.markdown("#### Decomposition")
        decomp_col = st.selectbox("Variable for decomposition", numeric_cols, key="eda_decomp")
        if st.button("Run Decomposition", type="primary"):
            with st.spinner("Computing seasonal decomposition..."):
                result = plot_decomposition(df, decomp_col)
                if result and isinstance(result[0], plt.Figure):
                    fig_d, _ = result
                    st.pyplot(fig_d, use_container_width=True)
                    download_figure(fig_d, f"decomposition_{decomp_col}.svg")
                    plt.close(fig_d)
                else:
                    st.warning("Insufficient data for decomposition (need ≥24 obs).")

    # ── Tab 3: Seasonality ────────────────────────────────
    with tab3:
        col = st.selectbox("Variable", numeric_cols, key="eda_seas")
        c1, c2 = st.columns(2)
        with c1:
            fig_m, _ = plot_monthly_trend(df, col)
            st.pyplot(fig_m, use_container_width=True)
            download_figure(fig_m, f"monthly_trend_{col}.svg")
            plt.close(fig_m)
        with c2:
            fig_y, _ = plot_yearly_trend(df, col)
            st.pyplot(fig_y, use_container_width=True)
            download_figure(fig_y, f"yearly_trend_{col}.svg")
            plt.close(fig_y)

    # ── Tab 4: Outliers ───────────────────────────────────
    with tab4:
        col = st.selectbox("Variable", numeric_cols, key="eda_out")
        c1, c2 = st.columns(2)
        with c1:
            fig_b, _ = plot_outlier_box(df)
            st.pyplot(fig_b, use_container_width=True)
            download_figure(fig_b, "outlier_box.svg")
            plt.close(fig_b)
        with c2:
            fig_ot, _ = plot_outlier_timeseries(df, col)
            st.pyplot(fig_ot, use_container_width=True)
            download_figure(fig_ot, f"outlier_timeseries_{col}.svg")
            plt.close(fig_ot)

        with st.expander("📊 Outlier Summary"):
            data = df[col].dropna()
            q1, q3 = data.quantile(0.25), data.quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            n_out = ((data < lower) | (data > upper)).sum()
            st.write(f"**{col}** — IQR method (1.5×)")
            st.write(f"- Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f}")
            st.write(f"- Fences: [{lower:.2f}, {upper:.2f}]")
            st.write(f"- Outliers: {n_out} / {len(data)} ({100 * n_out / len(data):.1f}%)")
