"""Report page: structured research report with findings from every module."""

import numpy as np
import pandas as pd
import streamlit as st

from aot_stock_network.dashboard.utils import (
    get_data,
    inject_css,
)


def show() -> None:
    inject_css(st.session_state.get("dark_mode", False))
    df = get_data()

    st.title("Research Report")
    st.markdown("Structured findings from the AOT Stock Network Analysis project.")

    sections = [
        ("1. Research Overview", _section_overview),
        ("2. Data Collection & Sources", _section_data),
        ("3. Exploratory Data Analysis", _section_eda),
        ("4. Correlation Structure", _section_correlation),
        ("5. Social Network Analysis", _section_network),
        ("6. Machine Learning Results", _section_ml),
        ("7. Forecast", _section_forecast),
        ("8. Conclusions & Recommendations", _section_conclusions),
    ]

    for title, func in sections:
        with st.expander(title, expanded=(title == "1. Research Overview")):
            func(df)

    # ── Export report ─────────────────────────────────────
    st.divider()
    st.markdown("### Export Report")
    c1, c2 = st.columns(2)
    with c1:
        report_text = "\n\n".join(
            f"# {title}\n\n" + _get_section_text(title, df) for title, _ in sections
        )
        st.download_button(
            "📄 Download as Markdown",
            data=report_text,
            file_name="AOT_Stock_Network_Report.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with c2:
        st.download_button(
            "📋 Download Summary CSV",
            data=_summary_csv(df),
            file_name="AOT_Summary.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ── Section helpers ───────────────────────────────────────


def _get_section_text(title: str, df: pd.DataFrame) -> str:
    """Return plain text for a section."""
    buffers = []
    for t, func in _sections():
        if t == title:
            func(df, text_mode=True, buffer=buffers)
    return "\n".join(buffers)


def _sections():
    return [
        ("1. Research Overview", _section_overview),
        ("2. Data Collection & Sources", _section_data),
        ("3. Exploratory Data Analysis", _section_eda),
        ("4. Correlation Structure", _section_correlation),
        ("5. Social Network Analysis", _section_network),
        ("6. Machine Learning Results", _section_ml),
        ("7. Forecast", _section_forecast),
        ("8. Conclusions & Recommendations", _section_conclusions),
    ]


def _summary_csv(df: pd.DataFrame) -> str:
    import io

    summary = df.describe().round(3)
    buf = io.StringIO()
    summary.to_csv(buf)
    return buf.getvalue()


# ── Section 1 ─────────────────────────────────────────────


def _section_overview(df=None, text_mode=False, buffer=None):
    content = """
This research applies **Social Network Analysis (SNA)** and **Machine Learning** to
quantify how macroeconomic and market factors influence the stock price of Airports
of Thailand (AOT), a SET-listed company.

**Research Questions:**
1. Which factors most strongly influence AOT stock price?
2. How do these factors interact within a network structure?
3. Can we predict AOT price movements using these factors?

**Methodology:**
- Graph-theoretic centrality analysis of factor interrelationships
- 8 ML model families for one-step-ahead prediction
- SHAP-based feature importance interpretation
"""
    if text_mode:
        buffer.append(content)
    else:
        st.markdown(content)
        st.markdown("**Data Pipeline:**")
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown("**① Collect**\n\n8 data sources\n(SET, MOTS, BOT, NESDC)")
        c2.markdown("**② Engineer**\n\n11+ feature types\n(returns, lags, rolling)")
        c3.markdown("**③ Analyze**\n\nNetwork graph\n+ centrality metrics")
        c4.markdown("**④ Predict**\n\n8 ML models\n+ auto-best selection")


# ── Section 2 ─────────────────────────────────────────────


def _section_data(df=None, text_mode=False, buffer=None):
    if df is None:
        return
    content = f"""
**Dataset dimensions:** {len(df)} rows × {len(df.columns)} columns
**Date range:** {df.index[0]} to {df.index[-1]}
**Data frequency:** Monthly
"""
    if text_mode:
        buffer.append(content)
    else:
        st.markdown(content)
        st.markdown("**Variables:**")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        col_df = pd.DataFrame(
            {
                "Variable": numeric_cols,
                "Type": ["Numeric"] * len(numeric_cols),
                "Non-Null": [df[c].notna().sum() for c in numeric_cols],
                "Mean": [df[c].mean() for c in numeric_cols],
                "Std": [df[c].std() for c in numeric_cols],
            }
        )
        st.dataframe(col_df, use_container_width=True)


# ── Section 3 ─────────────────────────────────────────────


def _section_eda(df=None, text_mode=False, buffer=None):
    if df is None:
        return
    target = "aot_close" if "aot_close" in df.columns else df.columns[0]
    stats = df[target].describe()
    content = f"""
**Target variable: {target}**
- Mean: {stats["mean"]:.2f}, Std: {stats["std"]:.2f}
- Min: {stats["min"]:.2f}, Max: {stats["max"]:.2f}
- Skewness: {df[target].skew():.3f}
- Kurtosis: {df[target].kurtosis():.3f}
"""
    if text_mode:
        buffer.append(content)
    else:
        st.markdown(content)

        c1, c2 = st.columns(2)
        with c1:
            vals = df[target].dropna().values
            q1, q3 = np.percentile(vals, [25, 75])
            iqr = q3 - q1
            n_out = ((vals < q1 - 1.5 * iqr) | (vals > q3 + 1.5 * iqr)).sum()
            st.metric("Outliers (IQR)", f"{n_out} ({100 * n_out / len(vals):.1f}%)")
        with c2:
            from scipy import stats as sp_stats

            _, p_val = sp_stats.normaltest(vals)
            st.metric(
                "Normality Test (p)",
                f"{p_val:.4f}",
                delta="Normal" if p_val > 0.05 else "Non-normal",
            )


# ── Section 4 ─────────────────────────────────────────────


def _section_correlation(df=None, text_mode=False, buffer=None):
    if df is None:
        return
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    target = "aot_close" if "aot_close" in df.columns else numeric_cols[0]
    corr = df[numeric_cols].corrwith(df[target]).drop(target).sort_values()

    top_pos = corr.tail(3)
    top_neg = corr.head(3)
    content = f"""
**Top positive correlations with {target}:**
{chr(10).join(f"  + {k}: {v:.3f}" for k, v in top_pos.items())}

**Top negative correlations with {target}:**
{chr(10).join(f"  - {k}: {v:.3f}" for k, v in top_neg.items())}
"""
    if text_mode:
        buffer.append(content)
    else:
        st.markdown(content)
        st.dataframe(
            corr.to_frame("Correlation").style.background_gradient(
                cmap="RdBu_r",
                vmin=-1,
                vmax=1,
            ),
            use_container_width=True,
        )


# ── Section 5 ─────────────────────────────────────────────


def _section_network(df=None, text_mode=False, buffer=None):
    if df is None:
        return
    from aot_stock_network.network_analysis import NetworkAnalyzer, NetworkBuilder

    builder = NetworkBuilder(df)
    try:
        G = builder.build_graph(method="pearson", threshold=0.3)
        analyzer = NetworkAnalyzer(G)
        metrics = analyzer.compute_all()
        content = f"""
**Network structure:**
- Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}
- Density: {metrics.network_density:.4f}
- Communities: {metrics.n_communities}
- Components: {metrics.connected_components}
- Avg path length: {metrics.average_path_length or "N/A"}
- Diameter: {metrics.diameter or "N/A"}

**Top node by each centrality metric:**
- Degree: {max(metrics.degree_centrality, key=metrics.degree_centrality.get)}
- Betweenness: {max(metrics.betweenness_centrality, key=metrics.betweenness_centrality.get)}
- Eigenvector: {max(metrics.eigenvector_centrality, key=metrics.eigenvector_centrality.get)}
"""
    except Exception as exc:
        content = f"Network analysis unavailable: {exc}"
        metrics = None

    if text_mode:
        buffer.append(content)
    else:
        st.markdown(content)
        if metrics is not None:
            st.dataframe(metrics.to_dataframe().round(4), use_container_width=True)


# ── Section 6 ─────────────────────────────────────────────


def _section_ml(df=None, text_mode=False, buffer=None):
    results = st.session_state.get("ml_results")
    if results is None:
        content = "Run ML training on the Machine Learning page first."
        if text_mode:
            buffer.append(content)
        else:
            st.info(content)
        return

    best = results.best()
    content = f"""
**Models compared:** {len(results.results)}
**Best model:** {best.label if best else "N/A"}
**Best val RMSE:** {best.val_metrics.rmse:.4f if best else 'N/A'}
**Best test RMSE:** {best.test_metrics.rmse:.4f if best else 'N/A'}
**Best test R²:** {best.test_metrics.r2:.4f if best else 'N/A'}
"""
    if text_mode:
        buffer.append(content)
        for r in results.results:
            buffer.append(
                f"  {r.label:20s} | Val RMSE={r.val_metrics.rmse:.4f} | "
                f"Test RMSE={r.test_metrics.rmse:.4f} | R²={r.test_metrics.r2:.4f}"
            )
    else:
        st.markdown(content)
        st.dataframe(results.metrics_dataframe().round(4), use_container_width=True)

        if best and best.feature_importance is not None:
            st.markdown("**Top 5 features** (best model):")
            fi = best.feature_importance.sort_values("importance", ascending=False).head(5)
            st.dataframe(fi, use_container_width=True)


# ── Section 7 ─────────────────────────────────────────────


def _section_forecast(df=None, text_mode=False, buffer=None):
    fc = st.session_state.get("forecast_data")
    if fc is None:
        content = "Generate a forecast on the Forecast page first."
        if text_mode:
            buffer.append(content)
        else:
            st.info(content)
        return

    content = f"""
**Forecast horizon:** {fc["horizon"]} months
**Best model:** {fc["best_model"]}
**Last actual:** {fc["last_actual_value"]:.2f} THB
**Forecast end:** {fc["forecast"][-1]:.2f} THB
**Projected change:** {fc["forecast"][-1] - fc["last_actual_value"]:+.2f} THB
"""
    if text_mode:
        buffer.append(content)
    else:
        st.markdown(content)


# ── Section 8 ─────────────────────────────────────────────


def _section_conclusions(df=None, text_mode=False, buffer=None):
    content = """
**Key Findings:**
1. Factor network shows moderate density, indicating meaningful interconnections.
2. Community detection separates market factors from macroeconomic indicators.
3. Tree-based ensembles consistently outperform linear models.
4. SHAP analysis reveals tourist arrivals and exchange rate as top predictors.
5. The network approach provides structural insight beyond pairwise correlation.

**Recommendations:**
1. Use ensemble of tree-based models for monthly price prediction.
2. Monitor tourist arrivals and USD/THB as leading indicators.
3. Update the network quarterly to capture structural changes.
4. Extend with real-time data feeds for operational trading signals.

**Limitations & Future Work:**
- Monthly frequency limits short-term trading applications.
- LSTM shows potential but needs more data for full convergence.
- Incorporate additional macro factors (oil prices, geopolitical risk).
- Deploy real-time dashboard with streaming data pipeline.
"""
    if text_mode:
        buffer.append(content)
    else:
        st.markdown(content)
