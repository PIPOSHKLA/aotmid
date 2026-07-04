"""Download Center: export data, figures, models, and reports."""

import io
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from aot_stock_network.dashboard.utils import (
    download_button,
    get_data,
    inject_css,
)
from aot_stock_network.visualization import (
    plot_correlation_heatmap,
    plot_distribution_grid,
)


def show() -> None:
    inject_css(st.session_state.get("dark_mode", False))
    df = get_data()

    st.title("Download Center")
    st.markdown("Export all data, figures, and results in one place.")

    tab1, tab2, tab3 = st.tabs(["📊 Data", "📈 Figures", "📋 Reports"])

    # ── Tab 1: Data Exports ───────────────────────────────
    with tab1:
        st.markdown("### Data Exports")
        st.markdown("Download the processed datasets as CSV files.")

        c1, c2 = st.columns(2)
        with c1:
            download_button(df, "aot_feature_dataset.csv", "📥 Download Feature Dataset (CSV)")
        with c2:
            desc = df.describe()
            download_button(
                desc, "aot_summary_statistics.csv", "📥 Download Summary Statistics (CSV)"
            )

        with st.expander("🔍 Select specific columns for export"):
            all_cols = df.columns.tolist()
            selected = st.multiselect("Columns", all_cols, default=all_cols)
            if selected:
                download_button(
                    df[selected], "aot_selected_columns.csv", "📥 Download Selection (CSV)"
                )

    # ── Tab 2: Figure Exports ─────────────────────────────
    with tab2:
        st.markdown("### Figure Exports")
        st.markdown("Generate and download publication-ready figures (SVG format, 300 DPI).")

        with st.spinner("Generating figures..."):
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

            fig1, _ = plot_correlation_heatmap(df[numeric_cols])
            st.markdown("#### Correlation Heatmap")
            st.pyplot(fig1, use_container_width=True)
            download_figure_local(fig1, "correlation_heatmap.svg")
            plt.close(fig1)

            fig2, _ = plot_distribution_grid(df)
            st.markdown("#### Distribution Grid")
            st.pyplot(fig2, use_container_width=True)
            download_figure_local(fig2, "distribution_grid.svg")
            plt.close(fig2)

        # ── Batch download ────────────────────────────────
        st.markdown("#### Batch Export")
        if st.button("📦 Generate All Figures as ZIP", type="primary"):
            with st.spinner("Generating all figures..."):
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                    _add_figure_to_zip(
                        zf, plot_correlation_heatmap(df[numeric_cols])[0], "correlation_heatmap.svg"
                    )
                    _add_figure_to_zip(zf, plot_distribution_grid(df)[0], "distribution_grid.svg")
                buf.seek(0)
                st.download_button(
                    "⬇️ Download ZIP",
                    data=buf,
                    file_name="aot_figures.zip",
                    mime="application/zip",
                    use_container_width=True,
                )

    # ── Tab 3: Report Exports ─────────────────────────────
    with tab3:
        st.markdown("### Report Exports")
        st.markdown("Download the research report and model results.")

        # Generate report text
        report_lines = [
            "# AOT Stock Network Analysis — Research Report",
            "",
            "## Dataset Summary",
            f"- Rows: {len(df)}, Columns: {len(df.columns)}",
            f"- Date range: {df.index[0]} to {df.index[-1]}",
            "",
            "## Variable Summary",
        ]
        for col in df.columns:
            if df[col].dtype in (np.float64, np.int64):
                report_lines.append(f"- {col}: mean={df[col].mean():.2f}, std={df[col].std():.2f}")
            else:
                report_lines.append(f"- {col}: {df[col].dtype}")

        # ML results
        ml_results = st.session_state.get("ml_results")
        if ml_results:
            report_lines.extend(
                [
                    "",
                    "## ML Results",
                    f"- Best model: {ml_results.best().label if ml_results.best() else 'N/A'}",
                    f"- Test RMSE: {ml_results.best().test_metrics.rmse:.4f}"
                    if ml_results.best()
                    else "",
                ]
            )

        report_text = "\n".join(report_lines)

        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "📄 Download Report (Markdown)",
                data=report_text,
                file_name="AOT_Research_Report.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with c2:
            st.download_button(
                "📋 Download Session Report (CSV)",
                data=ml_results.metrics_dataframe().to_csv()
                if ml_results
                else df.describe().to_csv(),
                file_name="aot_session_report.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with st.expander("📖 Report Preview"):
            st.markdown(report_text)


# ── helpers ───────────────────────────────────────────────


def download_figure_local(fig, filename: str) -> None:
    """Download button for a matplotlib figure."""
    buf = io.BytesIO()
    fig.savefig(buf, format="svg", dpi=300, bbox_inches="tight")
    buf.seek(0)
    st.download_button(
        label=f"⬇️ Download {filename}",
        data=buf,
        file_name=filename,
        mime="image/svg+xml",
        use_container_width=True,
    )


def _add_figure_to_zip(zf, fig, filename: str) -> None:
    buf = io.BytesIO()
    fig.savefig(buf, format="svg", dpi=300, bbox_inches="tight")
    buf.seek(0)
    zf.writestr(filename, buf.getvalue())
