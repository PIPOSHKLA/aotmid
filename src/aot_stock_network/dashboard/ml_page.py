"""Machine Learning page: model comparison, predictions, feature importance, SHAP."""

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from aot_stock_network.dashboard.utils import (
    download_button,
    download_figure,
    get_data,
    inject_css,
)
from aot_stock_network.prediction import PredictionPipeline

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")


def show() -> None:
    inject_css(st.session_state.get("dark_mode", False))
    df = get_data()

    st.title("Machine Learning")
    st.markdown("Train, compare, and interpret ML models for AOT price prediction.")

    target = "aot_close" if "aot_close" in df.columns else df.columns[0]
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_candidates = [c for c in numeric_cols if c != target]

    # ── Training Controls ─────────────────────────────────
    with st.sidebar:
        st.markdown("### ML Controls")
        available_models = {
            "Linear Regression": "lr",
            "Random Forest": "rf",
            "XGBoost": "xgb",
            "LightGBM": "lgbm",
            "CatBoost": "cb",
            "ARIMA": "arima",
            "Prophet": "prophet",
            "LSTM": "lstm",
        }
        selected_labels = st.multiselect(
            "Models to train",
            list(available_models.keys()),
            default=["Random Forest", "XGBoost", "ARIMA"],
        )
        selected_models = [available_models[label] for label in selected_labels]

        test_months = st.slider("Test months", 3, 24, 12)
        val_months = st.slider("Validation months", 3, 12, 6)
        do_tune = st.checkbox(
            "Hyperparameter tuning",
            value=False,
            help="Grid search with TimeSeriesSplit CV. Increases runtime significantly.",
        )
        do_shap = st.checkbox("SHAP analysis", value=True)

    # ── Feature selection ─────────────────────────────────
    with st.expander("⚙️ Feature Configuration", expanded=False):
        selected_features = st.multiselect(
            "Features for ML models",
            feature_candidates,
            default=feature_candidates[: min(6, len(feature_candidates))],
        )
        if not selected_features:
            st.warning("Select at least one feature.")
            selected_features = feature_candidates[:1]

    # ── Run Training ──────────────────────────────────────
    if st.button("🚀 Train Models", type="primary", use_container_width=True):
        if not selected_models:
            st.error("Select at least one model.")
            return

        with st.spinner("Training models... This may take a moment."):
            pipe = PredictionPipeline(
                df[selected_features + [target]],
                target_col=target,
                test_months=test_months,
                val_months=val_months,
            )
            results = pipe.run(
                models=selected_models,
                tune=do_tune,
                calc_shap=do_shap,
            )
            st.session_state.ml_results = results
            st.session_state.ml_pipeline = pipe
            st.session_state.selected_model = results.best_model_name
            st.success(
                f"Training complete! Best model: "
                f"{results.best().label if results.best() else 'N/A'}"
            )

    # ── Display Results ───────────────────────────────────
    results = st.session_state.get("ml_results")
    if results is None:
        st.info("Configure models above and click **Train Models** to begin.")
        return

    # ── Summary table ─────────────────────────────────────
    st.markdown("### Model Comparison")
    df_metrics = results.metrics_dataframe()
    styled = df_metrics.style.background_gradient(
        cmap="RdYlGn_r",
        subset=["val_rmse", "val_mae", "val_mape"],
    ).background_gradient(cmap="RdYlGn", subset=["val_r2", "test_r2"])
    st.dataframe(styled, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    best = results.best()
    if best:
        c1.metric("Best Model", best.label)
        c2.metric("Validation RMSE", f"{best.val_metrics.rmse:.4f}")
        c3.metric("Test RMSE", f"{best.test_metrics.rmse:.4f}")

    st.divider()

    # ── Visualization tabs ────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📈 Predictions",
            "📊 Residuals",
            "⭐ Feature Importance",
            "🎯 SHAP",
        ]
    )

    model_names = [r.name for r in results.results]
    model_name = st.selectbox(
        "Select model for plots",
        model_names,
        index=model_names.index(st.session_state.selected_model)
        if st.session_state.selected_model in model_names
        else 0,
        key="ml_model_select",
    )

    # ── Tab 1: Predictions ────────────────────────────────
    with tab1:
        fig_p, _ = results.plot_predictions(model_name=model_name)
        st.pyplot(fig_p, use_container_width=True)
        download_figure(fig_p, f"predictions_{model_name}.svg")
        plt.close(fig_p)

        # Summary
        target_res = next((r for r in results.results if r.name == model_name), None)
        if target_res:
            st.markdown(
                f"**{target_res.label}** — "
                f"Test RMSE={target_res.test_metrics.rmse:.4f}, "
                f"R²={target_res.test_metrics.r2:.4f}"
            )

    # ── Tab 2: Residuals ─────────────────────────────────
    with tab2:
        fig_r, _ = results.plot_residuals(model_name=model_name)
        st.pyplot(fig_r, use_container_width=True)
        download_figure(fig_r, f"residuals_{model_name}.svg")
        plt.close(fig_r)

    # ── Tab 3: Feature Importance ─────────────────────────
    with tab3:
        fi_result = results.plot_feature_importance(model_name=model_name)
        if fi_result:
            fig_fi, _ = fi_result
            st.pyplot(fig_fi, use_container_width=True)
            download_figure(fig_fi, f"feature_importance_{model_name}.svg")
            plt.close(fig_fi)
        else:
            st.info(
                f"Feature importance not available for "
                f"{next((r.label for r in results.results if r.name == model_name), model_name)}. "
                "Only tree-based models and linear regression provide this."
            )

        # Full importance table
        target_res = next((r for r in results.results if r.name == model_name), None)
        if target_res and target_res.feature_importance is not None:
            with st.expander("📋 Full Importance Table"):
                fi_df = target_res.feature_importance.sort_values(
                    "importance",
                    ascending=False,
                )
                st.dataframe(fi_df, use_container_width=True)
                download_button(fi_df, f"feature_importance_{model_name}.csv")

    # ── Tab 4: SHAP ───────────────────────────────────────
    with tab4:
        shap_result = results.plot_shap(model_name=model_name)
        if shap_result:
            fig_s, _ = shap_result
            st.pyplot(fig_s, use_container_width=True)
            download_figure(fig_s, f"shap_summary_{model_name}.svg")
            plt.close(fig_s)
        else:
            st.info(
                "SHAP values not available for this model. "
                "SHAP is supported for tree-based models (RF, XGB, LGBM, CB) "
                "and linear regression."
            )

    # ── Download all metrics ──────────────────────────────
    with st.expander("⬇️ Export All Metrics"):
        download_button(df_metrics.reset_index(), "ml_model_comparison.csv")
