"""Tests for :mod:`aot_stock_network.prediction`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aot_stock_network.prediction import (
    EvalMetrics,
    PredictionPipeline,
)


class TestEvalMetrics:
    """Evaluation metric calculations."""

    def test_compute_returns_correct_types(self) -> None:
        y_true = np.array([50.0, 52.0, 51.0, 53.0, 54.0])
        y_pred = np.array([49.5, 52.5, 50.8, 53.2, 53.8])
        m = EvalMetrics.compute(y_true, y_pred)
        assert m.rmse > 0
        assert m.mae > 0
        assert 0 < m.mape < 100
        assert -1 < m.r2 <= 1

    def test_perfect_prediction(self) -> None:
        y = np.array([1.0, 2.0, 3.0])
        m = EvalMetrics.compute(y, y)
        assert m.rmse == 0.0
        assert m.r2 == 1.0

    def test_to_dict(self) -> None:
        y_true = np.array([50.0, 52.0])
        y_pred = np.array([49.5, 52.5])
        m = EvalMetrics.compute(y_true, y_pred)
        d = m.to_dict()
        assert "rmse" in d
        assert "mae" in d
        assert "mape" in d
        assert "r2" in d

    def test_str(self) -> None:
        y_true = np.array([50.0, 52.0])
        y_pred = np.array([49.5, 52.5])
        m = EvalMetrics.compute(y_true, y_pred)
        s = str(m)
        assert "RMSE" in s


class TestPredictionPipeline:
    """End-to-end pipeline validation."""

    def test_basic_rf_run(self, small_df: "pd.DataFrame") -> None:
        pipe = PredictionPipeline(
            small_df,
            target_col="aot_close",
            test_months=4,
            val_months=2,
        )
        results = pipe.run(models=["rf"], tune=False, calc_shap=False)
        assert len(results.results) == 1
        r = results.results[0]
        assert r.name == "rf"
        assert r.val_metrics is not None
        assert r.test_metrics is not None
        assert r.val_metrics.rmse > 0

    def test_multiple_models(self, small_df: "pd.DataFrame") -> None:
        pipe = PredictionPipeline(
            small_df,
            target_col="aot_close",
            test_months=4,
            val_months=2,
        )
        results = pipe.run(models=["lr", "rf", "xgb"], tune=False, calc_shap=False)
        assert len(results.results) == 3

    def test_best_model_returns_top(self, small_df: "pd.DataFrame") -> None:
        pipe = PredictionPipeline(
            small_df,
            target_col="aot_close",
            test_months=4,
            val_months=2,
        )
        results = pipe.run(models=["lr", "rf"], tune=False, calc_shap=False)
        best = results.best()
        assert best is not None
        assert best.name in ("lr", "rf")

    def test_summary(self, small_df: "pd.DataFrame") -> None:
        pipe = PredictionPipeline(
            small_df,
            target_col="aot_close",
            test_months=4,
            val_months=2,
        )
        results = pipe.run(models=["rf"], tune=False, calc_shap=False)
        s = results.summary()
        assert "rf" in s or "Random" in s

    def test_invalid_model_name(self, small_df: "pd.DataFrame") -> None:
        pipe = PredictionPipeline(
            small_df,
            target_col="aot_close",
            test_months=4,
            val_months=2,
        )
        results = pipe.run(models=["nonexistent"], tune=False, calc_shap=False)
        assert len(results.results) == 0

    def test_empty_results_summary(self, small_df: "pd.DataFrame") -> None:
        pipe = PredictionPipeline(
            small_df,
            target_col="aot_close",
            test_months=4,
            val_months=2,
        )
        results = pipe.run(models=["nonexistent"], tune=False, calc_shap=False)
        s = results.summary()
        assert "Train" in s


class TestPredictionResults:
    """Result container and plotting."""

    @pytest.fixture(autouse=True)
    def _setup(self, small_df: "pd.DataFrame") -> None:
        pipe = PredictionPipeline(
            small_df,
            target_col="aot_close",
            test_months=4,
            val_months=2,
        )
        self.results = pipe.run(models=["rf"], tune=False, calc_shap=False)

    def test_plot_model_comparison(self) -> None:
        from matplotlib.figure import Figure

        fig, _ = self.results.plot_model_comparison()
        assert isinstance(fig, Figure)

    def test_plot_predictions(self) -> None:
        from matplotlib.figure import Figure

        fig, _ = self.results.plot_predictions("rf")
        assert isinstance(fig, Figure)

    def test_metrics_dataframe(self) -> None:
        df = self.results.metrics_dataframe()
        assert len(df) > 0
        assert df.index.name == "model" or "params" in df.columns
