"""Tests for :mod:`aot_stock_network.visualization`."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from matplotlib.figure import Figure

from aot_stock_network.visualization import EDAVisualizer, VisualizationConfig


class TestEDAVisualizer:
    """Every plot method returns a Figure for SVG/PNG export."""

    @pytest.fixture(autouse=True)
    def _setup(self, sample_df: "pd.DataFrame") -> None:
        self.viz = EDAVisualizer(df=sample_df)
        self.config = VisualizationConfig()

    def test_config_defaults(self) -> None:
        assert self.config.figsize[0] == 12
        assert self.config.dpi == 150

    def test_plot_time_series(self) -> None:
        fig = self.viz.plot_time_series(column="aot_close")
        assert isinstance(fig, Figure)

    def test_plot_distribution(self) -> None:
        fig = self.viz.plot_distribution(column="aot_close")
        assert isinstance(fig, Figure)

    def test_plot_correlation_heatmap(self) -> None:
        fig = self.viz.plot_correlation_heatmap()
        assert isinstance(fig, Figure)

    def test_plot_monthly_trend(self) -> None:
        fig = self.viz.plot_monthly_trend(column="aot_close")
        assert isinstance(fig, Figure)

    def test_plot_yearly_trend(self) -> None:
        fig = self.viz.plot_yearly_trend(column="aot_close")
        assert isinstance(fig, Figure)

    def test_plot_scatter(self) -> None:
        fig = self.viz.plot_scatter(x_col="set_close", y_col="aot_close")
        assert isinstance(fig, Figure)

    def test_plot_outlier_box(self) -> None:
        fig = self.viz.plot_outlier_box()
        assert isinstance(fig, Figure)

    def test_plot_outlier_timeseries(self) -> None:
        fig = self.viz.plot_outlier_timeseries(column="aot_close")
        assert isinstance(fig, Figure)

    def test_load(self, tmp_path: "Path") -> None:
        dates = pd.period_range("2015-01", "2015-03", freq="M")
        df = pd.DataFrame({"aot_close": [50.0, 51.0, 52.0]}, index=dates)
        path = str(tmp_path / "test.csv")
        df.to_csv(path)
        viz = EDAVisualizer()
        viz.load(path)
        assert viz.df is not None
        assert len(viz.df) == 3

    def test_generate_all(self, tmp_path: "Path") -> None:
        output_dir = tmp_path / "figures"
        output_dir.mkdir()
        paths = self.viz.generate_all(
            output_dir=str(output_dir), include_pairplot=False, include_decomposition=False
        )
        assert isinstance(paths, dict)
