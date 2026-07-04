"""Tests for :mod:`aot_stock_network.feature_engineering`."""

from __future__ import annotations

import pandas as pd

from aot_stock_network.feature_engineering import FeatureEngineer


class TestFeatureEngineer:
    """FeatureEngineer builds feature sets from raw DataFrames."""

    def test_default_features(self, sample_df: "pd.DataFrame") -> None:
        fe = FeatureEngineer(df=sample_df)
        result = fe.build_all_features()
        assert "aot_close" in result.columns
        assert len(result) > 0

    def test_build_with_dataframe_param(self, sample_df: "pd.DataFrame") -> None:
        fe = FeatureEngineer()
        result = fe.build_all_features(df=sample_df)
        assert "aot_close" in result.columns

    def test_lags_created(self, sample_df: "pd.DataFrame") -> None:
        fe = FeatureEngineer(df=sample_df, lags=[1, 2])
        result = fe.build_all_features()
        assert any("_lag" in c for c in result.columns)

    def test_rolling_created(self, sample_df: "pd.DataFrame") -> None:
        fe = FeatureEngineer(df=sample_df, rolling_windows=[3])
        result = fe.build_all_features()
        assert any("_ma_" in c for c in result.columns) or any(
            "roll" in c.lower() for c in result.columns
        )

    def test_empty_dataframe(self) -> None:
        fe = FeatureEngineer(df=pd.DataFrame())
        result = fe.build_all_features()
        assert result.empty

    def test_feature_catalog(self, sample_df: "pd.DataFrame") -> None:
        fe = FeatureEngineer(df=sample_df)
        fe.build_all_features()
        catalog = fe.feature_catalog()
        assert isinstance(catalog, pd.DataFrame)
        assert len(catalog) > 0

    def test_summary(self, sample_df: "pd.DataFrame") -> None:
        fe = FeatureEngineer(df=sample_df)
        fe.build_all_features()
        s = fe.summary()
        assert len(s) > 0

    def test_load_on_init(self, tmp_path: "Path") -> None:
        dates = pd.period_range("2015-01", "2015-03", freq="M")
        df = pd.DataFrame({"aot_close": [50.0, 51.0, 52.0]}, index=dates)
        path = str(tmp_path / "test.csv")
        df.to_csv(path)
        fe = FeatureEngineer()
        loaded = fe.load(path)
        assert len(loaded) == 3
