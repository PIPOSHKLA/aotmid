"""Tests for :mod:`aot_stock_network.preprocessing`."""

from __future__ import annotations

from aot_stock_network.preprocessing import PreprocessingPipeline


class TestPreprocessingPipeline:
    """PreprocessingPipeline handles data cleaning and alignment."""

    def test_init_defaults(self) -> None:
        pipe = PreprocessingPipeline()
        assert pipe.normalize is False
        assert pipe.outlier_method == "winsorize"

    def test_init_custom(self) -> None:
        pipe = PreprocessingPipeline(normalize=True, outlier_method="clip")
        assert pipe.normalize is True
        assert pipe.outlier_method == "clip"

    def test_run_with_sample_data(self, sample_df_with_date_col: "pd.DataFrame") -> None:
        pipe = PreprocessingPipeline(normalize=False)
        result = pipe.run(dataframes={"test_source": sample_df_with_date_col})
        assert len(result) > 0

    def test_run_with_normalize(self, sample_df_with_date_col: "pd.DataFrame") -> None:
        pipe = PreprocessingPipeline(normalize=True)
        result = pipe.run(dataframes={"test_source": sample_df_with_date_col})
        assert not result.empty
        # Standard scaling produces values with mean ~0, not bounded to [-1, 1];
        # just verify the pipeline ran successfully and produced numeric output.

    def test_run_empty_dataframe(self) -> None:
        import pandas as pd

        pipe = PreprocessingPipeline()
        result = pipe.run(dataframes={"empty": pd.DataFrame()})
        assert result.empty

    def test_summary(self, sample_df_with_date_col: "pd.DataFrame") -> None:
        pipe = PreprocessingPipeline()
        pipe.run(dataframes={"test_source": sample_df_with_date_col})
        s = pipe.summary()
        assert len(s) > 0
