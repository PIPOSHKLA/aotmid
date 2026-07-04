"""Shared fixtures for the test suite."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

# Force non-interactive matplotlib backend for CI/headless environments
os.environ.setdefault("MPLBACKEND", "Agg")


@pytest.fixture(scope="session")
def sample_dates() -> pd.PeriodIndex:
    """120 months of monthly PeriodIndex (2015-01 to 2024-12)."""
    return pd.period_range("2015-01", "2024-12", freq="M")


@pytest.fixture(scope="session")
def sample_df(sample_dates: pd.PeriodIndex) -> pd.DataFrame:
    """Full synthetic DataFrame with all expected columns."""
    n = len(sample_dates)
    rng = np.random.default_rng(42)

    df = pd.DataFrame(
        {
            "aot_close": 50.0 + np.cumsum(rng.normal(0, 2, n)),
            "set_close": 1500.0 + np.cumsum(rng.normal(0, 10, n)),
            "fx_usdthb_rate": 32.0 + rng.normal(0, 0.5, n),
            "tourists_total_arrivals": 2e6 + rng.integers(-300_000, 300_000, n),
            "cpi_cpi_headline": 100.0 + np.cumsum(rng.normal(0, 0.1, n)),
            "policy_policy_rate": 1.5 + rng.normal(0, 0.1, n),
            "gdp_gdp": 500.0 + np.cumsum(rng.normal(0, 2, n)),
            "aot_volume": 2e7 + rng.integers(-5_000_000, 5_000_000, n),
        },
        index=sample_dates,
    )
    return df


@pytest.fixture(scope="session")
def sample_df_with_missing(sample_df: pd.DataFrame) -> pd.DataFrame:
    """Same data with small patches of NaN."""
    df = sample_df.copy()
    df.iloc[10:12, 0] = np.nan
    df.iloc[30:33, 2] = np.nan
    df.iloc[55, 4] = np.nan
    return df


@pytest.fixture(scope="session")
def small_df(sample_dates: pd.PeriodIndex) -> pd.DataFrame:
    """Tiny dataset (24 rows x 4 cols) for fast smoke tests."""
    n = 24
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "aot_close": 50.0 + np.cumsum(rng.normal(0, 1, n)),
            "set_close": 1500.0 + np.cumsum(rng.normal(0, 5, n)),
            "fx_usdthb_rate": 32.0 + rng.normal(0, 0.3, n),
            "aot_volume": 2e7 + rng.integers(-1_000_000, 1_000_000, n),
        },
        index=sample_dates[:24],
    )


@pytest.fixture(scope="session")
def sample_df_with_date_col(sample_dates: pd.PeriodIndex) -> pd.DataFrame:
    """Same as sample_df but with a 'date' column for PreprocessingPipeline."""
    df = pd.DataFrame(
        {
            "date": [str(d) for d in sample_dates],
            "aot_close": 50.0
            + np.cumsum(np.random.default_rng(42).normal(0, 2, len(sample_dates))),
            "set_close": 1500.0
            + np.cumsum(np.random.default_rng(42).normal(0, 10, len(sample_dates))),
            "fx_usdthb_rate": 32.0 + np.random.default_rng(42).normal(0, 0.5, len(sample_dates)),
            "tourists_total_arrivals": 2e6
            + np.random.default_rng(42).integers(-300_000, 300_000, len(sample_dates)),
            "cpi_cpi_headline": 100.0
            + np.cumsum(np.random.default_rng(42).normal(0, 0.1, len(sample_dates))),
            "policy_policy_rate": 1.5 + np.random.default_rng(42).normal(0, 0.1, len(sample_dates)),
            "gdp_gdp": 500.0 + np.cumsum(np.random.default_rng(42).normal(0, 2, len(sample_dates))),
            "aot_volume": 2e7
            + np.random.default_rng(42).integers(-5_000_000, 5_000_000, len(sample_dates)),
        },
    )
    return df
