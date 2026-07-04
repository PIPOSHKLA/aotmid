"""
feature_engineering.py — Feature Creation for AOT Stock Prediction
===================================================================

Builds predictive features on top of the aligned monthly dataset produced
by the PreprocessingPipeline. Covers all required feature types:

  Required features
  -----------------
  • Monthly Return      — (close_t - close_{t-1}) / close_{t-1}
  • Log Return          — ln(close_t / close_{t-1})
  • Moving Average      — rolling mean of close (windows 3, 6, 12)
  • Rolling Mean        — alias for Moving Average
  • Rolling Std         — rolling std of close (windows 3, 6, 12)
  • Lag 1               — value shifted by 1 month
  • Lag 3               — value shifted by 3 months
  • Lag 6               — value shifted by 6 months
  • Tourist Growth      — month-over-month tourist arrival growth
  • Exchange Rate Change— month-over-month USD/THB change
  • Volume Change       — month-over-month AOT volume change

All values are computed using only past information. Rolling statistics
use closed='left' semantics so the current observation is not included
in its own window.

Usage
-----
    from aot_stock_network.feature_engineering import FeatureEngineer

    fe = FeatureEngineer(df_aligned)
    df_features = fe.build_all_features()
    fe.export("data/processed/feature_dataset.csv")
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────
DEFAULT_ROLLING_WINDOWS = [3, 6, 12]
DEFAULT_LAGS = [1, 3, 6]
MIN_PERIODS_FOR_ROLLING = 3

# Expected column name prefixes from PreprocessingPipeline
# (used to auto-detect which columns are available)
EXPECTED_PREFIXES = {
    "aot": {"close", "volume", "open", "high", "low", "value"},
    "set": {"close", "volume"},
    "tourists": {"total_arrivals", "arrivals_china", "arrivals_east_asia"},
    "revenue": {"total_revenue_mb"},
    "fx": {"usdthb_rate", "bid_rate", "ask_rate"},
    "policy": {"policy_rate"},
    "cpi": {"cpi_headline", "cpi_core"},
}


# ──────────────────────────────────────────────────────────────
# Utility helpers
# ──────────────────────────────────────────────────────────────
def _setup_logging(level: str = "INFO") -> None:
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | features | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        logger.addHandler(handler)


def _find_column(
    df: pd.DataFrame,
    prefix: str,
    suffix: str,
    required: bool = False,
) -> Optional[str]:
    """Find a column by prefix_suffix pattern, returning None if missing."""
    candidates = [c for c in df.columns if c.startswith(f"{prefix}_") and c.endswith(suffix)]
    # Exact match first
    exact = f"{prefix}_{suffix}"
    if exact in candidates:
        return exact
    if candidates:
        return candidates[0]
    if required:
        logger.warning("column not found: '%s_%s'", prefix, suffix)
    return None


# ──────────────────────────────────────────────────────────────
# 1. Monthly Return
# ──────────────────────────────────────────────────────────────
def add_monthly_return(
    df: pd.DataFrame,
    price_col: str = "aot_close",
    target_col: str = "aot_return",
) -> pd.DataFrame:
    """Add monthly return column: (P_t - P_{t-1}) / P_{t-1}.

    Parameters
    ----------
    df : pd.DataFrame
        Monthly data with PeriodIndex.
    price_col : str, default "aot_close"
        Column name for the price series.
    target_col : str, default "aot_return"
        Name for the new return column.

    Returns
    -------
    pd.DataFrame with the new column added.

    Transformation documented
    -------------------------
    Monthly return = (close_current - close_previous) / close_previous
    The first observation will have NaN (no prior month).
    Values are decimal fractions (e.g., 0.05 = 5% return).
    """
    if price_col not in df.columns:
        logger.warning("add_monthly_return: column '%s' not found", price_col)
        return df

    df = df.copy()
    df[target_col] = df[price_col].pct_change()
    logger.info("add_monthly_return: '%s' from '%s'", target_col, price_col)
    return df


# ──────────────────────────────────────────────────────────────
# 2. Log Return
# ──────────────────────────────────────────────────────────────
def add_log_return(
    df: pd.DataFrame,
    price_col: str = "aot_close",
    target_col: str = "aot_log_return",
) -> pd.DataFrame:
    """Add log return column: ln(P_t / P_{t-1}).

    Parameters
    ----------
    df : pd.DataFrame
        Monthly data with PeriodIndex.
    price_col : str, default "aot_close"
        Column name for the price series.
    target_col : str, default "aot_log_return"
        Name for the new log return column.

    Returns
    -------
    pd.DataFrame with the new column added.

    Transformation documented
    -------------------------
    Log return = ln(close_current / close_previous)
    Log returns are approximately equal to simple returns for small
    changes and have better statistical properties (additivity over time).
    The first observation will have NaN.
    """
    if price_col not in df.columns:
        logger.warning("add_log_return: column '%s' not found", price_col)
        return df

    df = df.copy()
    df[target_col] = np.log(df[price_col] / df[price_col].shift(1))
    logger.info("add_log_return: '%s' from '%s'", target_col, price_col)
    return df


# ──────────────────────────────────────────────────────────────
# 3 & 4. Moving Average / Rolling Mean
# ──────────────────────────────────────────────────────────────
def add_moving_average(
    df: pd.DataFrame,
    col: str = "aot_close",
    windows: Optional[List[int]] = None,
    prefix: str = "aot",
) -> pd.DataFrame:
    """Add moving average (rolling mean) columns for specified windows.

    This also creates the 'Rolling Mean' feature (same computation).

    Parameters
    ----------
    df : pd.DataFrame
        Monthly data with PeriodIndex.
    col : str, default "aot_close"
        Column to compute rolling mean on.
    windows : list of int, optional
        Window sizes. Default: [3, 6, 12] months.
    prefix : str, default "aot"
        Prefix for output column names.

    Returns
    -------
    pd.DataFrame with MA columns added.

    Transformation documented
    -------------------------
    Moving Average (MA_n) = rolling mean of the last n months
      using min_periods=n (requires all n observations).
    The current month IS included in its own window (closed='right'
    is default for rolling).

    Also exported as 'Rolling Mean' (same computation, different name).
    """
    if windows is None:
        windows = DEFAULT_ROLLING_WINDOWS
    if col not in df.columns:
        logger.warning("add_moving_average: column '%s' not found", col)
        return df

    df = df.copy()
    for w in windows:
        if len(df) < w:
            logger.warning("add_moving_average: insufficient rows (%d) for window %d", len(df), w)
            continue
        ma_col = f"{prefix}_ma_{w}"
        rm_col = f"{prefix}_rolling_mean_{w}"
        values = df[col].rolling(window=w, min_periods=w).mean()
        df[ma_col] = values
        df[rm_col] = values.copy()

    logger.info("add_moving_average: windows=%s on '%s'", windows, col)
    return df


# ──────────────────────────────────────────────────────────────
# 5. Rolling Std
# ──────────────────────────────────────────────────────────────
def add_rolling_std(
    df: pd.DataFrame,
    col: str = "aot_close",
    windows: Optional[List[int]] = None,
    prefix: str = "aot",
) -> pd.DataFrame:
    """Add rolling standard deviation columns for specified windows.

    Parameters
    ----------
    df : pd.DataFrame
        Monthly data with PeriodIndex.
    col : str, default "aot_close"
        Column to compute rolling std on.
    windows : list of int, optional
        Window sizes. Default: [3, 6, 12] months.
    prefix : str, default "aot"
        Prefix for output column names.

    Returns
    -------
    pd.DataFrame with rolling std columns added.

    Transformation documented
    -------------------------
    Rolling Standard Deviation (std_n) = rolling std of the last n months.
    Measures volatility over the window. Current month is included in the
    window. Requires all n observations (min_periods=n).
    """
    if windows is None:
        windows = DEFAULT_ROLLING_WINDOWS
    if col not in df.columns:
        logger.warning("add_rolling_std: column '%s' not found", col)
        return df

    df = df.copy()
    for w in windows:
        if len(df) < w:
            continue
        std_col = f"{prefix}_rolling_std_{w}"
        df[std_col] = df[col].rolling(window=w, min_periods=w).std()

    logger.info("add_rolling_std: windows=%s on '%s'", windows, col)
    return df


# ──────────────────────────────────────────────────────────────
# 6, 7, 8. Lag features
# ──────────────────────────────────────────────────────────────
def add_lag_features(
    df: pd.DataFrame,
    col: str = "aot_close",
    lags: Optional[List[int]] = None,
    prefix: str = "aot",
) -> pd.DataFrame:
    """Add lagged (shifted) versions of a column.

    Parameters
    ----------
    df : pd.DataFrame
        Monthly data with PeriodIndex.
    col : str, default "aot_close"
        Column to lag.
    lags : list of int, optional
        Lag periods in months. Default: [1, 3, 6].
    prefix : str, default "aot"
        Prefix for output column names.

    Returns
    -------
    pd.DataFrame with lag columns added.

    Transformation documented
    -------------------------
    Lag_n = value shifted backward by n periods (i.e., value at t-n).
    The first n observations will be NaN.
    Lag features allow the model to use previous values of the same series
    as predictors, capturing autocorrelation in the target.
    """
    if lags is None:
        lags = DEFAULT_LAGS
    if col not in df.columns:
        logger.warning("add_lag_features: column '%s' not found", col)
        return df

    df = df.copy()
    for lag in lags:
        lag_col = f"{prefix}_lag_{lag}"
        df[lag_col] = df[col].shift(lag)

    logger.info("add_lag_features: lags=%s on '%s'", lags, col)
    return df


# ──────────────────────────────────────────────────────────────
# 9. Tourist Growth
# ──────────────────────────────────────────────────────────────
def add_tourist_growth(
    df: pd.DataFrame,
    tourist_col: Optional[str] = None,
    target_col: str = "tourist_growth",
) -> pd.DataFrame:
    """Add month-over-month tourist arrival growth rate.

    Parameters
    ----------
    df : pd.DataFrame
        Monthly data.
    tourist_col : str, optional
        Column with total tourist arrivals. Auto-detected if None.
    target_col : str, default "tourist_growth"
        Name for the new growth column.

    Returns
    -------
    pd.DataFrame with the new column added.

    Transformation documented
    -------------------------
    Tourist Growth = (arrivals_t - arrivals_{t-1}) / arrivals_{t-1}
    Negative values indicate a decline in arrivals. The first month will
    have NaN (no prior month for comparison).
    """
    if tourist_col is None:
        tourist_col = _find_column(df, "tourists", "total_arrivals")

    if tourist_col is None or tourist_col not in df.columns:
        logger.warning("add_tourist_growth: tourist arrivals column not found")
        return df

    df = df.copy()
    df[target_col] = df[tourist_col].pct_change()
    logger.info("add_tourist_growth: '%s' from '%s'", target_col, tourist_col)
    return df


# ──────────────────────────────────────────────────────────────
# 10. Exchange Rate Change
# ──────────────────────────────────────────────────────────────
def add_exchange_rate_change(
    df: pd.DataFrame,
    fx_col: Optional[str] = None,
    target_col: str = "fx_change",
) -> pd.DataFrame:
    """Add month-over-month USD/THB exchange rate change.

    Parameters
    ----------
    df : pd.DataFrame
        Monthly data.
    fx_col : str, optional
        Column with USD/THB rate. Auto-detected if None.
    target_col : str, default "fx_change"
        Name for the new change column.

    Returns
    -------
    pd.DataFrame with the new column added.

    Transformation documented
    -------------------------
    Exchange Rate Change = (rate_t - rate_{t-1}) / rate_{t-1}
    Positive values indicate THB depreciation (more baht per dollar).
    The first month will have NaN.
    """
    if fx_col is None:
        fx_col = _find_column(df, "fx", "usdthb_rate")

    if fx_col is None or fx_col not in df.columns:
        logger.warning("add_exchange_rate_change: fx column not found")
        return df

    df = df.copy()
    df[target_col] = df[fx_col].pct_change()
    logger.info("add_exchange_rate_change: '%s' from '%s'", target_col, fx_col)
    return df


# ──────────────────────────────────────────────────────────────
# 11. Volume Change
# ──────────────────────────────────────────────────────────────
def add_volume_change(
    df: pd.DataFrame,
    volume_col: Optional[str] = None,
    target_col: str = "volume_change",
) -> pd.DataFrame:
    """Add month-over-month AOT trading volume change.

    Parameters
    ----------
    df : pd.DataFrame
        Monthly data.
    volume_col : str, optional
        Column with AOT volume. Auto-detected if None.
    target_col : str, default "volume_change"
        Name for the new change column.

    Returns
    -------
    pd.DataFrame with the new column added.

    Transformation documented
    -------------------------
    Volume Change = (volume_t - volume_{t-1}) / volume_{t-1}
    Captures changes in trading activity. The first month will have NaN.
    """
    if volume_col is None:
        volume_col = _find_column(df, "aot", "volume")

    if volume_col is None or volume_col not in df.columns:
        logger.warning("add_volume_change: volume column not found")
        return df

    df = df.copy()
    df[target_col] = df[volume_col].pct_change()
    logger.info("add_volume_change: '%s' from '%s'", target_col, volume_col)
    return df


# ──────────────────────────────────────────────────────────────
# 12. Feature Engineer orchestrator
# ──────────────────────────────────────────────────────────────
@dataclass
class FeatureEngineeringReport:
    """Summary of feature engineering results."""

    input_shape: Tuple[int, int] = (0, 0)
    output_shape: Tuple[int, int] = (0, 0)
    features_added: List[str] = field(default_factory=list)
    n_nan_rows: int = 0
    output_path: str = ""


class FeatureEngineer:
    """Orchestrator for building all predictive features.

    Takes the aligned monthly DataFrame from PreprocessingPipeline and
    creates all required feature groups. Also creates lagged versions
    of key exogenous variables (tourist arrivals, USD/THB rate, policy
    rate, CPI) to maximize model input richness.

    Parameters
    ----------
    df : pd.DataFrame, optional
        Aligned monthly DataFrame. Can be loaded later via load().
    auto_detect : bool, default True
        If True, auto-detect available columns and only add features
        for which source columns exist.
    rolling_windows : list of int, default [3, 6, 12]
        Window sizes for moving averages and rolling std.
    lags : list of int, default [1, 3, 6]
        Lag periods for lag features.
    """

    def __init__(
        self,
        df: Optional[pd.DataFrame] = None,
        auto_detect: bool = True,
        rolling_windows: Optional[List[int]] = None,
        lags: Optional[List[int]] = None,
        log_level: str = "INFO",
    ):
        self.df = df.copy() if df is not None else None
        self.auto_detect = auto_detect
        self.rolling_windows = rolling_windows or DEFAULT_ROLLING_WINDOWS
        self.lags = lags or DEFAULT_LAGS
        self.report = FeatureEngineeringReport()

        _setup_logging(log_level)

        # Track which transforms were applied (for reporting and dedup)
        self._applied_transforms: Set[str] = set()

        if self.df is not None:
            self.report.input_shape = self.df.shape
            logger.info(
                "FeatureEngineer initialized | input=%d rows x %d cols | windows=%s | lags=%s",
                self.df.shape[0],
                self.df.shape[1],
                self.rolling_windows,
                self.lags,
            )

    # ── Data loading ────────────────────────────────────────
    def load(self, path: Union[str, Path]) -> pd.DataFrame:
        """Load aligned monthly dataset from CSV."""
        p = Path(path)
        df = pd.read_csv(p, parse_dates=False, index_col=0)
        if "month" in df.columns:
            df["month"] = pd.to_datetime(df["month"], errors="coerce")
            df.set_index("month", inplace=True)
        df.index = pd.PeriodIndex(df.index, freq="M")
        self.df = df
        self.report.input_shape = df.shape
        logger.info("load: loaded %d rows x %d cols from %s", df.shape[0], df.shape[1], p)
        return df

    # ── Column detection ────────────────────────────────────
    def _has_column(self, pattern: str) -> bool:
        """Check if any column matches 'pattern'."""
        if self.df is None:
            return False
        return any(pattern in c for c in self.df.columns)

    def _find_first(self, candidates: List[str]) -> Optional[str]:
        """Return the first column from *candidates* that exists in df."""
        if self.df is None:
            return None
        for c in candidates:
            if c in self.df.columns:
                return c
        return None

    # ── Feature building methods ────────────────────────────
    def build_all_features(
        self,
        df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Run all feature engineering steps and return the enriched DataFrame.

        Parameters
        ----------
        df : pd.DataFrame, optional
            Override the DataFrame (if not using the one from __init__).

        Returns
        -------
        pd.DataFrame with all features added.
        """
        if df is not None:
            self.df = df.copy()
            self.report.input_shape = self.df.shape

        if self.df is None:
            raise ValueError("No DataFrame provided. Pass one to __init__() or load().")

        logger.info("build_all_features: starting with %d rows x %d cols", *self.df.shape)

        n_before = self.df.shape[1]

        # ── AOT price-based features ────────────────────────
        aot_close = _find_column(self.df, "aot", "close")
        aot_volume = _find_column(self.df, "aot", "volume")

        if aot_close:
            self.df = add_monthly_return(self.df, price_col=aot_close)
            self.df = add_log_return(self.df, price_col=aot_close)
            self.df = add_moving_average(
                self.df, col=aot_close, windows=self.rolling_windows, prefix="aot"
            )
            self.df = add_rolling_std(
                self.df, col=aot_close, windows=self.rolling_windows, prefix="aot"
            )
            self.df = add_lag_features(self.df, col=aot_close, lags=self.lags, prefix="aot")

        if aot_volume:
            self.df = add_lag_features(self.df, col=aot_volume, lags=self.lags, prefix="aot_volume")

        # ── SET Index features ──────────────────────────────
        set_close = _find_column(self.df, "set", "close")
        if set_close:
            self.df = add_moving_average(self.df, col=set_close, windows=[3, 6], prefix="set")
            self.df = add_lag_features(self.df, col=set_close, lags=[1, 3], prefix="set")

        # ── Tourism features ────────────────────────────────
        self.df = add_tourist_growth(self.df)
        tourists_total = _find_column(self.df, "tourists", "total_arrivals")
        if tourists_total:
            self.df = add_lag_features(
                self.df, col=tourists_total, lags=[1, 3, 6], prefix="tourists"
            )

        # ── Exchange rate features ──────────────────────────
        self.df = add_exchange_rate_change(self.df)
        fx_rate = _find_column(self.df, "fx", "usdthb_rate")
        if fx_rate:
            self.df = add_lag_features(self.df, col=fx_rate, lags=[1, 3, 6], prefix="fx")

        # ── Volume change (AOT) ─────────────────────────────
        self.df = add_volume_change(self.df)

        # ── Macro lag features ──────────────────────────────
        for prefix in ["policy", "cpi"]:
            for col in list(self.df.columns):
                if col.startswith(f"{prefix}_") and col.count("_") == 1:
                    if col not in self.df.select_dtypes(include=[np.number]).columns:
                        continue
                    self.df = add_lag_features(self.df, col=col, lags=[1, 3], prefix=prefix)

        # ── Drop rows with NaN (from lag creation at start of series) ──
        n_nan = self.df.isna().any(axis=1).sum()
        self.df = self.df.dropna()
        self.report.n_nan_rows = n_nan

        n_added = self.df.shape[1] - n_before
        self.report.features_added = [
            c
            for c in self.df.columns
            if any(ext in c for ext in ["return", "ma_", "rolling_std", "lag_", "growth", "change"])
        ]
        self.report.output_shape = self.df.shape

        logger.info(
            "build_all_features: %d features added -> %d rows x %d cols (dropped %d NaN rows)",
            n_added,
            self.df.shape[0],
            self.df.shape[1],
            n_nan,
        )

        return self.df

    # ── Export ──────────────────────────────────────────────
    def export(self, path: Union[str, Path] = "data/processed/feature_dataset.csv") -> Path:
        """Export the feature-enriched dataset to CSV.

        Parameters
        ----------
        path : str or Path, default "data/processed/feature_dataset.csv"
            Output path.

        Returns
        -------
        Path to the exported file.
        """
        if self.df is None:
            raise ValueError("No data to export. Run build_all_features() first.")

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        # Convert PeriodIndex to string for CSV
        df_out = self.df.copy()
        df_out.index = df_out.index.astype(str)
        df_out.to_csv(p, encoding="utf-8-sig")

        self.report.output_path = str(p)
        logger.info("export: feature dataset -> %s (%d rows, %d cols)", p, *self.df.shape)
        return p

    # ── Summary ─────────────────────────────────────────────
    def summary(self) -> str:
        """Return a human-readable summary of features created."""
        r = self.report
        lines = [
            "FeatureEngineering Summary",
            "===========================",
            f"Input shape     : {r.input_shape[0]} rows x {r.input_shape[1]} cols",
            f"Output shape    : {r.output_shape[0]} rows x {r.output_shape[1]} cols",
            f"Features added  : {len(r.features_added)}",
            f"NaN rows dropped: {r.n_nan_rows}",
            f"Output path     : {r.output_path}",
            "",
            "Feature groups:",
        ]

        # Group features by prefix
        if self.df is not None:
            groups = {}
            for c in self.df.columns:
                prefix = c.split("_")[0] if "_" in c else c
                groups.setdefault(prefix, []).append(c)
            for prefix in sorted(groups):
                cols = groups[prefix]
                lines.append(f"  {prefix}: {len(cols)} columns")

        return "\n".join(lines)

    def feature_catalog(self) -> pd.DataFrame:
        """Return a DataFrame mapping each feature column to its description.

        Returns
        -------
        pd.DataFrame with columns: feature, description, group, type.
        """
        if self.df is None:
            return pd.DataFrame()

        catalog = []
        for col in self.df.columns:
            desc = col  # Will be populated with human-readable description
            # Infer description from column name
            if "return" in col and "log" in col:
                desc = "Log return (ln(close_t / close_t-1))"
            elif "return" in col:
                desc = "Monthly return ((close_t - close_t-1) / close_t-1)"
            elif "ma_" in col:
                w = col.split("_")[-1]
                desc = f"Moving average (window={w} months)"
            elif "rolling_std" in col:
                w = col.split("_")[-1]
                desc = f"Rolling standard deviation (window={w} months)"
            elif "lag_" in col:
                parts = col.split("_lag_")
                lag = parts[-1] if len(parts) > 1 else "?"
                desc = f"Lagged value ({lag}-month shift)"
            elif "growth" in col:
                desc = "Month-over-month growth rate"
            elif "change" in col:
                desc = "Month-over-month percentage change"

            group = col.split("_")[0] if "_" in col else "other"
            dtype = str(self.df[col].dtype)

            catalog.append({"feature": col, "description": desc, "group": group, "dtype": dtype})

        return pd.DataFrame(catalog)


# ──────────────────────────────────────────────────────────────
# Convenience entry point
# ──────────────────────────────────────────────────────────────
__all__ = [
    "FeatureEngineer",
    "FeatureEngineeringReport",
    "add_monthly_return",
    "add_log_return",
    "add_moving_average",
    "add_rolling_std",
    "add_lag_features",
    "add_tourist_growth",
    "add_exchange_rate_change",
    "add_volume_change",
]
