"""
preprocessing.py — Data Cleaning, Aggregation, and Alignment
=============================================================

Loads processed datasets from the DataLoader output (data/processed/*.csv),
performs cleaning, outlier treatment, optional normalization, aggregates
daily/irregular series to monthly frequency, and aligns all series on a
common monthly time index.

Flow
----
  1. Load available CSVs from data/processed/
  2. For each dataset:
       a. Remove duplicate rows
       b. Parse date column and sort chronologically
       c. Handle missing values (forward-fill, then interpolate, then drop)
       d. Detect and winsorize outliers (IQR method)
       e. Optionally normalize (StandardScaler or MinMaxScaler)
       f. Aggregate daily/irregular series to monthly frequency
  3. Merge all monthly-aligned datasets on the time index (full outer join)
  4. Export the aligned dataset to data/processed/aligned_monthly.csv

Usage
-----
    from aot_stock_network.preprocessing import PreprocessingPipeline

    pp = PreprocessingPipeline(data_dir="data")
    df_aligned = pp.run()
    # df_aligned is a monthly DataFrame with all series merged
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────
DEFAULT_PROCESSED_DIR = "data/processed"
DEFAULT_OUTPUT_PATH = "data/processed/aligned_monthly.csv"
MAX_FFILL_PERIODS = 2
MAX_INTERPOLATE_PERIODS = 3
OUTLIER_IQR_MULTIPLIER = 1.5
MIN_ROWS_FOR_OUTLIER_DETECTION = 10

# Predefined column aggregation rules for daily -> monthly resampling.
# Key   = source name (as used in DataLoader)
# Value = dict mapping column -> aggregation function
AGGREGATION_RULES: Dict[str, Dict[str, str]] = {
    "set_aot": {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "value": "sum",
    },
    "set_index": {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "value": "sum",
    },
    "bot_usdthb": {
        "usdthb_rate": "mean",
        "bid_rate": "mean",
        "ask_rate": "mean",
    },
}

# Sources that should be resampled with forward-fill (irregular frequency)
FFILL_SOURCES = {"bot_policy_rate"}

# Sources that are already monthly and need no aggregation
MONTHLY_SOURCES = {
    "mots_tourists",
    "mots_revenue",
    "bot_inflation",
}

# Column prefix mapping for merged dataset
COLUMN_PREFIX: Dict[str, str] = {
    "set_aot": "aot",
    "set_index": "set",
    "mots_tourists": "tourists",
    "mots_revenue": "revenue",
    "bot_usdthb": "fx",
    "bot_policy_rate": "policy",
    "bot_inflation": "cpi",
}


# ──────────────────────────────────────────────────────────────
# Utility helpers
# ──────────────────────────────────────────────────────────────
def _setup_logging(level: str = "INFO") -> None:
    """Configure logging for this module."""
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | preprocessing | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        logger.addHandler(handler)


def _infer_date_column(df: pd.DataFrame) -> Optional[str]:
    """Return the name of the date column in a DataFrame.

    Checks common date-column names ('date', 'month', 'announce_date',
    'effective_date') and returns the first match.
    """
    candidates = ["month", "date", "announce_date", "effective_date"]
    for col in candidates:
        if col in df.columns:
            return col
    for col in df.columns:
        if "date" in col.lower() or "month" in col.lower():
            return col
    return None


def _parse_dates(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Parse the date column to datetime and set as index."""
    if date_col not in df.columns:
        logger.warning("Date column '%s' not found in DataFrame", date_col)
        return df

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df.dropna(subset=[date_col], inplace=True)
    df.sort_values(date_col, inplace=True)
    df.set_index(date_col, inplace=True)
    return df


# ──────────────────────────────────────────────────────────────
# 1. Remove duplicates
# ──────────────────────────────────────────────────────────────
def remove_duplicates(
    df: pd.DataFrame,
    subset: Optional[List[str]] = None,
    keep: str = "last",
) -> pd.DataFrame:
    """Remove duplicate rows from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    subset : list of str, optional
        Columns to consider for duplicate detection. If None, use all columns.
    keep : str, default "last"
        Which duplicate to keep ('first', 'last', False).

    Returns
    -------
    pd.DataFrame with duplicates removed.

    Transformation documented
    -------------------------
    Removes rows where all columns in *subset* are identical. Keeps the
    last occurrence (most recent for time-series data).
    """
    n_before = len(df)
    df = df.drop_duplicates(subset=subset, keep=keep)
    n_removed = n_before - len(df)
    if n_removed > 0:
        logger.info(
            "remove_duplicates: removed %d rows (%.1f%%)", n_removed, 100 * n_removed / n_before
        )
    return df


# ──────────────────────────────────────────────────────────────
# 2. Handle missing values
# ──────────────────────────────────────────────────────────────
def handle_missing_values(
    df: pd.DataFrame,
    max_ffill: int = MAX_FFILL_PERIODS,
    max_interpolate: int = MAX_INTERPOLATE_PERIODS,
    drop_remaining: bool = True,
) -> pd.DataFrame:
    """Handle missing values in a time-series DataFrame.

    Strategy (applied in order):
      1. Forward-fill (carry last observation forward) for up to *max_ffill*
         consecutive missing periods.
      2. Linear interpolation for up to *max_interpolate* consecutive missing
         periods.
      3. Optionally drop rows that still contain NaN values.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with DatetimeIndex.
    max_ffill : int, default 2
        Maximum number of consecutive NaN to forward-fill.
    max_interpolate : int, default 3
        Maximum number of consecutive NaN to linearly interpolate.
    drop_remaining : bool, default True
        If True, drop rows where any column remains NaN after filling.

    Returns
    -------
    pd.DataFrame with missing values handled.

    Transformation documented
    -------------------------
    Missing values are filled in two stages:
      1. Forward-fill limited to 2 consecutive NaN periods
         (assumes short gaps can be carried forward).
      2. Linear interpolation limited to 3 consecutive NaN periods
         (assumes medium gaps can be interpolated).
      3. Any remaining NaN rows are dropped (long gaps).

    For monthly data, max_ffill=2 means up to 2 months of missing data
    can be filled. For daily data, this means up to 2 days.
    """
    df = df.copy()
    n_before = len(df)

    # Forward-fill
    df = df.ffill(limit=max_ffill)

    # Linear interpolation (time-based)
    if max_interpolate > 0:
        df = df.interpolate(method="linear", limit=max_interpolate)

    # Drop remaining NaN
    if drop_remaining:
        df = df.dropna()
        n_dropped = n_before - len(df)
        if n_dropped > 0:
            logger.info(
                "handle_missing: dropped %d rows with remaining NaN (%.1f%%)",
                n_dropped,
                100 * n_dropped / n_before,
            )

    return df


# ──────────────────────────────────────────────────────────────
# 3. Detect and handle outliers
# ──────────────────────────────────────────────────────────────
def detect_outliers_iqr(
    series: pd.Series,
    multiplier: float = OUTLIER_IQR_MULTIPLIER,
) -> pd.Series:
    """Detect outliers using the Interquartile Range (IQR) method.

    Parameters
    ----------
    series : pd.Series
        Numeric data.
    multiplier : float, default 1.5
        IQR multiplier (standard Tukey: 1.5 for outliers, 3 for extreme).

    Returns
    -------
    Boolean Series where True indicates an outlier.

    Transformation documented
    -------------------------
    Outliers defined by Tukey's fences:
      Lower = Q1 - multiplier * IQR
      Upper = Q3 + multiplier * IQR
    Standard multiplier 1.5 captures ~99.3% of normal data for Gaussian.
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    return (series < lower) | (series > upper)


def winsorize_series(
    series: pd.Series,
    lower_percentile: float = 0.01,
    upper_percentile: float = 0.99,
) -> pd.Series:
    """Winsorize (clip) a series at specified percentiles.

    Parameters
    ----------
    series : pd.Series
        Numeric data.
    lower_percentile : float, default 0.01
        Lower percentile bound (e.g., 0.01 = 1st percentile).
    upper_percentile : float, default 0.99
        Upper percentile bound (e.g., 0.99 = 99th percentile).

    Returns
    -------
    Winsorized series with extreme values clipped.

    Transformation documented
    -------------------------
    Extreme values are clipped to the specified percentile boundaries.
    This retains all data points while limiting the influence of
    extreme values. Default: clip at 1st and 99th percentiles.
    """
    lower = series.quantile(lower_percentile)
    upper = series.quantile(upper_percentile)
    return series.clip(lower, upper)


def handle_outliers(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = "winsorize",
    iqr_multiplier: float = OUTLIER_IQR_MULTIPLIER,
    lower_percentile: float = 0.01,
    upper_percentile: float = 0.99,
    drop: bool = False,
) -> pd.DataFrame:
    """Detect and handle outliers in specified columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    columns : list of str, optional
        Columns to process. If None, process all numeric columns.
    method : str, default "winsorize"
        Treatment method: "winsorize" (clip) or "cap" (clip at IQR fences).
    iqr_multiplier : float, default 1.5
        IQR multiplier for "cap" method.
    lower_percentile : float, default 0.01
        Lower bound for winsorize.
    upper_percentile : float, default 0.99
        Upper bound for winsorize.
    drop : bool, default False
        If True, drop outlier rows instead of clipping.

    Returns
    -------
    pd.DataFrame with outliers handled.

    Transformation documented
    -------------------------
    Outlier treatment applied column-by-column:
      - 'winsorize': clip extreme values at percentile thresholds
      - 'cap': clip at Tukey IQR fences (Q1 - 1.5*IQR, Q3 + 1.5*IQR)
    """
    df = df.copy()
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    n_total_outliers = 0
    for col in columns:
        if col not in df.columns or df[col].nunique() < MIN_ROWS_FOR_OUTLIER_DETECTION:
            continue

        if method == "winsorize":
            df[col] = winsorize_series(
                df[col],
                lower_percentile=lower_percentile,
                upper_percentile=upper_percentile,
            )
        elif method == "cap":
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - iqr_multiplier * iqr
            upper = q3 + iqr_multiplier * iqr
            outliers = (df[col] < lower) | (df[col] > upper)
            n_total_outliers += outliers.sum()
            if drop:
                df = df[~outliers]
            else:
                df[col] = df[col].clip(lower, upper)

    if n_total_outliers > 0:
        logger.info(
            "handle_outliers: treated %d outliers across %d columns (method=%s)",
            n_total_outliers,
            len(columns),
            method,
        )

    return df


# ──────────────────────────────────────────────────────────────
# 4. Normalize data
# ──────────────────────────────────────────────────────────────
def normalize_data(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = "standard",
    scaler_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Normalize (scale) numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    columns : list of str, optional
        Columns to scale. If None, scale all numeric columns.
    method : str, default "standard"
        Scaling method:
          - "standard": StandardScaler (z-score, mean=0, std=1)
          - "minmax": MinMaxScaler (range [0, 1])
          - "robust": RobustScaler (median & IQR based)
    scaler_path : str or Path, optional
        If provided, save the fitted scaler via joblib for later inverse
        transformation.

    Returns
    -------
    pd.DataFrame with scaled columns.

    Transformation documented
    -------------------------
    Normalization rescales features to a common range or distribution:
      - 'standard':  x_scaled = (x - mean) / std
      - 'minmax':     x_scaled = (x - min) / (max - min)
      - 'robust':     x_scaled = (x - median) / IQR

    NOTE: For ML projects, normalization should be fit on training data
    and transformed on test data to avoid data leakage. The scaler is
    optionally persisted for this purpose.
    """
    df = df.copy()
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    existing_cols = [c for c in columns if c in df.columns and df[c].nunique() > 1]
    if not existing_cols:
        logger.warning("normalize: no columns with variance found")
        return df

    from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

    scaler_map = {
        "standard": StandardScaler,
        "minmax": MinMaxScaler,
        "robust": RobustScaler,
    }

    ScalerClass = scaler_map.get(method)
    if ScalerClass is None:
        raise ValueError(f"Unknown normalization method '{method}'. Use: {list(scaler_map)}")

    scaler = ScalerClass()
    scaled_values = scaler.fit_transform(df[existing_cols])
    df[existing_cols] = scaled_values

    if scaler_path:
        import joblib

        p = Path(scaler_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, p)
        logger.info("normalize: scaler saved to %s", p)

    logger.info("normalize: scaled %d columns via '%s'", len(existing_cols), method)
    return df


# ──────────────────────────────────────────────────────────────
# 5. Aggregate daily data to monthly
# ──────────────────────────────────────────────────────────────
def aggregate_daily_to_monthly(
    df: pd.DataFrame,
    source_name: str,
    agg_rules: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Aggregate daily time-series data to monthly frequency.

    Parameters
    ----------
    df : pd.DataFrame
        Daily data with DatetimeIndex.
    source_name : str
        Name of the data source (used to look up aggregation rules).
    agg_rules : dict, optional
        Custom aggregation rules mapping column -> function.
        If None, uses predefined AGGREGATION_RULES for the source.

    Returns
    -------
    pd.DataFrame with monthly frequency.

    Transformation documented
    -------------------------
    Daily -> Monthly aggregation uses the following rules per column type:
      - 'open' / 'first' : First observation of the month
      - 'high' / 'max'   : Maximum value in the month
      - 'low' / 'min'    : Minimum value in the month
      - 'close' / 'last' : Last observation of the month
      - 'volume' / 'sum' : Sum over the month
      - 'value' / 'sum'  : Sum over the month
      - 'mean'           : Mean over the month

    The resulting index is a PeriodIndex with monthly frequency.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        logger.warning("aggregate_daily: index is not DatetimeIndex for '%s'", source_name)
        return df

    if agg_rules is None:
        agg_rules = AGGREGATION_RULES.get(source_name, {})

    if not agg_rules:
        available = list(df.select_dtypes(include=[np.number]).columns)
        logger.warning(
            "aggregate_daily: no rules for '%s', using 'last' for all numeric columns",
            source_name,
        )
        agg_rules = {col: "last" for col in available}

    # Filter to only columns present in the DataFrame
    agg_rules = {col: func for col, func in agg_rules.items() if col in df.columns}

    if not agg_rules:
        logger.warning("aggregate_daily: no columns match rules for '%s'", source_name)
        return df

    monthly = df[list(agg_rules.keys())].resample("ME").agg(agg_rules)

    # Convert to PeriodIndex for consistent monthly alignment
    monthly.index = monthly.index.to_period("M")

    # Prefix columns with source name for merge clarity
    prefix = COLUMN_PREFIX.get(source_name, source_name)
    monthly.columns = [f"{prefix}_{col}" for col in monthly.columns]

    return monthly


def aggregate_irregular_to_monthly(
    df: pd.DataFrame,
    source_name: str,
) -> pd.DataFrame:
    """Resample irregular time-series data to monthly using forward-fill.

    Parameters
    ----------
    df : pd.DataFrame
        Irregular data with DatetimeIndex (e.g., policy rate announcements).
    source_name : str
        Name of the data source.

    Returns
    -------
    pd.DataFrame with monthly frequency.

    Transformation documented
    -------------------------
    Irregular -> Monthly: the most recent observed value before the end of
    each month is carried forward. This captures the "as of" value for
    non-regularly updated series (e.g., policy rate, which only changes
    at MPC meetings).
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        logger.warning("aggregate_irregular: index is not DatetimeIndex for '%s'", source_name)
        return df

    # Keep only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        logger.warning("aggregate_irregular: no numeric columns for '%s'", source_name)
        return df

    # Resample to daily, forward-fill, then take last day of month
    daily = df[numeric_cols].resample("D").ffill()
    monthly = daily.resample("ME").last()

    # Convert to PeriodIndex
    monthly.index = monthly.index.to_period("M")

    prefix = COLUMN_PREFIX.get(source_name, source_name)
    monthly.columns = [f"{prefix}_{col}" for col in monthly.columns]

    return monthly


# ──────────────────────────────────────────────────────────────
# 6. Prepare monthly datasets (already monthly sources)
# ──────────────────────────────────────────────────────────────
def prepare_monthly_source(
    df: pd.DataFrame,
    source_name: str,
) -> pd.DataFrame:
    """Standardize an already-monthly dataset for alignment.

    Ensures the index is a monthly PeriodIndex and columns are prefixed.

    Parameters
    ----------
    df : pd.DataFrame
        Monthly data (index is date-based or has a date column).
    source_name : str
        Name of the data source.

    Returns
    -------
    pd.DataFrame with PeriodIndex (monthly).
    """
    df = df.copy()

    # If date is a column, convert to index
    date_col = _infer_date_column(df)
    if date_col and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df.dropna(subset=[date_col], inplace=True)
        df.set_index(date_col, inplace=True)

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
        df.dropna(inplace=True)

    # Convert to monthly PeriodIndex
    df.index = df.index.to_period("M")

    # Drop duplicate periods
    df = df[~df.index.duplicated(keep="last")]

    # Prefix columns
    prefix = COLUMN_PREFIX.get(source_name, source_name)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    col_map = {c: f"{prefix}_{c}" for c in numeric_cols}
    df.rename(columns=col_map, inplace=True)

    return df[numeric_cols] if not numeric_cols else df[[f"{prefix}_{c}" for c in numeric_cols]]


# ──────────────────────────────────────────────────────────────
# 7. Merge all aligned datasets
# ──────────────────────────────────────────────────────────────
def merge_monthly_datasets(
    datasets: Dict[str, pd.DataFrame],
    how: str = "outer",
) -> pd.DataFrame:
    """Merge multiple monthly-aligned DataFrames on their PeriodIndex.

    Parameters
    ----------
    datasets : dict of str -> pd.DataFrame
        Dictionary mapping source name to monthly DataFrame.
        Each DataFrame should have a PeriodIndex with monthly frequency.
    how : str, default "outer"
        Merge method ('outer', 'inner', 'left', 'right').

    Returns
    -------
    pd.DataFrame with all columns merged on the time index.

    Transformation documented
    -------------------------
    All monthly datasets are merged on their PeriodIndex using a full outer
    join by default. This preserves all time points from all sources.
    Gaps are then forward-filled (up to 2 months) to handle sources that
    start at different dates or have missing months.
    """
    valid = {name: df for name, df in datasets.items() if df is not None and not df.empty}

    if not valid:
        logger.error("merge: no valid datasets to merge")
        return pd.DataFrame()

    merged = None
    for name, df in valid.items():
        if merged is None:
            merged = df
        else:
            merged = merged.join(df, how=how)

    if merged is not None and not merged.empty:
        merged.sort_index(inplace=True)
        # Forward-fill short gaps (max 2 months) after merge
        merged = merged.ffill(limit=2)
        merged.index.name = "month"

    logger.info(
        "merge: combined %d datasets -> %d rows x %d cols",
        len(valid),
        len(merged) if merged is not None else 0,
        len(merged.columns) if merged is not None else 0,
    )

    return merged


# ──────────────────────────────────────────────────────────────
# 8. Full preprocessing pipeline
# ──────────────────────────────────────────────────────────────
@dataclass
class PreprocessingReport:
    """Summary report of what the preprocessing pipeline did."""

    sources_loaded: List[str] = field(default_factory=list)
    sources_aggregated: List[str] = field(default_factory=list)
    final_shape: Tuple[int, int] = (0, 0)
    n_rows_dropped: int = 0
    n_outliers_treated: int = 0
    output_path: str = ""


class PreprocessingPipeline:
    """End-to-end preprocessing orchestrator.

    Loads processed CSVs, cleans each one, aggregates daily/irregular
    sources to monthly, merges everything into a single aligned DataFrame,
    and exports the result.

    Parameters
    ----------
    data_dir : str or Path, default "data"
        Root data directory containing 'processed/' subdirectory.
    output_path : str or Path, optional
        Where to write the aligned monthly CSV.
    normalize : bool, default False
        If True, apply StandardScaler after cleaning (per-column).
    outlier_method : str, default "winsorize"
            Method for outlier treatment ('winsorize' or 'cap').
    log_level : str, default "INFO"
        Logging verbosity.
    """

    def __init__(
        self,
        data_dir: Union[str, Path] = "data",
        output_path: Optional[Union[str, Path]] = None,
        normalize: bool = False,
        outlier_method: str = "winsorize",
        log_level: str = "INFO",
    ):
        self.data_dir = Path(data_dir).resolve()
        self.processed_dir = self.data_dir / "processed"
        self.output_path = (
            Path(output_path) if output_path else self.processed_dir / "aligned_monthly.csv"
        )
        self.normalize = normalize
        self.outlier_method = outlier_method

        _setup_logging(log_level)
        self.report = PreprocessingReport()

        logger.info(
            "PreprocessingPipeline | data_dir=%s | normalize=%s | outlier=%s",
            self.data_dir,
            normalize,
            outlier_method,
        )

    # ── Source loading ──────────────────────────────────────
    def _available_sources(self) -> List[str]:
        """Return list of source names that have processed CSVs available."""
        csv_files = sorted(self.processed_dir.glob("*.csv"))
        # Exclude the aligned output itself
        sources = []
        for f in csv_files:
            stem = f.stem
            if stem != "aligned_monthly" and not stem.startswith("feature_"):
                sources.append(stem)
        return sources

    def _load_source(self, source_name: str) -> Optional[pd.DataFrame]:
        """Load a single processed CSV into a DataFrame."""
        csv_path = self.processed_dir / f"{source_name}.csv"
        if not csv_path.exists():
            logger.warning("load: CSV not found for '%s' at %s", source_name, csv_path)
            return None

        try:
            df = pd.read_csv(csv_path)
            logger.info(
                "load: loaded '%s' (%d rows, %d cols)", source_name, len(df), len(df.columns)
            )
            return df
        except Exception as exc:
            logger.error("load: failed to load '%s': %s", source_name, exc)
            return None

    # ── Single-source cleaning ──────────────────────────────
    def _clean_source(self, df: pd.DataFrame, source_name: str) -> Optional[pd.DataFrame]:
        """Apply all cleaning steps to a single source's DataFrame."""
        if df is None or df.empty:
            return None

        n0 = len(df)

        # 1. Remove duplicates
        df = remove_duplicates(df)

        # 2. Identify and parse date column
        date_col = _infer_date_column(df)
        if date_col is None:
            logger.warning("clean: no date column found for '%s', skipping", source_name)
            return None
        df = _parse_dates(df, date_col)

        # 3. Handle missing values
        df = handle_missing_values(df)

        # 4. Handle outliers (numeric columns only)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            df = handle_outliers(df, columns=numeric_cols, method=self.outlier_method)

        # 5. Optional normalization
        if self.normalize and numeric_cols:
            df = normalize_data(df, columns=numeric_cols)

        n_removed = n0 - len(df)
        self.report.n_rows_dropped += n_removed

        logger.info(
            "clean: '%s' -> %d rows (dropped %d)",
            source_name,
            len(df),
            n_removed,
        )
        return df

    # ── Frequency alignment ─────────────────────────────────
    def _align_to_monthly(self, df: pd.DataFrame, source_name: str) -> Optional[pd.DataFrame]:
        """Convert a cleaned DataFrame to monthly frequency based on source type."""
        if df is None or df.empty:
            return None

        if source_name in AGGREGATION_RULES:
            result = aggregate_daily_to_monthly(df, source_name)
            self.report.sources_aggregated.append(source_name)
        elif source_name in FFILL_SOURCES:
            result = aggregate_irregular_to_monthly(df, source_name)
            self.report.sources_aggregated.append(source_name)
        elif source_name in MONTHLY_SOURCES:
            result = prepare_monthly_source(df, source_name)
        else:
            logger.warning("align: unknown frequency for '%s', treating as monthly", source_name)
            result = prepare_monthly_source(df, source_name)

        return result

    # ── Export ──────────────────────────────────────────────
    def _export(self, df: pd.DataFrame) -> Path:
        """Write the aligned monthly dataset to CSV."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.output_path, encoding="utf-8-sig")
        logger.info(
            "export: aligned dataset written to %s (%d rows, %d cols)", self.output_path, *df.shape
        )
        return self.output_path

    # ── Run ─────────────────────────────────────────────────
    def run(
        self,
        source_names: Optional[List[str]] = None,
        dataframes: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> pd.DataFrame:
        """Execute the full preprocessing pipeline.

        Parameters
        ----------
        source_names : list of str, optional
            Specific sources to process. If None, process all available CSVs.
        dataframes : dict of str -> pd.DataFrame, optional
            Pre-loaded DataFrames to use instead of loading from CSV.
            Keys should match source names. If provided, these override
            CSV loading for the specified sources.

        Returns
        -------
        pd.DataFrame
            Aligned monthly dataset with all sources merged.
        """
        all_sources: Dict[str, pd.DataFrame] = {}

        # Determine which sources to process
        if dataframes:
            all_sources.update(dataframes)

        csv_sources = source_names or self._available_sources()
        missing_from_csv = [s for s in csv_sources if s not in all_sources]

        # Load from CSV for any not provided as dataframes
        for name in missing_from_csv:
            all_sources[name] = self._load_source(name)

        if not all_sources:
            logger.error("run: no data sources available. Fetch data first with DataLoader.")
            return pd.DataFrame()

        self.report.sources_loaded = list(all_sources.keys())
        logger.info("run: processing %d sources: %s", len(all_sources), list(all_sources.keys()))

        # Clean and align each source
        monthly_datasets: Dict[str, pd.DataFrame] = {}
        for name, df in all_sources.items():
            if df is None:
                continue
            cleaned = self._clean_source(df, name)
            if cleaned is None:
                continue
            monthly = self._align_to_monthly(cleaned, name)
            if monthly is not None and not monthly.empty:
                monthly_datasets[name] = monthly

        if not monthly_datasets:
            logger.error("run: no monthly datasets produced")
            return pd.DataFrame()

        # Merge all
        aligned = merge_monthly_datasets(monthly_datasets)

        if aligned.empty:
            logger.error("run: merged dataset is empty")
            return aligned

        # Export
        self._export(aligned)

        # Update report
        self.report.final_shape = aligned.shape
        self.report.output_path = str(self.output_path)

        logger.info(
            "run: pipeline complete -> %d rows x %d columns -> %s",
            aligned.shape[0],
            aligned.shape[1],
            self.output_path,
        )

        return aligned

    def summary(self) -> str:
        """Return a human-readable summary of the pipeline run."""
        r = self.report
        lines = [
            "PreprocessingPipeline Summary",
            "==============================",
            f"Sources loaded  : {len(r.sources_loaded)}",
            f"Sources agg'd   : {len(r.sources_aggregated)}",
            f"Final shape     : {r.final_shape[0]} rows x {r.final_shape[1]} cols",
            f"Rows dropped    : {r.n_rows_dropped}",
            f"Output          : {r.output_path}",
        ]
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# Convenience entry point
# ──────────────────────────────────────────────────────────────
__all__ = [
    "PreprocessingPipeline",
    "PreprocessingReport",
    "remove_duplicates",
    "handle_missing_values",
    "detect_outliers_iqr",
    "winsorize_series",
    "handle_outliers",
    "normalize_data",
    "aggregate_daily_to_monthly",
    "aggregate_irregular_to_monthly",
    "prepare_monthly_source",
    "merge_monthly_datasets",
]
