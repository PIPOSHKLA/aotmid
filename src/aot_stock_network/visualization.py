"""
visualization.py — Publication-Quality EDA Figures for AOT Stock Network
=========================================================================

Generates all exploratory data analysis plots required for the project:
  — Distribution plots          (histogram + KDE, density)
  — Time series                 (full-range line charts)
  — Monthly trends              (seasonal subseries by calendar month)
  — Yearly trends               (year-over-year comparison)
  — Scatter plots               (pairwise relationships, colour-coded)
  — Correlation heatmap         (full-feature annotated matrix)
  — Feature correlation matrix  (target-centric correlation vector)
  — Pair plot                   (multi-panel scatter + distribution)
  — Outlier detection           (box plots, IQR-highlighted time series)
  — Trend decomposition         (additive seasonal decomposition)

Every figure is a matplotlib Figure object that can be:
  • displayed in Streamlit via st.pyplot(fig)
  • exported as SVG via fig.savefig("plot.svg")
  • composed into multi-panel layouts

Usage
-----
    from aot_stock_network.visualization import EDAVisualizer

    viz = EDAVisualizer()
    viz.load("data/processed/feature_dataset.csv")

    # Generate everything to a folder
    viz.generate_all(output_dir="reports/figures")

    # Or build custom multi-panel figures
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    viz.plot_distribution("aot_return", ax=axes[0, 0])
    viz.plot_time_series("aot_close", ax=axes[0, 1])
    viz.plot_monthly_trend("aot_close", ax=axes[1, 0])
    viz.plot_scatter("tourists_total_arrivals", "aot_close", ax=axes[1, 1])
    viz.export(fig, "reports/figures/custom_dashboard.svg")
"""

from __future__ import annotations

import logging
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.dates as mdates

# matplotlib / seaborn
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.offsetbox import AnchoredText

# statsmodels for decomposition
from statsmodels.tsa.seasonal import seasonal_decompose

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Suppress FutureWarning noise
# ──────────────────────────────────────────────────────────────
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="seaborn")

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────
DEFAULT_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]

PUBLICATION_COLORS = {
    "primary": "#1f77b4",
    "secondary": "#ff7f0e",
    "target": "#d62728",
    "positive": "#2ca02c",
    "negative": "#d62728",
    "grid": "#e0e0e0",
    "text": "#333333",
    "fill": "#1f77b4",
    "fill_opacity": 0.3,
}


@dataclass
class VisualizationConfig:
    """Styling and layout configuration for all plots."""

    style: str = "whitegrid"
    palette: str = "husl"
    figsize: Tuple[float, float] = (12, 5.5)
    dpi: int = 150
    font_family: str = "DejaVu Sans"
    title_size: int = 15
    label_size: int = 13
    tick_size: int = 11
    legend_size: int = 10
    annotation_size: int = 9
    colors: List[str] = field(default_factory=lambda: DEFAULT_COLORS.copy())
    save_format: str = "svg"
    tight_layout_pad: float = 1.5
    rgba_alpha: float = 0.7

    def apply(self) -> None:
        """Apply this configuration to matplotlib and seaborn."""
        plt.rcParams.update(
            {
                "font.family": self.font_family,
                "font.size": self.label_size,
                "axes.titlesize": self.title_size,
                "axes.labelsize": self.label_size,
                "xtick.labelsize": self.tick_size,
                "ytick.labelsize": self.tick_size,
                "legend.fontsize": self.legend_size,
                "figure.dpi": self.dpi,
                "savefig.dpi": 300,
                "savefig.format": self.save_format,
                "savefig.bbox": "tight",
                "savefig.pad_inches": 0.15,
                "axes.edgecolor": "#cccccc",
                "axes.grid": True,
                "grid.alpha": 0.3,
                "grid.linestyle": "--",
                "axes.spines.top": False,
                "axes.spines.right": False,
            }
        )
        sns.set_style(self.style)
        sns.set_palette(self.palette)


# ──────────────────────────────────────────────────────────────
# Utility helpers
# ──────────────────────────────────────────────────────────────
def _setup_logging(level: str = "INFO") -> None:
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | viz | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        logger.addHandler(handler)


def _describe_column(df: pd.DataFrame, col: str) -> str:
    """Short description for a column based on its name."""
    desc = col
    if "return" in col:
        desc = "Return"
    elif "ma_" in col:
        desc = f"MA-{col.split('_')[-1]}m"
    elif "rolling_std" in col:
        desc = f"Rolling Std-{col.split('_')[-1]}m"
    elif "lag_" in col:
        desc = col
    elif "growth" in col:
        desc = "Growth Rate"
    elif "change" in col:
        desc = "Change Rate"
    return desc


def _select_numeric(df: pd.DataFrame, max_cols: int = 30) -> List[str]:
    """Return numeric columns, dropping near-constant and id-like columns."""
    cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cols = [c for c in cols if df[c].nunique() > 5]
    return cols[:max_cols]


def _select_key_features(df: pd.DataFrame) -> List[str]:
    """Return a curated subset of features for pair plots and focused analysis."""
    candidates = [
        "aot_close",
        "aot_return",
        "aot_log_return",
        "aot_ma_3",
        "aot_rolling_std_3",
        "aot_volume",
        "set_close",
        "tourists_total_arrivals",
        "tourist_growth",
        "fx_usdthb_rate",
        "fx_change",
        "policy_policy_rate",
        "cpi_cpi_headline",
        "volume_change",
    ]
    return [c for c in candidates if c in df.columns]


def _darken_color(hex_color: str, factor: float = 0.7) -> str:
    """Darken a hex color by a factor (0=black, 1=unchanged)."""
    import matplotlib.colors as mcolors

    rgb = mcolors.hex2color(hex_color)
    darkened = tuple(c * factor for c in rgb)
    return mcolors.to_hex(darkened)


# ──────────────────────────────────────────────────────────────
# Exporter
# ──────────────────────────────────────────────────────────────
def export_figure(
    fig: Figure,
    path: Union[str, Path],
    fmt: str = "svg",
    dpi: int = 300,
) -> Path:
    """Export a matplotlib Figure to a file.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to export.
    path : str or Path
        Output path (extension is added based on fmt).
    fmt : str, default "svg"
        File format: 'svg', 'png', 'pdf', 'eps'.
    dpi : int, default 300
        Resolution for raster formats.

    Returns
    -------
    Path to the exported file.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # Ensure extension matches format
    if not p.suffix or p.suffix != f".{fmt}":
        p = p.with_suffix(f".{fmt}")
    fig.savefig(p, format=fmt, dpi=dpi, bbox_inches="tight", pad_inches=0.15)
    logger.info("Exported: %s (%s, %dx%d)", p.name, fmt, fig.get_figwidth(), fig.get_figheight())
    return p


# ──────────────────────────────────────────────────────────────
# 1. Distribution Plots
# ──────────────────────────────────────────────────────────────
def plot_distribution(
    df: pd.DataFrame,
    column: str,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    color: str = PUBLICATION_COLORS["primary"],
    show_stats: bool = True,
) -> Tuple[Figure, plt.Axes]:
    """Plot histogram with KDE overlay for a single column.

    Parameters
    ----------
    df : pd.DataFrame
        Data.
    column : str
        Column name to plot.
    ax : plt.Axes, optional
        Existing axes. If None, creates a new figure.
    title : str, optional
        Plot title. Auto-generated if None.
    color : str, default "#1f77b4"
        Bar and KDE color.
    show_stats : bool, default True
        Annotate mean, median, std on the plot.

    Returns
    -------
    (fig, ax) tuple.
    """
    if column not in df.columns:
        logger.warning("Column '%s' not found in DataFrame", column)
        return _empty_figure(f"Column '{column}' not found")

    data = df[column].dropna()
    if len(data) < 5:
        return _empty_figure(f"Insufficient data for '{column}'")

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.figure

    # Histogram + KDE
    sns.histplot(
        data,
        kde=True,
        bins=30,
        color=color,
        edgecolor="white",
        linewidth=0.5,
        alpha=0.65,
        ax=ax,
    )

    # Vertical lines for mean / median
    mean_val = data.mean()
    median_val = data.median()
    ax.axvline(
        mean_val,
        color=_darken_color(color),
        linestyle="--",
        linewidth=1.5,
        label=f"Mean={mean_val:.4f}",
    )
    ax.axvline(
        median_val, color="gray", linestyle=":", linewidth=1.5, label=f"Median={median_val:.4f}"
    )

    if show_stats:
        stats_text = (
            f"N={len(data)}\n"
            f"Mean={mean_val:.4f}\n"
            f"Std={data.std():.4f}\n"
            f"Skew={data.skew():.3f}\n"
            f"Kurt={data.kurtosis():.3f}"
        )
        at = AnchoredText(stats_text, loc="upper right", frameon=True, prop={"size": 8})
        at.patch.set_boxstyle("round,pad=0.2")
        at.patch.set_facecolor("white")
        at.patch.set_alpha(0.85)
        ax.add_artist(at)

    ax.set_title(title or f"Distribution of {column}", fontweight="bold", pad=12)
    ax.set_xlabel(_describe_column(df, column))
    ax.set_ylabel("Frequency")
    ax.legend(loc="upper left", frameon=True, fontsize=8)

    return fig, ax


def plot_distribution_grid(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    cols_per_row: int = 3,
    figsize: Tuple[float, float] = (15, 12),
) -> Tuple[Figure, List[plt.Axes]]:
    """Multi-panel distribution grid for multiple columns."""
    if columns is None:
        columns = _select_key_features(df)[:9]
    n = len(columns)
    n_rows = (n + cols_per_row - 1) // cols_per_row
    fig, axes = plt.subplots(n_rows, cols_per_row, figsize=figsize)
    axes_flat = axes.flatten() if n > 1 else [axes]

    for i, col in enumerate(columns):
        plot_distribution(df, col, ax=axes_flat[i])

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Feature Distributions", fontweight="bold", fontsize=16, y=1.01)
    fig.tight_layout(pad=2.0)
    return fig, axes_flat


# ──────────────────────────────────────────────────────────────
# 2. Time Series Plots
# ──────────────────────────────────────────────────────────────
def plot_time_series(
    df: pd.DataFrame,
    column: str,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    color: str = PUBLICATION_COLORS["primary"],
    highlight_periods: Optional[List[Tuple[str, str]]] = None,
    ylabel: Optional[str] = None,
) -> Tuple[Figure, plt.Axes]:
    """Plot a time series with optional crisis period highlighting.

    Parameters
    ----------
    df : pd.DataFrame
        Data with PeriodIndex or DatetimeIndex.
    column : str
        Column name to plot.
    ax : plt.Axes, optional
        Existing axes.
    title : str, optional
        Plot title.
    color : str, default "#1f77b4"
        Line color.
    highlight_periods : list of (start, end) str, optional
        Periods to highlight as shaded regions (e.g., COVID-19).
    ylabel : str, optional
        Y-axis label.

    Returns
    -------
    (fig, ax) tuple.
    """
    if column not in df.columns:
        return _empty_figure(f"Column '{column}' not found")

    data = df[column].dropna()
    if len(data) < 3:
        return _empty_figure(f"Insufficient data for '{column}'")

    idx = _get_index(data.to_frame())
    if idx is None:
        return _empty_figure("No time index found")

    if ax is None:
        fig, ax = plt.subplots(figsize=VisualizationConfig().figsize)
    else:
        fig = ax.figure

    ax.plot(idx, data.values, color=color, linewidth=1.8, alpha=0.9)

    # Highlight periods (e.g., COVID-19)
    if highlight_periods:
        for start, end in highlight_periods:
            start_ts = pd.Timestamp(start)
            end_ts = pd.Timestamp(end)
            ax.axvspan(start_ts, end_ts, alpha=0.12, color=PUBLICATION_COLORS["negative"], zorder=0)
            ax.axvline(
                start_ts,
                color=PUBLICATION_COLORS["negative"],
                linestyle=":",
                linewidth=0.8,
                alpha=0.5,
            )
            ax.axvline(
                end_ts,
                color=PUBLICATION_COLORS["negative"],
                linestyle=":",
                linewidth=0.8,
                alpha=0.5,
            )

    # Fill under the curve
    ax.fill_between(idx, data.values, alpha=PUBLICATION_COLORS["fill_opacity"], color=color)

    ax.set_title(title or f"{column} Over Time", fontweight="bold", pad=12)
    ax.set_ylabel(ylabel or _describe_column(df, column))
    ax.set_xlabel("")

    _format_time_axis(ax, idx)

    fig.tight_layout(pad=VisualizationConfig().tight_layout_pad)
    return fig, ax


def plot_time_series_multi(
    df: pd.DataFrame,
    columns: List[str],
    figsize: Tuple[float, float] = (14, 10),
) -> Tuple[Figure, plt.Axes]:
    """Multi-panel time series for multiple columns on shared x-axis."""
    n = len(columns)
    fig, axes = plt.subplots(n, 1, figsize=figsize, sharex=True)
    if n == 1:
        axes = [axes]

    colors = sns.color_palette("husl", n)
    for i, (col, color) in enumerate(zip(columns, colors)):
        plot_time_series(df, col, ax=axes[i], color=color)
        axes[i].set_title(col, fontweight="bold", fontsize=12)

    axes[-1].set_xlabel("Date")
    fig.suptitle("Key Time Series Overview", fontweight="bold", fontsize=14, y=1.01)
    fig.tight_layout(pad=2.0)
    return fig, axes


# ──────────────────────────────────────────────────────────────
# 3. Monthly Trends (Seasonal Subseries)
# ──────────────────────────────────────────────────────────────
def plot_monthly_trend(
    df: pd.DataFrame,
    column: str,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    color: str = PUBLICATION_COLORS["primary"],
) -> Tuple[Figure, plt.Axes]:
    """Plot mean ± std by calendar month to reveal seasonality.

    Parameters
    ----------
    df : pd.DataFrame
        Data with PeriodIndex.
    column : str
        Column to analyze.
    ax : plt.Axes, optional
        Existing axes.
    title : str, optional
        Plot title.
    color : str, default "#1f77b4"
        Bar and line color.

    Returns
    -------
    (fig, ax) tuple.
    """
    if column not in df.columns:
        return _empty_figure(f"Column '{column}' not found")

    idx = _get_index(df)
    if idx is None:
        return _empty_figure("No time index")

    months = idx.month
    monthly_stats = df.groupby(months)[column].agg(["mean", "std", "count"])

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.figure

    month_labels = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]

    x = np.arange(1, 13)
    valid = monthly_stats.index
    means = monthly_stats["mean"].reindex(x)
    stds = monthly_stats["std"].reindex(x)

    ax.bar(
        valid,
        monthly_stats["mean"],
        yerr=monthly_stats["std"],
        color=color,
        alpha=0.7,
        edgecolor="white",
        linewidth=0.8,
        capsize=4,
        error_kw={"linewidth": 1.5, "alpha": 0.6},
    )

    # Overall mean line
    overall_mean = df[column].mean()
    ax.axhline(
        overall_mean,
        color="gray",
        linestyle="--",
        linewidth=1,
        alpha=0.7,
        label=f"Overall Mean={overall_mean:.4f}",
    )

    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_labels)
    ax.set_title(title or f"Monthly Seasonality: {column}", fontweight="bold", pad=12)
    ax.set_xlabel("Month")
    ax.set_ylabel(_describe_column(df, column))
    ax.legend(loc="best", fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    fig.tight_layout(pad=VisualizationConfig().tight_layout_pad)
    return fig, ax


# ──────────────────────────────────────────────────────────────
# 4. Yearly Trends
# ──────────────────────────────────────────────────────────────
def plot_yearly_trend(
    df: pd.DataFrame,
    column: str,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> Tuple[Figure, plt.Axes]:
    """Plot yearly trend: line per year showing monthly values.

    Parameters
    ----------
    df : pd.DataFrame
        Data with PeriodIndex.
    column : str
        Column to plot.
    ax : plt.Axes, optional
        Existing axes.
    title : str, optional
        Plot title.

    Returns
    -------
    (fig, ax) tuple.
    """
    if column not in df.columns:
        return _empty_figure(f"Column '{column}' not found")

    idx = _get_index(df)
    if idx is None:
        return _empty_figure("No time index")

    data = df[[column]].copy()
    data["year"] = idx.year
    data["month"] = idx.month

    pivot = data.pivot_table(index="month", columns="year", values=column)

    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 5.5))
    else:
        fig = ax.figure

    colors = sns.color_palette("husl", len(pivot.columns))
    month_labels = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]

    for i, (year, values) in enumerate(pivot.items()):
        ax.plot(
            range(1, 13),
            values.values,
            marker="o",
            linewidth=1.8,
            markersize=4,
            label=str(year),
            color=colors[i],
            alpha=0.85,
        )

    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_labels)
    ax.set_title(title or f"Yearly Trend: {column}", fontweight="bold", pad=12)
    ax.set_xlabel("Month")
    ax.set_ylabel(_describe_column(df, column))
    ax.legend(title="Year", frameon=True, fontsize=8, title_fontsize=9, loc="best")

    fig.tight_layout(pad=VisualizationConfig().tight_layout_pad)
    return fig, ax


# ──────────────────────────────────────────────────────────────
# 5. Scatter Plots
# ──────────────────────────────────────────────────────────────
def plot_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    hue_col: Optional[str] = None,
    color: str = PUBLICATION_COLORS["primary"],
    add_regression: bool = True,
) -> Tuple[Figure, plt.Axes]:
    """Scatter plot of y vs x with optional regression line and hue.

    Parameters
    ----------
    df : pd.DataFrame
        Data.
    x_col : str
        X-axis column.
    y_col : str
        Y-axis column.
    ax : plt.Axes, optional
        Existing axes.
    title : str, optional
        Plot title.
    hue_col : str, optional
        Categorical column for point colouring.
    color : str, default "#1f77b4"
        Point color (ignored if hue_col is set).
    add_regression : bool, default True
        Overlay OLS regression line with CI band.

    Returns
    -------
    (fig, ax) tuple.
    """
    for col in [x_col, y_col]:
        if col not in df.columns:
            return _empty_figure(f"Column '{col}' not found")

    data = df[[x_col, y_col] + ([hue_col] if hue_col else [])].dropna()
    if len(data) < 5:
        return _empty_figure(f"Insufficient data (n={len(data)})")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))
    else:
        fig = ax.figure

    if hue_col and hue_col in df.columns:
        scatter = ax.scatter(
            data[x_col],
            data[y_col],
            c=_numeric_hue(data[hue_col]),
            cmap="viridis",
            alpha=0.7,
            edgecolors="white",
            linewidth=0.5,
            s=40,
        )
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label(hue_col, fontsize=10)
    else:
        ax.scatter(
            data[x_col],
            data[y_col],
            color=color,
            alpha=0.65,
            edgecolors="white",
            linewidth=0.5,
            s=40,
        )

    if add_regression and len(data) >= 10:
        try:
            sns.regplot(
                data=data,
                x=x_col,
                y=y_col,
                scatter=False,
                ci=95,
                line_kws={"color": _darken_color(color, 0.6), "linewidth": 1.8},
                ax=ax,
            )
        except Exception:
            pass

    # Pearson correlation
    r = data[x_col].corr(data[y_col])
    ax.annotate(
        f"r = {r:.4f}",
        xy=(0.05, 0.95),
        xycoords="axes fraction",
        fontsize=10,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
    )

    ax.set_title(title or f"{y_col} vs {x_col}", fontweight="bold", pad=12)
    ax.set_xlabel(_describe_column(df, x_col))
    ax.set_ylabel(_describe_column(df, y_col))

    fig.tight_layout(pad=VisualizationConfig().tight_layout_pad)
    return fig, ax


def _numeric_hue(s: pd.Series) -> np.ndarray:
    """Convert hue series to numeric array for colour mapping."""
    if s.dtype == object or s.dtype.name == "category":
        codes = s.astype("category").cat.codes
        return codes.values
    return s.values


def plot_scatter_matrix(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (14, 14),
) -> Tuple[Figure, np.ndarray]:
    """Scatter matrix for a subset of features."""
    if columns is None:
        columns = _select_key_features(df)[:5]
    n = len(columns)
    fig, axes = plt.subplots(n, n, figsize=figsize)

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if i == j:
                # Diagonal: distribution
                plot_distribution(df, columns[i], ax=ax)
                ax.set_ylabel("")
                ax.set_xlabel("")
            else:
                plot_scatter(df, columns[j], columns[i], ax=ax, add_regression=(n <= 5))
                if i < n - 1:
                    ax.set_xlabel("")
                if j > 0:
                    ax.set_ylabel("")

    fig.suptitle("Scatter Matrix of Key Features", fontweight="bold", fontsize=14, y=1.01)
    fig.tight_layout(pad=2.0)
    return fig, axes


# ──────────────────────────────────────────────────────────────
# 6. Correlation Heatmap
# ──────────────────────────────────────────────────────────────
def plot_correlation_heatmap(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    cmap: str = "RdBu_r",
    annot: bool = True,
    figsize: Tuple[float, float] = (12, 10),
) -> Tuple[Figure, plt.Axes]:
    """Plot an annotated correlation heatmap with hierarchical clustering.

    Parameters
    ----------
    df : pd.DataFrame
        Data.
    columns : list of str, optional
        Columns to include. If None, uses all numeric columns with variance.
    ax : plt.Axes, optional
        Existing axes.
    title : str, optional
        Plot title.
    cmap : str, default "RdBu_r"
        Colormap for heatmap.
    annot : bool, default True
        Annotate each cell with the correlation value.
    figsize : tuple, default (12, 10)
        Figure size.

    Returns
    -------
    (fig, ax) tuple.
    """
    if columns is None:
        columns = _select_numeric(df, max_cols=25)

    available = [c for c in columns if c in df.columns]
    if len(available) < 2:
        return _empty_figure("Need at least 2 columns for heatmap")

    corr = df[available].corr()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        center=0,
        annot=annot,
        fmt=".2f",
        linewidths=0.5,
        square=True,
        cbar_kws={"shrink": 0.75, "label": "Pearson r"},
        ax=ax,
    )

    ax.set_title(title or "Feature Correlation Heatmap", fontweight="bold", pad=16)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)

    fig.tight_layout(pad=VisualizationConfig().tight_layout_pad)
    return fig, ax


# ──────────────────────────────────────────────────────────────
# 7. Feature Correlation Matrix (Target-Centric)
# ──────────────────────────────────────────────────────────────
def plot_feature_correlation_matrix(
    df: pd.DataFrame,
    target_col: str = "aot_close",
    columns: Optional[List[str]] = None,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    top_n: int = 20,
) -> Tuple[Figure, plt.Axes]:
    """Plot target-centric correlation bar chart (all features vs target).

    Parameters
    ----------
    df : pd.DataFrame
        Data.
    target_col : str, default "aot_close"
        Target column.
    columns : list of str, optional
        Features to correlate with target. If None, all numeric columns.
    ax : plt.Axes, optional
        Existing axes.
    title : str, optional
        Plot title.
    top_n : int, default 20
        Show top N features by absolute correlation.

    Returns
    -------
    (fig, ax) tuple.
    """
    if target_col not in df.columns:
        return _empty_figure(f"Target '{target_col}' not found")

    if columns is None:
        columns = _select_numeric(df, max_cols=50)

    features = [c for c in columns if c != target_col and c in df.columns]
    if not features:
        return _empty_figure("No feature columns found")

    corr = df[features + [target_col]].corr()[target_col].drop(target_col).dropna()
    corr = corr.abs().sort_values(ascending=False).head(top_n)
    corr_signed = df[features + [target_col]].corr()[target_col].drop(target_col)
    corr_signed = corr_signed.loc[corr.index]

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, max(5, len(corr) * 0.3)))
    else:
        fig = ax.figure

    colors = [
        PUBLICATION_COLORS["positive"] if v >= 0 else PUBLICATION_COLORS["negative"]
        for v in corr_signed.values
    ]

    bars = ax.barh(
        range(len(corr_signed)), corr_signed.values, color=colors, alpha=0.8, edgecolor="white"
    )

    ax.set_yticks(range(len(corr_signed)))
    ax.set_yticklabels(corr_signed.index, fontsize=9)
    ax.set_xlabel("Pearson Correlation with Target", fontsize=12)
    ax.set_title(title or f"Feature Correlation with {target_col}", fontweight="bold", pad=12)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.invert_yaxis()

    # Annotate values
    for i, (v, bar) in enumerate(zip(corr_signed.values, bars)):
        label_x = v + 0.02 if v >= 0 else v - 0.12
        ax.text(label_x, i, f"{v:.3f}", va="center", fontsize=8)

    fig.tight_layout(pad=VisualizationConfig().tight_layout_pad)
    return fig, ax


# ──────────────────────────────────────────────────────────────
# 8. Pair Plot
# ──────────────────────────────────────────────────────────────
def plot_pairplot(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    hue_col: Optional[str] = None,
    title: Optional[str] = None,
    sample: Optional[int] = 500,
) -> Figure:
    """Create a seaborn pairplot for a subset of features.

    Parameters
    ----------
    df : pd.DataFrame
        Data.
    columns : list of str, optional
        Columns to include. If None, uses key features.
    hue_col : str, optional
        Column for categorical colouring (e.g., 'year').
    title : str, optional
        Figure title.
    sample : int, optional
        If set, downsample to N rows for performance.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if columns is None:
        columns = _select_key_features(df)[:6]

    available = [c for c in columns if c in df.columns]
    if len(available) < 2:
        return _empty_figure("Need at least 2 columns for pairplot")[0]

    plot_df = df[available + ([hue_col] if hue_col and hue_col in df.columns else [])].dropna()

    if sample and len(plot_df) > sample:
        plot_df = plot_df.sample(n=sample, random_state=42)

    if hue_col and hue_col in plot_df.columns:
        plot_df[hue_col] = plot_df[hue_col].astype(str)
    else:
        hue_col = None

    pair = sns.pairplot(
        plot_df,
        vars=available,
        hue=hue_col,
        diag_kind="kde",
        plot_kws={"alpha": 0.5, "s": 15, "edgecolor": "white"},
        diag_kws={"alpha": 0.6},
    )

    pair.fig.suptitle(title or "Pair Plot of Key Features", fontweight="bold", fontsize=14, y=1.02)
    pair.fig.tight_layout(pad=2.0)

    return pair.fig


# ──────────────────────────────────────────────────────────────
# 9. Outlier Detection
# ──────────────────────────────────────────────────────────────
def plot_outlier_box(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (14, 6),
) -> Tuple[Figure, plt.Axes]:
    """Box plots for outlier detection across multiple columns.

    Parameters
    ----------
    df : pd.DataFrame
        Data.
    columns : list of str, optional
        Columns to plot. If None, uses key numeric features.
    ax : plt.Axes, optional
        Existing axes.
    title : str, optional
        Plot title.
    figsize : tuple, default (14, 6)
        Figure size.

    Returns
    -------
    (fig, ax) tuple.
    """
    if columns is None:
        columns = _select_numeric(df, max_cols=15)

    available = [c for c in columns if c in df.columns and df[c].nunique() > 5]
    if not available:
        return _empty_figure("No suitable columns for box plots")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    plot_data = df[available].melt(var_name="Feature", value_name="Value")

    sns.boxplot(
        data=plot_data,
        x="Feature",
        y="Value",
        ax=ax,
        palette="husl",
        width=0.6,
        linewidth=0.8,
        fliersize=2,
    )

    ax.set_title(title or "Outlier Detection: Box Plots", fontweight="bold", pad=12)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=45)

    fig.tight_layout(pad=VisualizationConfig().tight_layout_pad)
    return fig, ax


def plot_outlier_timeseries(
    df: pd.DataFrame,
    column: str,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    color: str = PUBLICATION_COLORS["primary"],
    iqr_multiplier: float = 1.5,
) -> Tuple[Figure, plt.Axes]:
    """Time series with outlier points highlighted in red.

    Parameters
    ----------
    df : pd.DataFrame
        Data with time index.
    column : str
        Column to analyze.
    ax : plt.Axes, optional
        Existing axes.
    title : str, optional
        Plot title.
    color : str, default "#1f77b4"
        Line color.
    iqr_multiplier : float, default 1.5
        IQR multiplier for outlier detection.

    Returns
    -------
    (fig, ax) tuple.
    """
    if column not in df.columns:
        return _empty_figure(f"Column '{column}' not found")

    data = df[column].dropna()
    idx = _get_index(data.to_frame())
    if idx is None or len(data) < 5:
        return _empty_figure("Insufficient data")

    # IQR outlier detection
    q1, q3 = data.quantile(0.25), data.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - iqr_multiplier * iqr, q3 + iqr_multiplier * iqr
    outlier_mask = (data < lower) | (data > upper)

    if ax is None:
        fig, ax = plt.subplots(figsize=VisualizationConfig().figsize)
    else:
        fig = ax.figure

    ax.plot(idx, data.values, color=color, linewidth=1.5, alpha=0.7, label="Series")
    ax.scatter(
        idx[outlier_mask],
        data[outlier_mask].values,
        color=PUBLICATION_COLORS["negative"],
        s=30,
        zorder=5,
        edgecolors="white",
        linewidth=0.5,
        label=f"Outliers (IQR x{iqr_multiplier})",
    )

    # Fences
    ax.axhline(
        lower,
        color="gray",
        linestyle="--",
        linewidth=0.8,
        alpha=0.5,
        label=f"Lower fence={lower:.3f}",
    )
    ax.axhline(
        upper,
        color="gray",
        linestyle="--",
        linewidth=0.8,
        alpha=0.5,
        label=f"Upper fence={upper:.3f}",
    )

    ax.set_title(title or f"Outlier Detection: {column}", fontweight="bold", pad=12)
    ax.set_ylabel(_describe_column(df, column))
    ax.legend(fontsize=8, frameon=True)
    _format_time_axis(ax, idx)

    fig.tight_layout(pad=VisualizationConfig().tight_layout_pad)
    return fig, ax


# ──────────────────────────────────────────────────────────────
# 10. Trend Decomposition
# ──────────────────────────────────────────────────────────────
def plot_decomposition(
    df: pd.DataFrame,
    column: str = "aot_close",
    model: str = "additive",
    period: int = 12,
    figsize: Tuple[float, float] = (12, 9),
) -> Tuple[Figure, np.ndarray]:
    """Seasonal decomposition of a time series into trend, seasonal, residual.

    Parameters
    ----------
    df : pd.DataFrame
        Monthly data with PeriodIndex or DatetimeIndex.
    column : str, default "aot_close"
        Column to decompose.
    model : str, default "additive"
        Decomposition model ('additive' or 'multiplicative').
    period : int, default 12
        Seasonal period (12 for monthly data with yearly seasonality).
    figsize : tuple, default (12, 9)
        Figure size.

    Returns
    -------
    (fig, axes) tuple with 4 subplots: observed, trend, seasonal, residual.
    """
    if column not in df.columns:
        return _empty_figure(f"Column '{column}' not found"), np.array([])

    data = df[column].dropna()
    if len(data) < period * 2:
        return _empty_figure(
            f"Need at least {period * 2} observations for decomposition (have {len(data)})"
        ), np.array([])

    # Build a proper DatetimeIndex time series for statsmodels
    raw_idx = data.index
    if isinstance(raw_idx, pd.PeriodIndex):
        ts_index = raw_idx.to_timestamp(how="end")
    elif isinstance(raw_idx, pd.DatetimeIndex):
        ts_index = raw_idx
    else:
        ts_index = pd.date_range(start=raw_idx[0], periods=len(raw_idx), freq="ME")

    ts = pd.Series(data.values, index=ts_index)
    ts = ts.asfreq("ME")  # ensure explicit monthly frequency

    try:
        decomp = seasonal_decompose(ts, model=model, period=period, extrapolate_trend="freq")
    except Exception as exc:
        return _empty_figure(f"Decomposition failed: {exc}"), np.array([])

    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
    labels = ["Observed", "Trend", "Seasonal", "Residual"]

    for ax, component, label in zip(
        axes, [decomp.observed, decomp.trend, decomp.seasonal, decomp.resid], labels
    ):
        ax.plot(component.index, component.values, linewidth=1.5)
        ax.set_ylabel(label, fontweight="bold", fontsize=10)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.grid(True, alpha=0.3)

    axes[0].set_title(
        f"Time Series Decomposition: {column} ({model}, period={period})", fontweight="bold", pad=12
    )
    axes[-1].set_xlabel("Date")

    _format_time_axis(axes[-1], ts.index)

    fig.tight_layout(pad=2.0)
    return fig, axes


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def _get_index(df: pd.DataFrame) -> Optional[pd.DatetimeIndex]:
    """Return a DatetimeIndex if a time index exists.

    Converts PeriodIndex to DatetimeIndex so matplotlib can consume it.
    """
    idx = df.index
    if isinstance(idx, pd.DatetimeIndex):
        return idx
    if isinstance(idx, pd.PeriodIndex):
        try:
            idx.dtype  # may raise if mixed types
            ts = idx.to_timestamp()
            # freq info may be lost; assign monthly-end frequency
            if ts.freq is None:
                ts = ts.asfreq("ME")
            return ts
        except Exception:
            return None
    # Try converting
    try:
        return pd.DatetimeIndex(idx)
    except Exception:
        return None


def _format_time_axis(ax: plt.Axes, idx: pd.DatetimeIndex) -> None:
    """Apply date formatting to the x-axis."""
    dates = idx

    n = len(dates)
    if n > 60:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1, 7]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    elif n > 24:
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    else:
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")


def _empty_figure(message: str) -> Tuple[Figure, Optional[plt.Axes]]:
    """Return a figure with a centered error message."""
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        fontsize=13,
        transform=ax.transAxes,
        style="italic",
        color="gray",
    )
    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])
    logger.warning("Empty figure: %s", message)
    return fig, ax


# ──────────────────────────────────────────────────────────────
# EDAVisualizer — orchestrator
# ──────────────────────────────────────────────────────────────
class EDAVisualizer:
    """Orchestrator for all EDA visualizations.

    Loads the feature dataset and provides methods to generate individual
    plots or produce a complete EDA report as SVG files.

    Parameters
    ----------
    df : pd.DataFrame, optional
        Pre-loaded feature dataset.
    config : VisualizationConfig, optional
        Styling configuration.
    log_level : str, default "INFO"
        Logging verbosity.
    """

    def __init__(
        self,
        df: Optional[pd.DataFrame] = None,
        config: Optional[VisualizationConfig] = None,
        log_level: str = "INFO",
    ):
        self.df = df
        self.config = config or VisualizationConfig()
        self.config.apply()
        self._generated_paths: List[Path] = []

        _setup_logging(log_level)

        if self.df is not None:
            logger.info("EDAVisualizer initialized: %d rows x %d cols", *self.df.shape)

    def load(self, path: Union[str, Path]) -> pd.DataFrame:
        """Load feature dataset from CSV."""
        p = Path(path)
        df = pd.read_csv(p, index_col=0)
        if "month" in df.columns:
            df["month"] = pd.to_datetime(df["month"], errors="coerce")
            df.set_index("month", inplace=True)
        df.index = pd.PeriodIndex(df.index, freq="M")
        self.df = df
        logger.info("Loaded: %s (%d rows x %d cols)", p, *df.shape)
        return df

    # ── Single-plot wrappers ────────────────────────────────
    def plot_distribution(self, column: str, **kw) -> Figure:
        fig, _ = plot_distribution(self.df, column, **kw)
        return fig

    def plot_time_series(self, column: str, **kw) -> Figure:
        fig, _ = plot_time_series(self.df, column, **kw)
        return fig

    def plot_monthly_trend(self, column: str, **kw) -> Figure:
        fig, _ = plot_monthly_trend(self.df, column, **kw)
        return fig

    def plot_yearly_trend(self, column: str, **kw) -> Figure:
        fig, _ = plot_yearly_trend(self.df, column, **kw)
        return fig

    def plot_scatter(self, x_col: str, y_col: str, **kw) -> Figure:
        fig, _ = plot_scatter(self.df, x_col, y_col, **kw)
        return fig

    def plot_correlation_heatmap(self, **kw) -> Figure:
        fig, _ = plot_correlation_heatmap(self.df, **kw)
        return fig

    def plot_feature_correlation_matrix(self, **kw) -> Figure:
        fig, _ = plot_feature_correlation_matrix(self.df, **kw)
        return fig

    def plot_outlier_box(self, **kw) -> Figure:
        fig, _ = plot_outlier_box(self.df, **kw)
        return fig

    def plot_outlier_timeseries(self, column: str, **kw) -> Figure:
        fig, _ = plot_outlier_timeseries(self.df, column, **kw)
        return fig

    def plot_decomposition(self, **kw) -> Figure:
        fig, _ = plot_decomposition(self.df, **kw)
        return fig

    # ── Save helper ─────────────────────────────────────────
    def export(self, fig: Figure, path: Union[str, Path]) -> Path:
        """Export a single figure to SVG (or other format)."""
        p = export_figure(fig, path)
        self._generated_paths.append(p)
        return p

    # ── Generate all ────────────────────────────────────────
    def generate_all(
        self,
        output_dir: Union[str, Path] = "reports/figures",
        include_pairplot: bool = True,
        include_decomposition: bool = True,
        highlight_covid: bool = True,
    ) -> Dict[str, Path]:
        """Generate all EDA figures and save as SVG files.

        Parameters
        ----------
        output_dir : str or Path, default "reports/figures"
            Directory to save figures.
        include_pairplot : bool, default True
            Include pair plot (can be slow for large feature sets).
        include_decomposition : bool, default True
            Include trend decomposition.
        highlight_covid : bool, default True
            Shade COVID-19 period on time series.

        Returns
        -------
        dict mapping figure name -> Path to saved SVG file.
        """
        if self.df is None:
            logger.error("No data loaded. Call load() first.")
            return {}

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths: Dict[str, Path] = {}

        covid_periods = None
        if highlight_covid:
            covid_periods = [("2020-03", "2021-12")]

        plot_generators = {
            "01_distribution_grid": lambda: plot_distribution_grid(self.df),
            "02_time_series_aot": lambda: plot_time_series(
                self.df,
                "aot_close",
                highlight_periods=covid_periods,
                title="AOT Closing Price Over Time",
                color=PUBLICATION_COLORS["target"],
            ),
            "03_time_series_multi": lambda: plot_time_series_multi(
                self.df,
                _select_key_features(self.df)[:4],
            ),
            "04_monthly_trend": lambda: plot_monthly_trend(
                self.df,
                "aot_close",
                title="Monthly Seasonality: AOT Close",
            ),
            "05_yearly_trend": lambda: plot_yearly_trend(
                self.df,
                "aot_close",
                title="Yearly Trend: AOT Close by Month",
            ),
            "06_scatter_aot_vs_tourists": lambda: plot_scatter(
                self.df,
                "tourists_total_arrivals",
                "aot_close",
                title="AOT Close vs Tourist Arrivals",
            ),
            "07_scatter_aot_vs_set": lambda: plot_scatter(
                self.df,
                "set_close",
                "aot_close",
                title="AOT Close vs SET Index",
            ),
            "08_scatter_aot_vs_fx": lambda: plot_scatter(
                self.df,
                "fx_usdthb_rate",
                "aot_close",
                title="AOT Close vs USD/THB",
            ),
            "09_correlation_heatmap": lambda: plot_correlation_heatmap(self.df),
            "10_feature_correlation_matrix": lambda: plot_feature_correlation_matrix(self.df),
            "11_outlier_box": lambda: plot_outlier_box(self.df),
            "12_outlier_timeseries": lambda: plot_outlier_timeseries(
                self.df,
                "aot_return",
                title="Outlier Detection: AOT Returns",
            ),
        }

        if include_decomposition:
            plot_generators["13_decomposition"] = lambda: plot_decomposition(self.df)[0]

        if include_pairplot:
            plot_generators["14_pairplot"] = lambda: plot_pairplot(self.df)

        for name, generator in plot_generators.items():
            try:
                result = generator()
                if isinstance(result, tuple):
                    fig = result[0]
                else:
                    fig = result
                path = self.export(fig, out / name)
                paths[name] = path
                plt.close(fig)
            except Exception as exc:
                logger.error("Failed to generate '%s': %s", name, exc)

        logger.info("Generated %d/%d figures in %s", len(paths), len(plot_generators), out)
        return paths

    # ── Streamlit integration ───────────────────────────────
    def streamlit_figures(self) -> List[Tuple[str, Figure]]:
        """Return list of (title, figure) tuples for Streamlit display.

        Usage in Streamlit:
            for title, fig in viz.streamlit_figures():
                st.subheader(title)
                st.pyplot(fig)
        """
        import streamlit as st  # noqa: F401

        items = [
            ("Feature Distributions", plot_distribution_grid(self.df)),
            (
                "AOT Closing Price Over Time",
                plot_time_series(
                    self.df,
                    "aot_close",
                    title="AOT Closing Price Over Time",
                    color=PUBLICATION_COLORS["target"],
                ),
            ),
            (
                "Monthly Seasonality",
                plot_monthly_trend(
                    self.df,
                    "aot_close",
                    title="Monthly Seasonality: AOT Close",
                ),
            ),
            (
                "AOT Close vs Tourist Arrivals",
                plot_scatter(
                    self.df,
                    "tourists_total_arrivals",
                    "aot_close",
                ),
            ),
            ("Correlation Heatmap", plot_correlation_heatmap(self.df)),
            ("Feature Correlation with Target", plot_feature_correlation_matrix(self.df)),
            ("Outlier Detection", plot_outlier_box(self.df)),
            ("Trend Decomposition", plot_decomposition(self.df)[0]),
        ]
        result = []
        for title, fig_ax_tuple in items:
            fig = fig_ax_tuple[0] if isinstance(fig_ax_tuple, tuple) else fig_ax_tuple
            result.append((title, fig))
        return result

    # ── Summary ─────────────────────────────────────────────
    def summary(self) -> str:
        """Return a text summary of available data."""
        if self.df is None:
            return "No data loaded."
        numeric = self.df.select_dtypes(include=[np.number]).columns.tolist()
        idx = _get_index(self.df)
        if idx is None:
            date_range = "N/A"
        else:
            date_range = f"{idx[0]} to {idx[-1]}"
        lines = [
            "EDAVisualizer Summary",
            "======================",
            f"Rows: {len(self.df)}",
            f"Columns: {len(self.df.columns)}",
            f"Numeric features: {len(numeric)}",
            f"Date range: {date_range}",
        ]
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# Module exports
# ──────────────────────────────────────────────────────────────
__all__ = [
    "EDAVisualizer",
    "VisualizationConfig",
    "export_figure",
    "plot_distribution",
    "plot_distribution_grid",
    "plot_time_series",
    "plot_time_series_multi",
    "plot_monthly_trend",
    "plot_yearly_trend",
    "plot_scatter",
    "plot_scatter_matrix",
    "plot_correlation_heatmap",
    "plot_feature_correlation_matrix",
    "plot_pairplot",
    "plot_outlier_box",
    "plot_outlier_timeseries",
    "plot_decomposition",
]
