"""
DataLoader — AOT Stock Network Data Collection Module
======================================================

Main entry point for acquiring, validating, caching, and documenting
all official data sources for the AOT Stock Network project.

Usage
-----
    from aot_stock_network.data import DataLoader

    dl = DataLoader()
    dl.fetch_all()                # Download all sources
    dl.validate_all()             # Validate all fetched data
    df = dl.get("set_aot")        # Get processed DataFrame
    dl.export_metadata_report()   # Generate data catalog

Official data sources
---------------------
  SET          — Stock Exchange of Thailand (AOT price, SET Index)
  MOTS         — Ministry of Tourism and Sports (arrivals, revenue)
  BOT          — Bank of Thailand (FX, policy rate, inflation)
  DATA.GO.TH   — Thailand Open Government Data Portal
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from aot_stock_network.data.fetcher import (
    FetchError,
    ParseError,
    fetch_source,
    get_cache_age_days,
)
from aot_stock_network.data.sources import (
    SOURCE_REGISTRY,
    DataSource,
    get_source,
    list_sources,
    source_registry_summary,
)
from aot_stock_network.data.validator import (
    ValidationResult,
    validate_dataframe,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Default paths
# ──────────────────────────────────────────────────────────────
DEFAULT_DATA_DIR = Path("data")
DEFAULT_CACHE_DIR = Path("data") / ".cache"
DEFAULT_PROCESSED_DIR = Path("data") / "processed"
DEFAULT_METADATA_PATH = Path("data") / "data_catalog.json"


# ──────────────────────────────────────────────────────────────
# Metadata tracker
# ──────────────────────────────────────────────────────────────
@dataclass
class FetchRecord:
    """A single fetch event logged to metadata."""

    source_name: str
    fetch_time: str
    rows_fetched: int
    columns: List[str]
    date_range: Optional[List[str]] = None
    validation: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class DataCatalog:
    """Full metadata for all data sources."""

    project: str = "AOT Stock Network"
    last_updated: str = ""
    sources: Dict[str, List[FetchRecord]] = field(default_factory=dict)

    def add_record(self, record: FetchRecord):
        self.sources.setdefault(record.source_name, []).append(record)
        self.last_updated = datetime.now().isoformat(timespec="seconds")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project": self.project,
            "last_updated": self.last_updated,
            "sources": {
                name: [asdict(r) for r in records] for name, records in self.sources.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataCatalog":
        catalog = cls(project=data.get("project", ""), last_updated=data.get("last_updated", ""))
        for source_name, records in data.get("sources", {}).items():
            catalog.sources[source_name] = [FetchRecord(**r) for r in records]
        return catalog


# ──────────────────────────────────────────────────────────────
# Logging setup
# ──────────────────────────────────────────────────────────────
def _setup_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root = logging.getLogger("aot_stock_network")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not root.handlers:
        root.addHandler(handler)
    # Also log to file
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    fh = logging.FileHandler(
        log_dir / f"data_collection_{datetime.now():%Y%m%d}.log", encoding="utf-8"
    )
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"))
    root.addHandler(fh)


# ──────────────────────────────────────────────────────────────
# DataLoader
# ──────────────────────────────────────────────────────────────
class DataLoader:
    """
    Orchestrator for acquiring, validating, and managing all data sources.

    Parameters
    ----------
    data_dir : str or Path, default "data"
        Root directory for raw, processed, and external data.
    cache_dir : str or Path, optional
        Directory for HTTP response cache. Defaults to data_dir / ".cache".
    log_level : str, default "INFO"
        Logging verbosity (DEBUG, INFO, WARNING, ERROR).
    auto_setup : bool, default True
        If True, create directory structure on init.
    """

    def __init__(
        self,
        data_dir: Union[str, Path] = "data",
        cache_dir: Optional[Union[str, Path]] = None,
        log_level: str = "INFO",
        auto_setup: bool = True,
    ):
        self.data_dir = Path(data_dir).resolve()
        self.cache_dir = Path(cache_dir).resolve() if cache_dir else self.data_dir / ".cache"
        self.processed_dir = self.data_dir / "processed"
        self.metadata_path = self.data_dir / "data_catalog.json"

        _setup_logging(log_level)

        self._sources: Dict[str, DataSource] = SOURCE_REGISTRY.copy()
        self._cache: Dict[str, pd.DataFrame] = {}
        self._metadata = self._load_metadata()
        self._validation_cache: Dict[str, ValidationResult] = {}

        if auto_setup:
            self._setup_directories()

        logger.info(
            "DataLoader initialized | data_dir=%s | sources=%d",
            self.data_dir,
            len(self._sources),
        )

    # ── Directory setup ──────────────────────────────────────
    def _setup_directories(self) -> None:
        for d in [self.data_dir, self.cache_dir, self.processed_dir]:
            d.mkdir(parents=True, exist_ok=True)
        for sub in ["raw", "external"]:
            (self.data_dir / sub).mkdir(exist_ok=True)
        for source_name in self._sources:
            (self.cache_dir / "raw" / source_name).mkdir(parents=True, exist_ok=True)

    # ── Metadata persistence ─────────────────────────────────
    def _load_metadata(self) -> DataCatalog:
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                logger.info("Loaded metadata from %s", self.metadata_path)
                return DataCatalog.from_dict(data)
            except (json.JSONDecodeError, KeyError) as exc:
                logger.warning("Could not load metadata, starting fresh: %s", exc)
        return DataCatalog()

    def _save_metadata(self) -> None:
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self._metadata.to_dict(), f, indent=2, ensure_ascii=False)
        logger.debug("Metadata saved to %s", self.metadata_path)

    def _log_fetch(
        self,
        source_name: str,
        rows: int,
        columns: List[str],
        date_range: Optional[List[str]] = None,
        validation: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        record = FetchRecord(
            source_name=source_name,
            fetch_time=datetime.now().isoformat(timespec="seconds"),
            rows_fetched=rows,
            columns=columns,
            date_range=date_range,
            validation=validation,
            error=error,
        )
        self._metadata.add_record(record)
        self._save_metadata()

    # ── Data persistence ─────────────────────────────────────
    def _save_processed(self, df: pd.DataFrame, source_name: str) -> Path:
        """Save processed DataFrame as CSV."""
        self.processed_dir.mkdir(exist_ok=True)
        csv_path = self.processed_dir / f"{source_name}.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        logger.debug("Saved processed data: %s (%d rows)", csv_path, len(df))
        return csv_path

    def _load_processed(self, source_name: str) -> Optional[pd.DataFrame]:
        csv_path = self.processed_dir / f"{source_name}.csv"
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path, parse_dates=True)
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"], errors="coerce")
                elif "month" in df.columns:
                    df["month"] = pd.to_datetime(df["month"], errors="coerce")
                elif "announce_date" in df.columns:
                    df["announce_date"] = pd.to_datetime(df["announce_date"], errors="coerce")
                logger.debug("Loaded cached processed data: %s", csv_path)
                return df
            except Exception as exc:
                logger.warning("Failed to load cached data for '%s': %s", source_name, exc)
        return None

    # ── Source info ──────────────────────────────────────────
    def list_sources(self, category: Optional[str] = None) -> List[str]:
        """Return list of source names, optionally filtered by category."""
        return list_sources(category)

    def list_categories(self) -> List[str]:
        """Return list of all data categories."""
        return sorted(set(s.category for s in self._sources.values()))

    def get_source_info(self, source_name: str) -> Dict[str, Any]:
        """Return full metadata dict for a single source."""
        source = get_source(source_name)
        return source.to_dict()

    def print_source_registry(self) -> None:
        """Print a human-readable summary of all sources."""
        print(source_registry_summary())

    # ── Fetch ────────────────────────────────────────────────
    def fetch(
        self,
        source_name: Optional[str] = None,
        force: bool = False,
        save: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for one or all sources.

        Parameters
        ----------
        source_name : str, optional
            Name of a single source to fetch. If None, fetch all sources.
        force : bool, default=False
            If True, bypass cache and re-download.
        save : bool, default=True
            If True, save processed DataFrames to disk.

        Returns
        -------
        dict of str -> pd.DataFrame
            Mapping of source names to their DataFrames.
        """
        if source_name is not None:
            names = [source_name]
        else:
            names = list(self._sources.keys())

        results: Dict[str, pd.DataFrame] = {}
        errors: List[str] = []

        for name in names:
            source = self._sources[name]

            if not force:
                cached = self._load_processed(name)
                if cached is not None and len(cached) > 0:
                    logger.info("Using cached processed data for '%s'", name)
                    self._cache[name] = cached
                    results[name] = cached
                    continue

            logger.info("Fetching '%s' from %s ...", name, source.institution)
            t0 = time.time()

            try:
                df = fetch_source(source, self.cache_dir, force=force)

                duration = time.time() - t0
                date_range = None
                if "date" in df.columns and not df.empty:
                    dates = df["date"]
                    if pd.api.types.is_datetime64_any_dtype(dates) or dates.dtype == object:
                        parsed = pd.to_datetime(dates, errors="coerce").dropna()
                        if not parsed.empty:
                            date_range = [
                                parsed.min().isoformat()[:10],
                                parsed.max().isoformat()[:10],
                            ]

                self._cache[name] = df

                if save and not df.empty:
                    self._save_processed(df, name)

                self._log_fetch(
                    source_name=name,
                    rows=len(df),
                    columns=list(df.columns),
                    date_range=date_range,
                )

                logger.info(
                    "Fetched '%s': %d rows, %d cols in %.1fs",
                    name,
                    len(df),
                    len(df.columns),
                    duration,
                )
                results[name] = df

            except (FetchError, ParseError) as exc:
                logger.error("Failed to fetch '%s': %s", name, exc)
                self._log_fetch(
                    source_name=name,
                    rows=0,
                    columns=[],
                    error=str(exc),
                )
                errors.append(name)

        if errors:
            logger.warning("Fetched %d/%d sources; failures: %s", len(results), len(names), errors)

        return results

    def fetch_all(self, force: bool = False) -> Dict[str, pd.DataFrame]:
        """Convenience: fetch every registered source."""
        return self.fetch(source_name=None, force=force)

    def fetch_source(self, source_name: str, force: bool = False) -> pd.DataFrame:
        """Convenience: fetch a single source by name."""
        result = self.fetch(source_name=source_name, force=force)
        return result.get(source_name, pd.DataFrame())

    # ── Validate ─────────────────────────────────────────────
    def validate(self, source_name: Optional[str] = None) -> Dict[str, ValidationResult]:
        """
        Validate fetched data for one or all sources.

        Parameters
        ----------
        source_name : str, optional
            Single source to validate. If None, validate all cached sources.

        Returns
        -------
        dict of str -> ValidationResult
        """
        if source_name is not None:
            names = [source_name]
        else:
            names = list(self._cache.keys())
            if not names:
                names = list(self._sources.keys())
                for n in names:
                    cached = self._load_processed(n)
                    if cached is not None:
                        self._cache[n] = cached

        results: Dict[str, ValidationResult] = {}

        for name in names:
            source = self._sources.get(name)
            if source is None:
                logger.warning("No source definition for '%s', skipping", name)
                continue

            df = self._cache.get(name)
            if df is None:
                df = self._load_processed(name)
                if df is None:
                    logger.warning("No data available to validate '%s'. Fetch first.", name)
                    continue
                self._cache[name] = df

            logger.info("Validating '%s' (%d rows, %d cols) ...", name, len(df), len(df.columns))
            result = validate_dataframe(df, source)
            self._validation_cache[name] = result
            results[name] = result

            print(f"  {result.summary}")

        return results

    def validate_all(self) -> Dict[str, ValidationResult]:
        """Convenience: validate all cached data."""
        return self.validate()

    # ── Get data ─────────────────────────────────────────────
    def get(self, source_name: str) -> pd.DataFrame:
        """
        Get processed DataFrame for a source.

        Loads from memory cache first, then from disk cache, then
        raises KeyError.

        Parameters
        ----------
        source_name : str
            Name of the data source.

        Returns
        -------
        pd.DataFrame
        """
        if source_name in self._cache:
            return self._cache[source_name]

        df = self._load_processed(source_name)
        if df is not None:
            self._cache[source_name] = df
            return df

        raise KeyError(
            f"No data found for '{source_name}'. "
            f"Call fetch('{source_name}') first, or run fetch_all()."
        )

    # ── Update ───────────────────────────────────────────────
    def update(self, source_names: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Incremental update: fetch only stale or missing sources.

        A source is considered stale if:
          - It has no cached processed file, OR
          - The cache age exceeds its frequency-based expiry.

        Parameters
        ----------
        source_names : list of str, optional
            Subset of sources to update. If None, update all stale sources.

        Returns
        -------
        dict of str -> pd.DataFrame
        """
        from datetime import timedelta

        if source_names is None:
            candidates = list(self._sources.keys())
        else:
            candidates = source_names

        to_fetch: List[str] = []
        for name in candidates:
            processed_path = self.processed_dir / f"{name}.csv"
            if not processed_path.exists():
                to_fetch.append(name)
                continue
            source = self._sources[name]
            max_age = get_cache_age_days(source)
            mtime = datetime.fromtimestamp(processed_path.stat().st_mtime)
            if (datetime.now() - mtime) > timedelta(days=max_age):
                to_fetch.append(name)

        if not to_fetch:
            logger.info("All sources up-to-date (no stale sources found)")
            return {}

        logger.info("Update: fetching %d stale sources: %s", len(to_fetch), to_fetch)
        return self.fetch(source_name=None if len(to_fetch) == len(self._sources) else to_fetch[0])

    # ── Metadata / reporting ─────────────────────────────────
    def get_metadata(self, source_name: Optional[str] = None) -> Any:
        """
        Get fetch history for a source or the full catalog.

        Parameters
        ----------
        source_name : str, optional
            If provided, return fetch records for that source.
            Otherwise, return the full DataCatalog.

        Returns
        -------
        DataCatalog or list of FetchRecord
        """
        if source_name is None:
            return self._metadata
        return self._metadata.sources.get(source_name, [])

    def export_metadata_report(self, path: Optional[Union[str, Path]] = None) -> Path:
        """
        Export the full data catalog as a JSON report.

        Parameters
        ----------
        path : str or Path, optional
            Output path. Default: data/data_catalog.json

        Returns
        -------
        Path to the exported report.
        """
        output = Path(path) if path else self.metadata_path
        self._save_metadata()
        logger.info("Data catalog exported to %s", output)
        return output

    def data_summary(self) -> pd.DataFrame:
        """Return a summary DataFrame of all sources and their fetch status."""
        rows = []
        for name, source in self._sources.items():
            records = self._metadata.sources.get(name, [])
            last_fetch = records[-1].fetch_time if records else "—"
            rows_count = records[-1].rows_fetched if records else 0
            has_error = records[-1].error is not None if records else False
            rows.append(
                {
                    "source": name,
                    "category": source.category,
                    "institution": source.institution,
                    "frequency": source.frequency,
                    "last_fetch": last_fetch,
                    "rows": rows_count,
                    "has_error": has_error,
                }
            )
        return pd.DataFrame(rows)

    # ── Clear ────────────────────────────────────────────────
    def clear_cache(self, source_name: Optional[str] = None) -> None:
        """
        Clear cached data for a source or all sources.

        Parameters
        ----------
        source_name : str, optional
            If provided, clear only that source's cache.
            Otherwise, clear all caches.
        """
        if source_name is not None:
            paths = [
                self.cache_dir / "raw" / source_name,
                self.processed_dir / f"{source_name}.csv",
            ]
        else:
            paths = [self.cache_dir, self.processed_dir]

        for p in paths:
            if p.exists():
                if p.is_dir():
                    import shutil

                    shutil.rmtree(p)
                    logger.info("Cleared cache directory: %s", p)
                else:
                    p.unlink()
                    logger.info("Cleared cached file: %s", p)

        if source_name:
            self._cache.pop(source_name, None)
        else:
            self._cache.clear()

    # ── Context manager ──────────────────────────────────────
    def __enter__(self) -> "DataLoader":
        return self

    def __exit__(self, *args) -> None:
        self._save_metadata()
