"""
HTTP and file-based data fetching with caching for the AOT Stock Network project.

Supports five fetch strategies:
  - html_table   : download HTML page and parse <table> into a DataFrame
  - json_api     : call a JSON REST endpoint and normalize the response
  - json_api_oaq : call the SET OAQ API with api-key auth (date iteration)
  - excel_download: download an .xlsx file and parse into a DataFrame
  - manual_file  : load a user-provided local file
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

from aot_stock_network.data.sources import DataSource

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────
DEFAULT_TIMEOUT = 60  # seconds
DEFAULT_RETRIES = 3
RATE_LIMIT_DELAY = 1.0  # seconds between requests
CACHE_EXPIRY_DAYS = {
    "daily": 1,
    "monthly": 7,
    "quarterly": 30,
    "irregular": 7,
    "various": 7,
}


# ──────────────────────────────────────────────────────────────
# Cache utilities
# ──────────────────────────────────────────────────────────────
def _cache_path(cache_dir: Path, source_name: str, suffix: str = ".pkl") -> Path:
    return cache_dir / f"{source_name}{suffix}"


def _raw_cache_path(cache_dir: Path, source_name: str, ext: str = ".html") -> Path:
    return cache_dir / "raw" / source_name / f"{source_name}{ext}"


def _cache_valid(cache_path: Path, max_age_days: int) -> bool:
    if not cache_path.exists():
        return False
    mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
    return (datetime.now() - mtime) < timedelta(days=max_age_days)


def _safe_filename(url: str) -> str:
    h = hashlib.sha256(url.encode()).hexdigest()[:16]
    return h


# ──────────────────────────────────────────────────────────────
# HTTP session
# ──────────────────────────────────────────────────────────────
_http_session: Optional[requests.Session] = None


def _get_session() -> requests.Session:
    global _http_session
    if _http_session is None:
        _http_session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            max_retries=DEFAULT_RETRIES,
            pool_connections=4,
            pool_maxsize=8,
        )
        _http_session.mount("https://", adapter)
        _http_session.mount("http://", adapter)
    return _http_session


# ──────────────────────────────────────────────────────────────
# Exceptions
# ──────────────────────────────────────────────────────────────
class FetchError(Exception):
    """Raised when a data source cannot be fetched."""

    def __init__(self, source_name: str, message: str, url: str = ""):
        self.source_name = source_name
        self.url = url
        super().__init__(f"[{source_name}] {message} (url={url})")


class ParseError(Exception):
    """Raised when fetched content cannot be parsed into a DataFrame."""

    def __init__(self, source_name: str, message: str):
        self.source_name = source_name
        super().__init__(f"[{source_name}] Parse error: {message}")


# ──────────────────────────────────────────────────────────────
# Parsers
# ──────────────────────────────────────────────────────────────
def _parse_html_table(html: str, source: DataSource) -> pd.DataFrame:
    """Parse an HTML <table> from a SET-style historical quote page into a DataFrame."""
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table", class_="table")
    if table is None:
        table = soup.find("table")
    if table is None:
        raise ParseError(source.name, "No <table> found in HTML response")

    rows = table.find_all("tr")
    if len(rows) < 2:
        raise ParseError(source.name, "Table has fewer than 2 rows (no data)")

    header_row = rows[0]
    headers = [th.get_text(strip=True) for th in header_row.find_all("th")]
    if not headers:
        headers = [td.get_text(strip=True) for td in header_row.find_all("td")]

    if not headers or len(headers) < 3:
        raise ParseError(source.name, f"Could not parse table headers: {headers}")

    data: List[List[str]] = []
    for row in rows[1:]:
        cells = [td.get_text(strip=True) for td in row.find_all("td")]
        if len(cells) >= len(headers):
            data.append(cells[: len(headers)])
        elif len(cells) > 0:
            data.append(cells)

    df = pd.DataFrame(data, columns=headers[: len(data[0])] if data else headers)

    column_map = {
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Change": "change",
        "%Change": "pct_change",
        "% Change": "pct_change",
        "Volume": "volume",
        "Value": "value",
        "Bid": "bid_rate",
        "Ask": "ask_rate",
    }
    df.rename(columns={c: column_map.get(c, c) for c in df.columns}, inplace=True)

    return df


def _parse_json_api(response_json: Dict[str, Any], source: DataSource) -> pd.DataFrame:
    """Parse a CKAN-style JSON API response into a DataFrame."""
    records = None
    if "result" in response_json:
        result = response_json["result"]
        if isinstance(result, dict):
            records = result.get("records", result.get("items", []))
        elif isinstance(result, list):
            records = result
    elif "data" in response_json:
        records = response_json["data"]
    elif isinstance(response_json, list):
        records = response_json

    if records is None:
        records = response_json.get("records", [])

    if not records:
        logger.warning("JSON API returned 0 records for '%s'", source.name)
        return pd.DataFrame()

    return pd.DataFrame(records)


def _parse_excel(content: bytes, source: DataSource) -> pd.DataFrame:
    """Parse an Excel (.xlsx) file from bytes into a DataFrame."""
    import io

    try:
        df = pd.read_excel(io.BytesIO(content), engine="openpyxl")
    except Exception as exc:
        raise ParseError(source.name, f"Excel parsing failed: {exc}") from exc
    return df


def _parse_oaq_response(
    response_json: Dict[str, Any], source: DataSource, target_symbol: str
) -> Optional[Dict[str, Any]]:
    """Extract a single security's data from the SET OAQ API response.

    The OAQ security-stat response has the structure:
    {
        "tradeDate": "2024-01-02",
        "dataRound": "EOD1",
        "securityList": [ { "symbol": "AOT", ... }, ... ],
        "indexList": [ { "indexName": "SET Index", ... }, ... ]
    }
    """
    # Handle security list (stocks)
    security_list = response_json.get("securityList", [])
    for sec in security_list:
        if sec.get("symbol") == target_symbol:
            sec["date"] = response_json.get("tradeDate", "")
            return sec

    # Handle index list (SET Index)
    index_list = response_json.get("indexList", [])
    for idx in index_list:
        if target_symbol in idx.get("indexName", ""):
            idx["date"] = response_json.get("tradeDate", "")
            idx["symbol"] = idx.get("indexName", "")
            return idx

    return None


def _generate_trading_dates(start_year: int, end_date: Optional[date] = None) -> List[str]:
    """Generate a list of potential trading dates (weekdays Mon-Fri) as YYYY-MM-DD strings.

    Parameters
    ----------
    start_year : int
        First year to include (Jan 1 of that year).
    end_date : date, optional
        Last date to include (defaults to today).

    Returns
    -------
    list of str
        Date strings in YYYY-MM-DD format for all weekdays in range.
    """
    if end_date is None:
        end_date = date.today()
    start_date = date(start_year, 1, 1)

    dates: List[str] = []
    current = start_date
    while current <= end_date:
        if current.weekday() < 5:  # Monday=0, Friday=4
            dates.append(current.isoformat())
        current += timedelta(days=1)
    return dates


def _fetch_oaq_api(
    source: DataSource,
    cache_dir: Path,
    force: bool,
) -> pd.DataFrame:
    """Fetch data from the SET OAQ (Online Asset Quotation) API.

    This API requires an api-key header obtained from SET Marketplace.
    If the key is not set via the SET_API_KEY env variable, falls back
    to manual file loading.

    The fetcher iterates through all trading days from source.data_start_year
    to the current date, requesting data for each day and caching individual
    responses.
    """
    # ── Check for API key ──────────────────────────────────
    api_key = source.api_key
    if not api_key:
        logger.warning(
            "No API key found for '%s'. "
            "Set the %s environment variable, or manually download data "
            "to data/raw/%s/",
            source.name,
            source.api_key_env_var or "SET_API_KEY",
            source.name,
        )
        raise FetchError(
            source.name,
            f"No API key available. Set {source.api_key_env_var or 'SET_API_KEY'} "
            f"environment variable or place a CSV file in data/raw/{source.name}/",
        )

    # ── Determine target symbol ────────────────────────────
    if source.name == "set_index":
        target_symbol = "SET Index"
    else:
        target_symbol = source.params.get("symbol", source.name.replace("set_", "").upper())

    # ── Prepare headers with API key ──────────────────────
    headers = dict(source.headers)
    headers["api-key"] = api_key
    auth_header_name = "api-key"

    # ── Generate trading dates ─────────────────────────────
    trading_dates = _generate_trading_dates(source.data_start_year)
    logger.info("OAQ: %s potential trading dates for '%s'", len(trading_dates), source.name)

    # ── Fetch data for each date ───────────────────────────
    records: List[Dict[str, Any]] = []
    cache_root = cache_dir / "raw" / source.name
    cache_root.mkdir(parents=True, exist_ok=True)
    errors = 0
    max_errors = 50  # tolerate N consecutive errors before aborting

    for i, trade_date in enumerate(trading_dates):
        # Check cache first
        cache_file = cache_root / f"{trade_date}.json"
        if not force and cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    resp_data = json.load(f)
                record = _parse_oaq_response(resp_data, source, target_symbol)
                if record:
                    records.append(record)
                continue
            except (json.JSONDecodeError, KeyError):
                pass  # Corrupt cache, re-fetch

        # Rate limit
        if i > 0:
            time.sleep(RATE_LIMIT_DELAY)

        params = {"tradeDate": trade_date}

        try:
            session = _get_session()
            resp = session.get(
                source.url,
                params=params,
                headers=headers,
                timeout=DEFAULT_TIMEOUT,
            )
            resp.raise_for_status()
            resp_data = resp.json()
        except requests.RequestException as exc:
            errors += 1
            if errors >= max_errors:
                logger.error(
                    "OAQ: Aborting after %d consecutive errors on date %s",
                    max_errors,
                    trade_date,
                )
                break
            logger.debug("OAQ: Error on %s: %s (error %d/%d)", trade_date, exc, errors, max_errors)
            continue
        except json.JSONDecodeError:
            errors += 1
            if errors >= max_errors:
                break
            continue

        # Cache the raw response
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(resp_data, f, ensure_ascii=False)

        # Extract record
        record = _parse_oaq_response(resp_data, source, target_symbol)
        if record:
            records.append(record)
            errors = 0  # Reset error counter on success

        if (i + 1) % 100 == 0:
            logger.info(
                "OAQ: %d/%d dates processed, %d records found",
                i + 1,
                len(trading_dates),
                len(records),
            )

    if not records:
        logger.warning(
            "OAQ: No data found for '%s' across %d dates", source.name, len(trading_dates)
        )
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # ── Map OAQ field names to canonical names ─────────────
    oaq_to_canonical = {
        "symbol": "symbol",
        "date": "date",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "last": "close",
        "change": "change",
        "pctChange": "pct_change",
        "pct_change": "pct_change",
        "volume": "volume",
        "totalVolume": "volume",
        "value": "value",
        "totalValue": "value",
        "indexName": "index_name",
    }
    df.rename(columns={c: oaq_to_canonical.get(c, c) for c in df.columns}, inplace=True)

    # ── Keep only defined fields if they exist ─────────────
    defined_fields = [f.name for f in source.fields if f.name in df.columns]
    df = df[defined_fields] if defined_fields else df

    # ── Parse date column ─────────────────────────────────
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df.dropna(subset=["date"], inplace=True)
        df.sort_values("date", inplace=True)

    logger.info("OAQ: Fetched %d records for '%s'", len(df), source.name)
    return df


# ──────────────────────────────────────────────────────────────
# Fetch implementation
# ──────────────────────────────────────────────────────────────
def _do_http_get(
    url: str,
    params: Dict[str, str],
    headers: Dict[str, str],
    timeout: int = DEFAULT_TIMEOUT,
) -> requests.Response:
    session = _get_session()
    resp = session.get(url, params=params, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp


def _fetch_html_table(
    source: DataSource,
    cache_dir: Path,
    force: bool,
) -> pd.DataFrame:
    """Fetch an HTML table (e.g., SET historical quotes) with pagination."""
    all_dfs: List[pd.DataFrame] = []
    page = 1

    while True:
        params = dict(source.params)
        params["page"] = str(page)
        cache_path = cache_dir / "raw" / source.name / f"page_{page:04d}.html"
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        if not force and cache_path.exists():
            raw_html = cache_path.read_text(encoding="utf-8")
            logger.debug("Cache HIT for %s page %d", source.name, page)
        else:
            logger.info("Fetching %s page %d from %s", source.name, page, source.url)
            time.sleep(RATE_LIMIT_DELAY)
            try:
                resp = _do_http_get(source.url, params, source.headers)
                raw_html = resp.text
            except requests.RequestException as exc:
                raise FetchError(
                    source.name,
                    f"HTTP request failed on page {page}: {exc}",
                    url=source.url,
                ) from exc
            cache_path.write_text(raw_html, encoding="utf-8")

        df_page = _parse_html_table(raw_html, source)
        if df_page.empty:
            break

        all_dfs.append(df_page)

        if len(df_page) < 15:
            break
        page += 1

    if not all_dfs:
        return pd.DataFrame()

    df = pd.concat(all_dfs, ignore_index=True)
    df.drop_duplicates(subset=["date"], inplace=True) if "date" in df.columns else None
    return df


def _fetch_json_api(
    source: DataSource,
    cache_dir: Path,
    force: bool,
) -> pd.DataFrame:
    """Fetch data from a JSON API (e.g., data.go.th CKAN API)."""
    cache_path = cache_dir / "raw" / source.name / "response.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if not force and cache_path.exists():
        logger.debug("Cache HIT for %s", source.name)
        with open(cache_path, "r", encoding="utf-8") as f:
            response_json = json.load(f)
    else:
        logger.info("Fetching %s from %s", source.name, source.url)
        time.sleep(RATE_LIMIT_DELAY)
        try:
            resp = _do_http_get(source.url, source.params, source.headers)
            response_json = resp.json()
        except requests.RequestException as exc:
            raise FetchError(
                source.name, f"JSON API request failed: {exc}", url=source.url
            ) from exc
        except json.JSONDecodeError as exc:
            raise ParseError(source.name, f"Response was not valid JSON: {exc}") from exc
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(response_json, f, ensure_ascii=False, indent=2)

    df = _parse_json_api(response_json, source)
    return df


def _fetch_excel_download(
    source: DataSource,
    cache_dir: Path,
    force: bool,
) -> pd.DataFrame:
    """Download and parse an Excel file (e.g., MOTS tourism statistics)."""
    cache_path = cache_dir / "raw" / source.name / "data.xlsx"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if not force and cache_path.exists():
        logger.debug("Cache HIT for %s", source.name)
        content = cache_path.read_bytes()
    else:
        logger.info("Downloading %s from %s", source.name, source.url)
        time.sleep(RATE_LIMIT_DELAY)
        try:
            resp = _do_http_get(source.url, source.params, source.headers)
            content = resp.content
        except requests.RequestException as exc:
            raise FetchError(
                source.name,
                f"Excel download failed: {exc}. Try manual download to {cache_path}.",
                url=source.url,
            ) from exc
        cache_path.write_bytes(content)

    df = _parse_excel(content, source)
    return df


def _fetch_manual_file(
    source: DataSource,
    cache_dir: Path,
    force: bool,
) -> pd.DataFrame:
    """Load a user-provided file from data/raw/<source_name>/."""
    data_dir = cache_dir / "raw" / source.name
    if not data_dir.exists():
        raise FetchError(
            source.name,
            f"No data found in {data_dir}. "
            f"Please manually download the file from {source.url} "
            f"and place it in {data_dir}/",
        )

    if source.frequency == "monthly":
        pattern = "*.xlsx"
    else:
        pattern = "*"

    files = list(data_dir.glob(pattern))
    files = [f for f in files if f.is_file() and f.suffix in (".xlsx", ".xls", ".csv", ".CSV")]

    if not files:
        raise FetchError(
            source.name,
            f"No Excel/CSV files found in {data_dir}. "
            f"Place a file downloaded from {source.url} there.",
        )

    latest = max(files, key=lambda p: p.stat().st_mtime)
    logger.info("Loading manual file for %s: %s", source.name, latest.name)

    if latest.suffix in (".xlsx", ".xls"):
        df = pd.read_excel(latest, engine="openpyxl")
    elif latest.suffix in (".csv", ".CSV"):
        df = pd.read_csv(latest)
    else:
        raise FetchError(source.name, f"Unsupported file format: {latest.suffix}")

    return df


# ──────────────────────────────────────────────────────────────
# Fetcher dispatcher
# ──────────────────────────────────────────────────────────────
FETCH_STRATEGY_MAP = {
    "html_table": _fetch_html_table,
    "json_api": _fetch_json_api,
    "json_api_oaq": _fetch_oaq_api,
    "excel_download": _fetch_excel_download,
    "manual_file": _fetch_manual_file,
}


def fetch_source(
    source: DataSource,
    cache_dir: Path,
    force: bool = False,
) -> pd.DataFrame:
    """
    Fetch data for a single DataSource.

    Parameters
    ----------
    source : DataSource
        The data source descriptor.
    cache_dir : Path
        Directory for caching raw HTTP responses.
    force : bool, default=False
        If True, bypass cache and re-download.

    Returns
    -------
    pd.DataFrame
        Parsed data with columns corresponding to source.fields.
    """
    strategy = source.fetch_strategy
    fetcher_fn = FETCH_STRATEGY_MAP.get(strategy)

    if fetcher_fn is None:
        raise ValueError(
            f"Unknown fetch strategy '{strategy}' for source '{source.name}'. "
            f"Supported: {list(FETCH_STRATEGY_MAP)}"
        )

    logger.info("Fetching '%s' via strategy '%s'", source.name, strategy)

    try:
        df = fetcher_fn(source, cache_dir, force)
        logger.info("Fetched %s: %d rows, %d columns", source.name, len(df), len(df.columns))
        return df
    except (FetchError, ParseError) as exc:
        logger.error("Failed to fetch '%s': %s", source.name, exc)
        raise


def save_raw_response(
    source: DataSource,
    response_content: bytes,
    cache_dir: Path,
    suffix: str = "",
) -> Path:
    """Save raw response bytes to the cache directory for reproducibility."""
    ext = suffix or ".bin"
    dst = cache_dir / "raw" / source.name / f"raw_{source.name}{ext}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(response_content)
    return dst


def get_cache_age_days(source: DataSource) -> int:
    """Return the maximum cache age in days for a source based on its frequency."""
    return CACHE_EXPIRY_DAYS.get(source.frequency, 7)
