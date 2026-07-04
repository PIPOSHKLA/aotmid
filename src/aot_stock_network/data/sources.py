"""
Data source definitions and registry for the AOT Stock Network project.

Each DataSource documents an official data source with its:
  - Institution and access URL
  - Fetch strategy and expected schema
  - Field-level validation rules
  - Usage notes for reproducibility

Source documentation
--------------------
SET_AOT       : Stock Exchange of Thailand — AOT daily trading data
SET_INDEX     : Stock Exchange of Thailand — SET Index daily data
MOTS_TOURISTS : Ministry of Tourism and Sports — intl. tourist arrivals
MOTS_REVENUE  : Ministry of Tourism and Sports — tourism revenue
BOT_USDTHB    : Bank of Thailand — USD / THB exchange rate
BOT_POLICY_RATE: Bank of Thailand — policy interest rate
BOT_INFLATION : Bank of Thailand — headline & core CPI
DATA_GO_TH    : Thailand Open Government Data — supplementary datasets
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class Field:
    """Schema definition for a single column in a data source."""

    name: str
    dtype: str  # date, float, int, string
    description: str
    nullable: bool = False
    validation_rules: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DataSource:
    """Immutable descriptor for an official data source."""

    name: str
    display_name: str
    description: str
    institution: str
    category: str  # stock, tourism, macro
    frequency: str  # daily, monthly, quarterly, irregular
    url: str
    fetch_strategy: str
    # html_table | json_api_oaq | json_api | excel_download | manual_file
    params: Dict[str, str] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)
    fields: List[Field] = field(default_factory=list)
    notes: str = ""
    data_start_year: int = 2015
    data_end_year: Optional[int] = None
    official_identifier: str = ""
    api_key_env_var: Optional[str] = None  # name of env var for API key

    @property
    def api_key(self) -> Optional[str]:
        if self.api_key_env_var:
            return os.environ.get(self.api_key_env_var)
        return None

    @property
    def field_names(self) -> List[str]:
        return [f.name for f in self.fields]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "institution": self.institution,
            "category": self.category,
            "frequency": self.frequency,
            "url": self.url,
            "fetch_strategy": self.fetch_strategy,
            "fields": [
                {
                    "name": f.name,
                    "dtype": f.dtype,
                    "description": f.description,
                    "nullable": f.nullable,
                }
                for f in self.fields
            ],
            "notes": self.notes,
            "data_start_year": self.data_start_year,
            "official_identifier": self.official_identifier,
        }


_common_headers = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9,th;q=0.8",
}

# ──────────────────────────────────────────────────────────────
# SOURCE 1: SET AOT — AOT public company daily trading summary
# ──────────────────────────────────────────────────────────────
SET_AOT = DataSource(
    name="set_aot",
    display_name="AOT Daily Stock Price — SET",
    description=(
        "Daily trading data for Airports of Thailand Public Company Limited "
        "(AOT) listed on the Stock Exchange of Thailand. Includes open, high, "
        "low, close, change, percentage change, trading volume, and value."
    ),
    institution="Stock Exchange of Thailand (SET)",
    category="stock",
    frequency="daily",
    url="https://marketplace.set.or.th/api/public/oaq-data/security-stat",
    fetch_strategy="json_api_oaq",
    params={"tradeDate": ""},
    headers={
        "User-Agent": _common_headers["User-Agent"],
        "Accept": "application/json",
    },
    fields=[
        Field("date", "date", "Trading date"),
        Field("open", "float", "Opening price (THB/share)", validation_rules={"min": 0}),
        Field("high", "float", "Highest price (THB/share)", validation_rules={"min": 0}),
        Field("low", "float", "Lowest price (THB/share)", validation_rules={"min": 0}),
        Field("close", "float", "Closing price (THB/share)", validation_rules={"min": 0}),
        Field("change", "float", "Price change from previous close (THB)"),
        Field("pct_change", "float", "Percentage change from previous close"),
        Field("volume", "int", "Number of shares traded", validation_rules={"min": 0}),
        Field("value", "float", "Trading value (THB x1000)", validation_rules={"min": 0}),
    ],
    notes=(
        "Data retrieved from the SET OAQ (Online Asset Quotation) API, the "
        "official real-time and end-of-day data service of the Stock Exchange "
        "of Thailand.\n"
        "Endpoint: marketplace.set.or.th/api/public/oaq-data/security-stat\n"
        "Authentication: Requires api-key header obtained by registering at:\n"
        "  https://www.set.or.th/app/online-data/developer-guide/generate-api-key\n"
        "The API returns all securities for a single trade date. The fetcher\n"
        "iterates through all trading days between data_start_year and the\n"
        "current date, filtering for symbol 'AOT'.\n"
        "If the API key is not available, manually download historical data\n"
        "from the SET website and place a CSV file in data/raw/set_aot/."
    ),
    data_start_year=2015,
    official_identifier="SET:AOT",
    api_key_env_var="SET_API_KEY",
)

# ──────────────────────────────────────────────────────────────
# SOURCE 2: SET INDEX — SET benchmark index
# ──────────────────────────────────────────────────────────────
SET_INDEX = DataSource(
    name="set_index",
    display_name="SET Index — Stock Exchange of Thailand",
    description=(
        "Daily closing values and trading summary for the SET Index, the "
        "benchmark equity index of the Stock Exchange of Thailand."
    ),
    institution="Stock Exchange of Thailand (SET)",
    category="stock",
    frequency="daily",
    url="https://marketplace.set.or.th/api/public/oaq-data/index-stat",
    fetch_strategy="json_api_oaq",
    params={"tradeDate": ""},
    headers={
        "User-Agent": _common_headers["User-Agent"],
        "Accept": "application/json",
    },
    fields=[
        Field("date", "date", "Trading date"),
        Field("open", "float", "Opening index level", validation_rules={"min": 0}),
        Field("high", "float", "Intra-day high", validation_rules={"min": 0}),
        Field("low", "float", "Intra-day low", validation_rules={"min": 0}),
        Field("close", "float", "Closing index level", validation_rules={"min": 0}),
        Field("change", "float", "Point change"),
        Field("pct_change", "float", "Percentage change"),
        Field("volume", "int", "Trading volume (shares)", validation_rules={"min": 0}),
        Field("value", "float", "Trading value (THB x1000)", validation_rules={"min": 0}),
    ],
    notes=(
        "SET Index data from the OAQ index-stat API endpoint at "
        "marketplace.set.or.th. Same authentication method as set_aot.\n"
        "The index response includes all SET index series; filter for "
        "'SET Index' as the index_name field."
    ),
    data_start_year=2015,
    official_identifier="SET:SET",
    api_key_env_var="SET_API_KEY",
)

# ──────────────────────────────────────────────────────────────
# SOURCE 3: MOTS TOURISTS — international tourist arrivals
# ──────────────────────────────────────────────────────────────
MOTS_TOURISTS = DataSource(
    name="mots_tourists",
    display_name="International Tourist Arrivals — MOTS",
    description=(
        "Monthly international tourist arrivals to Thailand, aggregated and "
        "by nationality/region. Published by the Ministry of Tourism and "
        "Sports (MOTS) approximately 45 days after month-end."
    ),
    institution="Ministry of Tourism and Sports (MOTS), Thailand",
    category="tourism",
    frequency="monthly",
    url="https://www.mots.go.th/more_news.php?cid=595",
    fetch_strategy="excel_download",
    params={"cid": "595"},
    headers=_common_headers,
    fields=[
        Field("month", "date", "Calendar month (yyyy-mm)"),
        Field(
            "total_arrivals", "float", "Total international arrivals", validation_rules={"min": 0}
        ),
        Field(
            "arrivals_east_asia", "float", "Arrivals from East Asia", validation_rules={"min": 0}
        ),
        Field("arrivals_europe", "float", "Arrivals from Europe", validation_rules={"min": 0}),
        Field(
            "arrivals_americas", "float", "Arrivals from the Americas", validation_rules={"min": 0}
        ),
        Field(
            "arrivals_south_asia", "float", "Arrivals from South Asia", validation_rules={"min": 0}
        ),
        Field("arrivals_oceania", "float", "Arrivals from Oceania", validation_rules={"min": 0}),
        Field(
            "arrivals_middle_east",
            "float",
            "Arrivals from Middle East",
            validation_rules={"min": 0},
        ),
        Field("arrivals_africa", "float", "Arrivals from Africa", validation_rules={"min": 0}),
        Field("arrivals_china", "float", "Arrivals from China", validation_rules={"min": 0}),
        Field("arrivals_malaysia", "float", "Arrivals from Malaysia", validation_rules={"min": 0}),
        Field(
            "arrivals_south_korea",
            "float",
            "Arrivals from South Korea",
            validation_rules={"min": 0},
        ),
        Field("arrivals_india", "float", "Arrivals from India", validation_rules={"min": 0}),
        Field("arrivals_russia", "float", "Arrivals from Russia", validation_rules={"min": 0}),
    ],
    notes=(
        "Ministry of Tourism and Sports publishes monthly tourism statistics "
        "as Excel (.xlsx) press releases on their website. The data can also "
        "be accessed via the MOTS Statistics page. Data starts from ~2015 "
        "for the current reporting format. Use data.go.th for bulk historical "
        "downloads. NOTE: if automatic download fails, manually download the "
        "latest Excel file from the MOTS website and place in data/raw/mots_tourists/."
    ),
    data_start_year=2015,
    official_identifier="MOTS:tourist_arrivals",
)

# ──────────────────────────────────────────────────────────────
# SOURCE 4: MOTS REVENUE — tourism receipts
# ──────────────────────────────────────────────────────────────
MOTS_REVENUE = DataSource(
    name="mots_revenue",
    display_name="Tourism Revenue — MOTS",
    description=(
        "Monthly tourism revenue (in THB) generated by international visitors "
        "to Thailand. Published alongside arrival statistics by MOTS."
    ),
    institution="Ministry of Tourism and Sports (MOTS), Thailand",
    category="tourism",
    frequency="monthly",
    url="https://www.mots.go.th/more_news.php?cid=595",
    fetch_strategy="excel_download",
    params={"cid": "595"},
    headers=_common_headers,
    fields=[
        Field("month", "date", "Calendar month (yyyy-mm)"),
        Field(
            "total_revenue_mb",
            "float",
            "Total tourism revenue (million THB)",
            validation_rules={"min": 0},
        ),
        Field(
            "revenue_east_asia_mb",
            "float",
            "Revenue from East Asia (million THB)",
            validation_rules={"min": 0},
        ),
        Field(
            "revenue_europe_mb",
            "float",
            "Revenue from Europe (million THB)",
            validation_rules={"min": 0},
        ),
        Field(
            "revenue_americas_mb",
            "float",
            "Revenue from Americas (million THB)",
            validation_rules={"min": 0},
        ),
        Field(
            "revenue_south_asia_mb",
            "float",
            "Revenue from South Asia (million THB)",
            validation_rules={"min": 0},
        ),
        Field(
            "revenue_oceania_mb",
            "float",
            "Revenue from Oceania (million THB)",
            validation_rules={"min": 0},
        ),
        Field(
            "revenue_middle_east_mb",
            "float",
            "Revenue from Middle East (million THB)",
            validation_rules={"min": 0},
        ),
        Field(
            "revenue_africa_mb",
            "float",
            "Revenue from Africa (million THB)",
            validation_rules={"min": 0},
        ),
    ],
    notes=(
        "Tourism revenue figures in the same MOTS Excel publications. "
        "Reported in million THB. Note that revenue data may follow a "
        "different publication schedule than arrivals."
    ),
    data_start_year=2015,
    official_identifier="MOTS:tourism_revenue",
)

# ──────────────────────────────────────────────────────────────
# SOURCE 5: BOT USD/THB — foreign exchange rate
# ──────────────────────────────────────────────────────────────
BOT_USDTHB = DataSource(
    name="bot_usdthb",
    display_name="USD/THB Exchange Rate — Bank of Thailand",
    description=(
        "Daily mid-market USD/THB exchange rate published by the Bank of "
        "Thailand. Used as a macroeconomic control variable."
    ),
    institution="Bank of Thailand (BOT)",
    category="macro",
    frequency="daily",
    url="https://www.bot.or.th/App/BTWS_STAT/statistics/ReportPage.aspx?reportID=117&language=en",
    fetch_strategy="html_table",
    params={"reportID": "117", "language": "en"},
    headers=_common_headers,
    fields=[
        Field("date", "date", "Calendar date"),
        Field(
            "usdthb_rate",
            "float",
            "USD/THB mid-market rate",
            validation_rules={"min": 10, "max": 50},
        ),
        Field("bid_rate", "float", "Bid rate", nullable=True),
        Field("ask_rate", "float", "Ask rate", nullable=True),
    ],
    notes=(
        "Bank of Thailand Statistics page reportID=117 provides daily "
        "USD/THB exchange rates. Data available from 2002 onward. "
        "Alternative: BOT API at apigw1.bot.or.th/bot/public/Stat-API/v2/"
        " (requires registration for API key). The HTML table fallback "
        "requires parsing the ASP.NET page. If automated parsing fails, "
        "download the CSV export from the BOT statistics page."
    ),
    data_start_year=2015,
    official_identifier="BOT:FX_USDTHB",
)

# ──────────────────────────────────────────────────────────────
# SOURCE 6: BOT POLICY RATE — policy interest rate
# ──────────────────────────────────────────────────────────────
BOT_POLICY_RATE = DataSource(
    name="bot_policy_rate",
    display_name="BOT Policy Interest Rate",
    description=(
        "Bank of Thailand's policy interest rate (1-day bilateral repurchase "
        "rate). Set by the Monetary Policy Committee (MPC) at approximately "
        "8 scheduled meetings per year. Key macroeconomic indicator."
    ),
    institution="Bank of Thailand (BOT)",
    category="macro",
    frequency="irregular",
    url="https://www.bot.or.th/App/BTWS_STAT/statistics/ReportPage.aspx?reportID=191&language=en",
    fetch_strategy="html_table",
    params={"reportID": "191", "language": "en"},
    headers=_common_headers,
    fields=[
        Field("announce_date", "date", "MPC announcement date"),
        Field("effective_date", "date", "Rate effective date", nullable=True),
        Field("policy_rate", "float", "Policy rate (%)", validation_rules={"min": 0, "max": 10}),
        Field("previous_rate", "float", "Previous policy rate (%)", nullable=True),
        Field("change_bps", "float", "Change in basis points", nullable=True),
    ],
    notes=(
        "BOT Statistics page reportID=191 shows MPC meeting history with "
        "policy rate decisions. Data from ~2000 onward. HTML table parsing "
        "of ASP.NET page. Manual check recommended if HTML parsing fails."
    ),
    data_start_year=2015,
    official_identifier="BOT:policy_rate",
)

# ──────────────────────────────────────────────────────────────
# SOURCE 7: BOT INFLATION — consumer price index
# ──────────────────────────────────────────────────────────────
BOT_INFLATION = DataSource(
    name="bot_inflation",
    display_name="Consumer Price Index — Bank of Thailand",
    description=(
        "Monthly headline and core Consumer Price Index (CPI) for Thailand. "
        "Headline CPI includes all items; core CPI excludes raw food and "
        "energy. Published by BOT's statistics bureau."
    ),
    institution="Bank of Thailand (BOT)",
    category="macro",
    frequency="monthly",
    url="https://www.bot.or.th/App/BTWS_STAT/statistics/ReportPage.aspx?reportID=409&language=en",
    fetch_strategy="html_table",
    params={"reportID": "409", "language": "en"},
    headers=_common_headers,
    fields=[
        Field("month", "date", "Calendar month"),
        Field(
            "cpi_headline",
            "float",
            "Headline CPI index (2019=100)",
            validation_rules={"min": 50, "max": 150},
        ),
        Field(
            "cpi_core",
            "float",
            "Core CPI index (2019=100)",
            validation_rules={"min": 50, "max": 150},
        ),
        Field("headline_inflation_pct", "float", "Headline CPI year-on-year change (%)"),
        Field("core_inflation_pct", "float", "Core CPI year-on-year change (%)"),
    ],
    notes=(
        "BOT Statistics page reportID=409 provides monthly CPI figures. "
        "Base year 2019=100. Data available from ~2010 onward. "
        "Alternative: Trade Policy and Strategy Office (TPSO) also "
        "publishes CPI data."
    ),
    data_start_year=2015,
    official_identifier="BOT:CPI",
)

# ──────────────────────────────────────────────────────────────
# SOURCE 8: DATA.GO.TH — Thailand Open Government Data
# ──────────────────────────────────────────────────────────────
DATA_GO_TH = DataSource(
    name="data_go_th",
    display_name="Thailand Open Government Data Portal",
    description=(
        "Supplementary economic and tourism datasets from Thailand's official "
        "open data portal (data.go.th). Used for additional validation and "
        "cross-referencing of primary sources."
    ),
    institution="Digital Government Development Agency (DGA), Thailand",
    category="macro",
    frequency="various",
    url="https://catalog.data.go.th/api/3/action/datastore_search",
    fetch_strategy="json_api",
    headers={"User-Agent": _common_headers["User-Agent"]},
    params={"resource_id": "", "limit": "1000"},
    fields=[
        Field("dataset_name", "string", "Name of the dataset"),
        Field("resource_id", "string", "CKAN resource identifier"),
        Field("record_count", "int", "Number of records fetched"),
        Field("last_updated", "date", "Dataset last update date"),
    ],
    notes=(
        "data.go.th is Thailand's CKAN-based open data portal. The API "
        "endpoint at /api/3/action/datastore_search returns JSON. "
        "Useful resources for this project include:\n"
        "  - Tourism statistics (cross-ref with MOTS)\n"
        "  - Economic indicators (cross-ref with BOT)\n"
        "  - Air transport statistics\n"
        "Search for resource IDs via /api/3/action/package_list or "
        "the web interface at catalog.data.go.th.\n"
        "Rate limit: ~10 req/s. No API key required for public data."
    ),
    data_start_year=2015,
    official_identifier="DGA:data_go_th",
)

# ──────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────
SOURCE_REGISTRY: Dict[str, DataSource] = {
    s.name: s
    for s in [
        SET_AOT,
        SET_INDEX,
        MOTS_TOURISTS,
        MOTS_REVENUE,
        BOT_USDTHB,
        BOT_POLICY_RATE,
        BOT_INFLATION,
        DATA_GO_TH,
    ]
}

# Category index
SOURCES_BY_CATEGORY: Dict[str, List[DataSource]] = {}
for s in SOURCE_REGISTRY.values():
    SOURCES_BY_CATEGORY.setdefault(s.category, []).append(s)


def get_source(name: str) -> DataSource:
    """Look up a DataSource by name; raises KeyError if not found."""
    if name not in SOURCE_REGISTRY:
        valid = ", ".join(sorted(SOURCE_REGISTRY))
        raise KeyError(f"Unknown data source '{name}'. Valid sources: {valid}")
    return SOURCE_REGISTRY[name]


def list_sources(category: Optional[str] = None) -> List[str]:
    """Return source names, optionally filtered by category."""
    if category is None:
        return sorted(SOURCE_REGISTRY.keys())
    return sorted(s.name for s in SOURCES_BY_CATEGORY.get(category, []))


def source_registry_summary() -> str:
    """Return a human-readable summary of all registered sources."""
    lines = []
    lines.append(f"{'Name':<20} {'Institution':<30} {'Category':<12} {'Freq':<12}")
    lines.append("-" * 80)
    for s in SOURCE_REGISTRY.values():
        lines.append(f"{s.name:<20} {s.institution[:28]:<30} {s.category:<12} {s.frequency:<12}")
    return "\n".join(lines)
