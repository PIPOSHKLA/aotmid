# AOT Stock Network

Social Network Analysis of Factors Influencing AOT Stock Price — a graduate research project combining graph theory, machine learning, and interactive dashboards.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.58-red)](https://streamlit.io/)
[![NetworkX](https://img.shields.io/badge/networkx-3.0-green)](https://networkx.org/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

---

## Overview

This project investigates how macroeconomic and market factors collectively influence the stock price of Airports of Thailand PCL (AOT). Rather than examining pairwise correlations in isolation, we adopt a **Social Network Analysis (SNA)** framework to model factor interdependencies as a weighted graph, then use the resulting structural insights to inform predictive models.

### Pipeline

```
Data Fetching → Preprocessing → Feature Engineering → Network Analysis → ML Prediction → Dashboard
    4 official sources      28 features        9 nodes, 5 centrality    8 models          Streamlit
    (SET, MOTS, BOT, DGA)                     metrics, communities     + SHAP             9 pages
```

---

## Quick Start

```bash
# 1. Clone and install
pip install -e ".[dev]"

# 2. Start the dashboard
streamlit run Home.py

# 3. (Optional) Fetch real data — requires SET_API_KEY
#    cp .env.example .env  # then edit SET_API_KEY=
make fetch
```

---

## Project Structure

```
AOTmid/
├── Home.py                           # Streamlit entry point
├── src/aot_stock_network/
│   ├── __init__.py                   # Package exports
│   ├── data/
│   │   ├── sources.py                # DataSource definitions (8 sources)
│   │   ├── fetcher.py                # HTTP fetching + caching
│   │   ├── validator.py              # Schema + range validation
│   │   └── loader.py                 # DataLoader orchestrator
│   ├── preprocessing.py              # Cleaning, alignment, outlier treatment
│   ├── feature_engineering.py        # 28 feature creation
│   ├── network_analysis.py           # NetworkX graph, centrality, communities
│   ├── prediction.py                 # 8 model families + SHAP
│   ├── visualization.py              # 14 publication-quality EDA figures
│   └── dashboard/                    # Streamlit multipage app
│       ├── utils.py                  # CSS, data loading, shared components
│       ├── home_page.py              # Project overview
│       ├── data_explorer_page.py     # Browse/filter data
│       ├── eda_page.py               # Distributions, time series, outliers
│       ├── correlation_page.py       # Heatmaps, scatter matrix
│       ├── social_network_page.py    # Interactive network graph
│       ├── ml_page.py                # Train/compare 8 models
│       ├── forecast_page.py          # Predictions + what-if scenarios
│       ├── report_page.py            # Structured research report
│       └── download_page.py          # Batch export
├── config/data_sources.yaml          # Source configuration
├── pyproject.toml                    # Package metadata + dependencies
├── Makefile                          # Common tasks
└── .env.example                      # Environment template
```

---

## Data Sources

| Source | Institution | Data | Frequency |
|--------|-------------|------|-----------|
| SET OAQ | Stock Exchange of Thailand | AOT price, SET Index | Daily |
| MOTS | Ministry of Tourism & Sports | Tourist arrivals, revenue | Monthly |
| BOT | Bank of Thailand | USD/THB, policy rate, CPI | Daily/Monthly |
| data.go.th | DGA Thailand | Supplementary indicators | Various |

All sources are officially published and freely accessible. SET OAQ requires a free API key.

---

## Key Findings

| Finding | Detail |
|---------|--------|
| **Network** | 2 Louvain communities: market (AOT, SET, CPI) vs macro (tourists, FX, policy rate) |
| **Bridge variable** | USD/THB — highest betweenness centrality (0.40) |
| **Best model** | Random Forest — test R² = 0.253 |
| **Top predictors** | Tourist arrivals, USD/THB rate, lagged AOT close |

---

## License

MIT
