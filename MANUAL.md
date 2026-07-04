# AOT Stock Network — Project Manual

## Table of Contents

1. [Installation Guide](#1-installation-guide)
2. [User Guide](#2-user-guide)
3. [Developer Guide](#3-developer-guide)
4. [Future Improvements](#4-future-improvements)

---

## 1. Installation Guide

### 1.1 Prerequisites

- **Python 3.10 or later** (tested on 3.12)
- **Git** (optional, for version control)
- **pip** (Python package manager, included with Python 3.10+)
- **4 GB RAM minimum** (8 GB recommended for LSTM training)

### 1.2 Environment Setup

**Option A — pip (recommended for Windows)**

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate     # Windows
# source .venv/bin/activate  # Linux/macOS

# Upgrade pip
python -m pip install --upgrade pip

# Install project with dev dependencies
pip install -e ".[dev]"

# (Optional) Install all ML dependencies
pip install -e ".[ml]"
```

**Option B — Conda**

```bash
conda env create -f environment.yml
conda activate aot-stock-network
pip install -e .
```

### 1.3 Data Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your SET API key
# SET_API_KEY=your_key_here
```

If you do not have a SET API key, the dashboard runs in **demo mode** using synthetic sample data — no key required.

### 1.4 Verify Installation

```bash
# Check imports
python -c "import aot_stock_network; print('OK')"

# Run tests
pytest tests/ -v

# Start the dashboard
streamlit run Home.py
```

### 1.5 Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'src'` | Run `pip install -e .` from project root |
| TensorFlow not detecting GPU | Install `tensorflow[and-cuda]` or use CPU-only |
| Prophet installation fails | Install `pystan` first: `pip install pystan==3.10.0` |
| LightGBM warning about OpenMP | Ignore — CPU training works fine without OpenMP on Windows |

---

## 2. User Guide

### 2.1 Launching the Dashboard

```bash
streamlit run Home.py
```

This opens a browser at `http://localhost:8501` with 9 pages organized in 5 sidebar sections.

### 2.2 Pages Reference

#### Home
- **Purpose**: Project overview, key metrics (data span, variable count), navigation hub
- **Usage**: Landing page — read the research context and scroll data preview

#### Data Explorer
- **Purpose**: Browse raw aligned data, filter by date range and columns
- **Usage**: Select columns to display, adjust date slider, search for specific values
- **Export**: Click "Download CSV" to save filtered data

#### EDA (Exploratory Data Analysis)
- **Purpose**: 14 publication-quality figures covering distributions, time series, seasonality, outliers
- **Tabs**: Distributions | Time Series | Seasonality | Outliers | Decomposition
- **Usage**: Each tab shows a subset of plots. Hover for details. Use toolbar to zoom.
- **Export**: Figures export as SVG via the Matplotlib export button

#### Correlation
- **Purpose**: Pearson/Spearman heatmap, target correlation bar chart, interactive scatter matrix
- **Usage**: Switch method with radio button. Scatter matrix supports subset selection.

#### Social Network Analysis
- **Purpose**: Interactive network graph of factor relationships
- **Features**:
  - **Node size** ∝ Degree centrality
  - **Node colour** = Louvain community
  - **Edge width** ∝ |correlation|
  - **Hover** shows all 5 centrality values
- **Sidebar controls**: Select correlation method, adjust threshold, choose edge options
- **Export**: SVG, PNG (Plotly toolbar), GraphML (Gephi/Cytoscape)

#### Machine Learning
- **Purpose**: Train and compare up to 8 model families
- **Workflow**:
  1. Select models (checkboxes)
  2. Choose features (multiselect defaults to all numeric)
  3. Toggle hyperparameter tuning (on by default)
  4. Click "Train Models"
- **Results tabs**: Comparison Table | Predictions | Residuals | Feature Importance | SHAP
- **Note**: LSTM and tuning take longer. First run with 2–3 fast models (LR, RF, ARIMA).

#### Forecast
- **Purpose**: Generate multi-step forecasts with confidence intervals
- **Workflow**:
  1. Set forecast horizon (3–24 months)
  2. Select model (default: best from ML page)
  3. Configure what-if scenarios (adjust tourist arrivals, FX rate)
- **Output**: Time-series plot with CI shading, numerical forecast table

#### Report
- **Purpose**: Structured 8-section research report with dynamic findings
- **Sections**: Introduction → Methodology → Data → Network → ML → Forecast → Conclusion → References
- **Export**: "Export as Markdown" button, "Export as PDF" (print from browser)

#### Download Center
- **Purpose**: Centralized export hub
- **Downloads**: Aligned data (CSV), feature dataset (CSV), all figures (ZIP), report (Markdown)

### 2.3 Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+F` | Search within interactive Plotly graphs |
| `Shift+Scroll` | Horizontal scroll on plots |
| `R` (Plotly) | Reset zoom |
| `Ctrl+Shift+S` | Save plot as SVG (Plotly toolbar) |

---

## 3. Developer Guide

### 3.1 Architecture Overview

```
Home.py (entry point)
  └── st.Page + st.navigation
      ├── Overview      → home_page.py
      ├── Analysis      → data_explorer_page.py, eda_page.py, correlation_page.py
      ├── Network       → social_network_page.py
      ├── Prediction    → ml_page.py, forecast_page.py
      └── Output        → report_page.py, download_page.py
```

Modules are loosely coupled through well-defined interfaces:

| Module | Key Class(es) | Input | Output |
|--------|---------------|-------|--------|
| `data.sources` | `DataSource` (frozen dataclass) | Source name | Source descriptor |
| `data.fetcher` | `fetch_source()` | DataSource | `pd.DataFrame` |
| `data.loader` | `DataLoader` | Source name | `pd.DataFrame` |
| `data.validator` | `validate_dataframe()` | DataFrame + DataSource | `ValidationResult` |
| `preprocessing` | `PreprocessingPipeline` | Raw CSVs | Aligned monthly `pd.DataFrame` |
| `feature_engineering` | `FeatureEngineer` | Aligned df | Feature-enriched `pd.DataFrame` |
| `network_analysis` | `NetworkBuilder`, `NetworkAnalyzer`, `NetworkVisualizer` | Feature df | `nx.Graph`, `NetworkMetrics` |
| `prediction` | `PredictionPipeline`, `PredictionResults` | Feature df | `ModelResult` list |
| `visualization` | `EDAVisualizer`, standalone plot functions | Data + column spec | `matplotlib.Figure` |

### 3.2 Adding a New Model

1. Create a wrapper class inheriting from `_BaseModelWrapper` in `prediction.py`
2. Implement `fit()`, `predict()`, optionally `tune()` and `get_feature_importance()`
3. Add the wrapper to `WRAPPER_CLASSES` dict
4. Add a grid to `MODEL_GRIDS` dict
5. Add a label to `MODEL_LABELS` dict
6. Done — the dashboard ML page auto-discovers it

### 3.3 Adding a New Data Source

1. Define a `DataSource` frozen dataclass in `data/sources.py`
2. Add it to the `SOURCE_REGISTRY` dict
3. Implement a fetch function in `data/fetcher.py` if needed (or reuse one of the 5 existing strategies)
4. Add aggregation rules in `preprocessing.py` (if daily/irregular)
5. Add column prefix mapping in `preprocessing.py`

### 3.4 Coding Conventions

- **Type hints**: Required for all public functions and methods. Use `from __future__ import annotations`.
- **Docstrings**: Google/NumPy style with `Parameters`, `Returns`, and `Transformation documented` sections.
- **Naming**: `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_CASE` for constants.
- **Logging**: Use `logging.getLogger(__name__)` — never `print()`.
- **Error handling**: Raise specific exceptions (`FetchError`, `ParseError`) with descriptive messages.
- **Tests**: Place in `tests/` mirroring the `src/` structure. Use `pytest`.

### 3.5 Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src/aot_stock_network --cov-report=term-missing

# Specific module
pytest tests/test_preprocessing.py -v
```

### 3.6 Linting and Type Checking

```bash
# Format code
black src/ tests/

# Lint
ruff check src/ tests/

# Type check
mypy src/
```

### 3.7 Build and Package

```bash
# Build wheel
python -m build

# Install locally (editable)
pip install -e .

# Upload to PyPI (requires credentials)
python -m twine upload dist/*
```

### 3.8 Caching Strategy

| Cache Type | Location | Expiry | Purpose |
|-----------|----------|--------|---------|
| HTTP response | `data/.cache/raw/<source>/` | Frequency-dependent | Avoid re-downloading |
| Processed CSV | `data/processed/<source>.csv` | Manual `clear_cache()` | Avoid re-processing |
| Data catalog | `data/data_catalog.json` | Persistent | Track fetch history |
| Streamlit | Via `@st.cache_data` with TTL | 1 hour | Dashboard performance |

---

## 4. Future Improvements

### Short-term (sprintable)

- **Higher-frequency data**: Incorporate weekly or daily data for more responsive forecasts
- **Additional factors**: Oil prices, geopolitical risk index, China economic indicators, US federal funds rate
- **Automated API key registration**: Script to guide users through SET OAQ registration
- **Multi-target prediction**: Jointly forecast AOT + SET Index for portfolio context
- **Model persistence**: Save/load trained models via joblib for reuse across sessions

### Medium-term

- **Real-time dashboard**: Deploy on Streamlit Cloud / AWS with scheduled daily data refresh
- **Portfolio integration**: Link forecast output to a simple portfolio simulator (e.g., "what if we buy/sell based on model signal")
- **Cross-validation across time periods**: Evaluate model stability by backtesting across multiple train/test splits
- **Ensemble of networks**: Compare graphs built with Pearson, Spearman, MI side by side
- **Sentiment analysis**: Scrape news headlines about AOT and incorporate NLP-based sentiment features

### Long-term

- **Causal network**: Use Granger causality or PCMI to build directed acyclic graphs instead of undirected correlation
- **Deep graph learning**: Replace manual centrality features with a Graph Neural Network (GNN) that learns node embeddings end-to-end
- **Automated reporting**: Generate a PDF research paper with embedded figures and statistical tables
- **Multi-stock analysis**: Extend beyond AOT to all 50 SET50 stocks for a full market network analysis
- **Active learning**: Online learning framework that updates the model as new monthly data arrives

### Known Limitations

- **No real data without SET API key**: The OAQ API is behind a free registration. Automated access requires a key.
- **Synthetic data in demo mode**: Generated from a random walk with injected correlations — not suitable for real trading decisions.
- **LSTM training time**: Grid search over LSTM hyperparameters can take 5+ minutes on CPU.
- **SHAP speed**: SHAP TreeExplainer is fast for XGBoost/LightGBM but slower for Random Forest (use `shap.sample` for large feature sets).
- **Single-threaded fetcher**: The OAQ API fetcher iterates sequentially through trading days — parallelization would speed up initial fetch.
