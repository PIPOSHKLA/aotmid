"""Dashboard package for the AOT Stock Network Streamlit application."""

from aot_stock_network.dashboard import utils
from aot_stock_network.dashboard.correlation_page import show as show_correlation
from aot_stock_network.dashboard.data_explorer_page import show as show_data_explorer
from aot_stock_network.dashboard.download_page import show as show_download
from aot_stock_network.dashboard.eda_page import show as show_eda
from aot_stock_network.dashboard.forecast_page import show as show_forecast
from aot_stock_network.dashboard.home_page import show as show_home
from aot_stock_network.dashboard.ml_page import show as show_ml
from aot_stock_network.dashboard.report_page import show as show_report
from aot_stock_network.dashboard.social_network_page import show as show_social_network

__all__ = [
    "utils",
    "show_home",
    "show_data_explorer",
    "show_eda",
    "show_correlation",
    "show_social_network",
    "show_ml",
    "show_forecast",
    "show_report",
    "show_download",
]
