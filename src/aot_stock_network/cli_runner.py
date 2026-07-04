"""Console-script entry points for ``aot-fetch``, ``aot-train``, etc."""

from __future__ import annotations

import sys

from loguru import logger

from aot_stock_network.logging_setup import setup_logging
from aot_stock_network.seed import set_seed


def _init() -> None:
    setup_logging()
    set_seed()


def fetch() -> None:
    _init()
    from aot_stock_network.data.loader import DataLoader

    logger.info("Fetching all data sources")
    DataLoader().fetch_all()
    logger.info("Fetch complete")


def train() -> None:
    _init()
    from aot_stock_network.data.loader import DataLoader
    from aot_stock_network.feature_engineering import FeatureEngineer
    from aot_stock_network.prediction import PredictionPipeline

    logger.info("Loading data")
    df = DataLoader().load_combined()
    features = FeatureEngineer().create_all_features(df)

    pipe = PredictionPipeline(df=features, target_col="aot_close")
    results = pipe.run(models=["lr", "rf", "xgb"], tune=False, calc_shap=False)
    logger.info("Training complete — {} models", len(results.results))


def preprocess() -> None:
    _init()
    from aot_stock_network.data.loader import DataLoader
    from aot_stock_network.preprocessing import PreprocessingPipeline

    logger.info("Preprocessing data")
    df = DataLoader().load_combined()
    pipe = PreprocessingPipeline()
    df = pipe.interpolate(df)
    out = DataLoader().processed_dir / "preprocessed.csv"
    df.to_csv(out)
    logger.info("Saved {}", out)


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "dashboard"
    getattr(sys.modules[__name__], cmd, lambda: None)()
