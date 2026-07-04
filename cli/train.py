"""Train and evaluate prediction models."""

from __future__ import annotations

from argparse import Namespace, _SubParsersAction
from pathlib import Path

from loguru import logger


def register_train_parser(sub: _SubParsersAction) -> None:
    p = sub.add_parser("train", help="Train prediction models")
    p.add_argument(
        "--models",
        "-m",
        nargs="+",
        default=["lr", "rf", "xgb", "lgb", "cat", "arima", "prophet", "lstm"],
        help="Model names to train",
    )
    p.add_argument("--target", default="aot_close", help="Target column name")
    p.add_argument("--tune", action="store_true", help="Enable hyperparameter tuning")
    p.add_argument("--shap", action="store_true", help="Calculate SHAP values")
    p.add_argument(
        "--output", "-o", type=Path, default=None, help="Directory to save trained models"
    )
    p.add_argument("--data", "-d", type=Path, default=None, help="Path to preprocessed feature CSV")


def handle_train(args: Namespace) -> None:
    from aot_stock_network.data.loader import DataLoader
    from aot_stock_network.feature_engineering import FeatureEngineer
    from aot_stock_network.prediction import PredictionPipeline

    logger.info("Loading data for training")
    dl = DataLoader()
    df = dl.load_combined()

    logger.info("Engineering features")
    fe = FeatureEngineer()
    features = fe.create_all_features(df)

    logger.info("Initialising pipeline (target={})", args.target)
    pipe = PredictionPipeline(
        df=features,
        target_col=args.target,
        test_months=12,
        val_months=6,
    )
    results = pipe.run(
        models=args.models,
        tune=args.tune,
        calc_shap=args.shap,
    )

    logger.info("Training complete — {} models evaluated", len(results.results))
    for r in results.results:
        name = r.model_name
        val_rmse = r.val_metrics.rmse if r.val_metrics else None
        test_rmse = r.test_metrics.rmse if r.test_metrics else None
        logger.info(
            "  {:<12} val RMSE={:<10.4f} test RMSE={:<10.4f}", name, val_rmse or 0, test_rmse or 0
        )

    # Save trained models
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        for r in results.results:
            path = args.output / f"{r.model_name}.pkl"
            import pickle

            with open(path, "wb") as f:
                pickle.dump(r, f)
        logger.info("Models saved to {}", args.output)
