"""Fetch data from all configured sources."""

from __future__ import annotations

from argparse import Namespace, _SubParsersAction
from pathlib import Path

from loguru import logger


def register_fetch_parser(sub: _SubParsersAction) -> None:
    p = sub.add_parser("fetch", help="Fetch data from all configured sources")
    p.add_argument(
        "--source", "-s", default=None, help="Specific source name to fetch (default: all)"
    )
    p.add_argument(
        "--output", "-o", type=Path, default=None, help="Output directory for fetched data"
    )
    p.add_argument("--force", "-f", action="store_true", help="Re-fetch even if cached")


def handle_fetch(args: Namespace) -> None:
    kwargs = {}
    if args.source:
        kwargs["sources"] = [args.source]
    if args.output:
        kwargs["output_dir"] = args.output
    if hasattr(args, "force"):
        kwargs["force"] = args.force

    logger.info("Fetching data (source={})", args.source or "all")
    from aot_stock_network.data.loader import DataLoader  # noqa: E402

    dl = DataLoader()
    dl.fetch_all(**kwargs)
    logger.info("Data fetch complete")
