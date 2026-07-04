from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

from loguru import logger

from aot_stock_network.config import settings


class InterceptHandler(logging.Handler):
    """Redirect standard-library logs to Loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        logger_opt = logger.opt(depth=7, exception=record.exc_info)
        logger_opt.log(record.levelno, record.getMessage())


def setup_logging(
    *,
    level: Optional[str] = None,
    json_format: bool = False,
    log_file: Optional[Path] = None,
) -> None:
    """Configure structured logging via Loguru.

    Parameters
    ----------
    level : str, optional
        Log level (default: settings.log_level).
    json_format : bool
        Whether to emit JSON-structured logs (for production/ELK).
    log_file : Path, optional
        Path to rotatable log file (default: settings.logs_dir / "app.log").
    """
    level = level or settings.log_level
    log_file = log_file or settings.logs_dir / "app.log"

    logger.remove()

    # Console sink (human-readable)
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>",
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    # File sink (rotatable, optionally JSON)
    if json_format:
        logger.add(
            str(log_file),
            level=level,
            format="{time} | {level} | {name}:{function}:{line} | {message}",
            rotation="10 MB",
            retention=settings.log_backup_count,
            serialize=True,
        )
    else:
        logger.add(
            str(log_file),
            level=level,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
            rotation="10 MB",
            retention=settings.log_backup_count,
            backtrace=True,
            diagnose=False,
        )

    # Redirect stdlib loggers
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    for name in (
        "urllib3",
        "matplotlib",
        "tensorflow",
        "prophet",
        "PIL",
        "h5py",
        "google",
        "botocore",
    ):
        logging.getLogger(name).handlers = [InterceptHandler()]
        logging.getLogger(name).propagate = False
