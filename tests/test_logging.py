"""Tests for logging configuration."""

from __future__ import annotations

import logging
from pathlib import Path

from loguru import logger


class TestLoggingSetup:
    """Loguru intercepts stdlib loggers correctly."""

    def test_loguru_captures_stdlib_warning(self, tmp_path: Path) -> None:
        from aot_stock_network.logging_setup import setup_logging

        log_file = tmp_path / "test.log"
        setup_logging(level="DEBUG", log_file=log_file)

        logging.getLogger("test_logger").warning("test stdlib message")

        lines = log_file.read_text(encoding="utf-8").strip().split("\n")
        assert any("test stdlib message" in line for line in lines)

    def test_setup_twice_no_error(self) -> None:
        from aot_stock_network.logging_setup import setup_logging

        setup_logging(level="DEBUG")
        setup_logging(level="INFO")

    def test_json_format(self, tmp_path: Path) -> None:
        from aot_stock_network.logging_setup import setup_logging

        log_file = tmp_path / "test.json"
        setup_logging(level="INFO", log_file=log_file, json_format=True)

        logger.info("json test message")

        content = log_file.read_text(encoding="utf-8")
        assert '"json test message"' in content
