from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class ProjectSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Paths ──────────────────────────────────────────────
    project_root: Path = Path(__file__).resolve().parent.parent.parent
    data_dir: Path = project_root / "data"
    cache_dir: Path = data_dir / "cache"
    processed_dir: Path = data_dir / "processed"
    raw_dir: Path = data_dir / "raw"
    reports_dir: Path = project_root / "reports"
    models_dir: Path = project_root / "models"
    logs_dir: Path = project_root / "logs"

    # ── SET API key ───────────────────────────────────────
    set_api_key: str = ""

    # ── Data ───────────────────────────────────────────────
    start_year: int = 2015
    end_year: int = 2024
    rate_limit_seconds: float = 1.0
    cache_ttl_hours: int = 24

    # ── Model defaults ─────────────────────────────────────
    test_months: int = 12
    val_months: int = 6
    random_state: int = 42
    lstm_lookback: int = 6
    lstm_epochs: int = 100
    lstm_batch_size: int = 16

    # ── Network ────────────────────────────────────────────
    correlation_method: Literal["pearson", "spearman", "mi"] = "pearson"
    correlation_threshold: float = 0.3

    # ── Plotting ───────────────────────────────────────────
    figure_dpi: int = 300
    figure_format: Literal["png", "svg"] = "png"
    dark_mode: bool = False

    # ── Logging ────────────────────────────────────────────
    log_level: str = "INFO"
    log_max_bytes: int = 10 * 1024 * 1024  # 10 MB
    log_backup_count: int = 5

    def ensure_dirs(self) -> None:
        for attr in (
            "data_dir",
            "cache_dir",
            "processed_dir",
            "raw_dir",
            "reports_dir",
            "models_dir",
            "logs_dir",
        ):
            path: Path = getattr(self, attr)
            path.mkdir(parents=True, exist_ok=True)


settings = ProjectSettings()
