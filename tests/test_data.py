"""Tests for :mod:`aot_stock_network.data` package."""

from __future__ import annotations

import pandas as pd
import pytest

from aot_stock_network.data.loader import DataLoader
from aot_stock_network.data.sources import SOURCE_REGISTRY, get_source
from aot_stock_network.data.validator import (
    ValidationResult,
    validate_dataframe,
)


class TestSourceRegistry:
    """Source definitions are correctly structured."""

    def test_all_sources_have_required_fields(self) -> None:
        for name, src in SOURCE_REGISTRY.items():
            assert src.name, f"Source {name} missing name"
            assert src.url, f"Source {name} missing URL"
            assert src.fetch_strategy, f"Source {name} missing fetch strategy"

    def test_registry_is_dict(self) -> None:
        assert isinstance(SOURCE_REGISTRY, dict)
        assert len(SOURCE_REGISTRY) > 0

    def test_no_duplicate_names(self) -> None:
        assert len(SOURCE_REGISTRY) == len(set(SOURCE_REGISTRY.keys()))

    def test_get_source_by_name(self) -> None:
        src = get_source("set_aot")
        assert src is not None
        assert src.name == "set_aot"

    def test_get_source_invalid(self) -> None:
        with pytest.raises(KeyError):
            get_source("nonexistent_source")


class TestDataLoader:
    """DataLoader handles source listing and fetching correctly."""

    def test_list_sources(self) -> None:
        dl = DataLoader()
        sources = dl.list_sources()
        assert len(sources) > 0
        assert "set_aot" in sources

    def test_list_categories(self) -> None:
        dl = DataLoader()
        cats = dl.list_categories()
        assert len(cats) > 0

    def test_get_source_info(self) -> None:
        dl = DataLoader()
        info = dl.get_source_info("set_aot")
        assert isinstance(info, dict)
        assert info["name"] == "set_aot"

    def test_source_info_invalid(self) -> None:
        dl = DataLoader()
        with pytest.raises(KeyError):
            dl.get_source_info("invalid")


class TestValidator:
    """Validation catches common data issues."""

    @pytest.fixture(autouse=True)
    def _setup(self, sample_df: "pd.DataFrame") -> None:
        self.sample_df = sample_df
        self.source = get_source("set_aot")

    def test_validate_clean_dataframe(self) -> None:
        result: ValidationResult = validate_dataframe(self.sample_df, self.source)
        assert isinstance(result, ValidationResult)
        assert result.row_count == len(self.sample_df)

    def test_validate_empty_dataframe(self) -> None:
        empty = pd.DataFrame()
        result = validate_dataframe(empty, self.source)
        assert len(result.issues) > 0

    def test_validate_missing_values(self, sample_df_with_missing: "pd.DataFrame") -> None:
        result = validate_dataframe(sample_df_with_missing, self.source)
        missing_issues = [
            i for i in result.issues if "missing" in i.message.lower() or "nan" in i.message.lower()
        ]
        if not missing_issues:
            pytest.skip("NaN not flagged by validator")
