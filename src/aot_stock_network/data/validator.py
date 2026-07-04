"""
Data validation for the AOT Stock Network project.

Validates each fetched DataFrame against its DataSource schema:
  - Required columns exist
  - Data types are correct
  - Value ranges are within expected bounds
  - Date columns are chronological and within expected range
  - No duplicate dates (for time-series sources)
  - Missing value ratio is below threshold
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from aot_stock_network.data.sources import DataSource

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Validation result types
# ──────────────────────────────────────────────────────────────
VALID = "VALID"
WARNING = "WARNING"
ERROR = "ERROR"


@dataclass
class ValidationIssue:
    """A single validation finding."""

    severity: str  # VALID, WARNING, ERROR
    rule: str
    message: str
    details: Optional[Dict[str, Any]] = None


@dataclass
class ValidationResult:
    """Aggregated result of validating a single source's DataFrame."""

    source_name: str
    row_count: int
    column_count: int
    issues: List[ValidationIssue] = field(default_factory=list)
    passed: bool = True

    @property
    def n_errors(self) -> int:
        return sum(1 for i in self.issues if i.severity == ERROR)

    @property
    def n_warnings(self) -> int:
        return sum(1 for i in self.issues if i.severity == WARNING)

    @property
    def summary(self) -> str:
        return (
            f"[{self.source_name}] rows={self.row_count}, cols={self.column_count}, "
            f"errors={self.n_errors}, warnings={self.n_warnings}, "
            f"passed={self.passed}"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_name": self.source_name,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "passed": self.passed,
            "n_errors": self.n_errors,
            "n_warnings": self.n_warnings,
            "issues": [
                {"severity": i.severity, "rule": i.rule, "message": i.message} for i in self.issues
            ],
        }


# ──────────────────────────────────────────────────────────────
# Validation rules
# ──────────────────────────────────────────────────────────────
MAX_MISSING_RATIO = 0.30  # 30% max missing allowed per column


def _check_columns(df: pd.DataFrame, source: DataSource) -> List[ValidationIssue]:
    issues = []
    expected = [f.name for f in source.fields]
    actual = list(df.columns)

    missing_cols = [c for c in expected if c not in actual]
    extra_cols = [c for c in actual if c not in expected]

    if missing_cols:
        issues.append(
            ValidationIssue(
                ERROR,
                "required_columns",
                f"Missing required columns: {missing_cols}",
                {"expected": expected, "missing": missing_cols},
            )
        )
    if extra_cols:
        issues.append(
            ValidationIssue(
                WARNING,
                "extra_columns",
                f"Unexpected columns found: {extra_cols}",
                {"expected": expected, "extra": extra_cols},
            )
        )
    if not missing_cols and not extra_cols:
        issues.append(ValidationIssue(VALID, "all_columns_present", "All required columns present"))
    return issues


def _check_missing(df: pd.DataFrame, source: DataSource) -> List[ValidationIssue]:
    issues = []
    for f in source.fields:
        if f.name not in df.columns:
            continue
        null_count = df[f.name].isna().sum()
        null_ratio = null_count / len(df) if len(df) > 0 else 0.0
        if null_ratio > MAX_MISSING_RATIO and not f.nullable:
            issues.append(
                ValidationIssue(
                    ERROR,
                    "missing_values",
                    f"Column '{f.name}' has {null_ratio:.1%} missing "
                    f"(threshold: {MAX_MISSING_RATIO:.0%})",
                    {"column": f.name, "null_ratio": null_ratio},
                )
            )
        elif null_ratio > 0:
            issues.append(
                ValidationIssue(
                    WARNING if not f.nullable else VALID,
                    "missing_values",
                    f"Column '{f.name}' has {null_count} missing values ({null_ratio:.1%})",
                    {"column": f.name, "null_count": null_count},
                )
            )
    return issues


def _check_ranges(df: pd.DataFrame, source: DataSource) -> List[ValidationIssue]:
    issues = []
    for f in source.fields:
        if f.name not in df.columns:
            continue
        rules = f.validation_rules
        col = df[f.name]
        numeric_col = pd.to_numeric(col, errors="coerce")

        if "min" in rules:
            violations = (numeric_col < rules["min"]).sum()
            if violations > 0:
                issues.append(
                    ValidationIssue(
                        ERROR,
                        "range_check",
                        f"Column '{f.name}' has {violations} values below min={rules['min']}",
                        {"column": f.name, "min": rules["min"], "violations": int(violations)},
                    )
                )
        if "max" in rules:
            violations = (numeric_col > rules["max"]).sum()
            if violations > 0:
                issues.append(
                    ValidationIssue(
                        ERROR,
                        "range_check",
                        f"Column '{f.name}' has {violations} values above max={rules['max']}",
                        {"column": f.name, "max": rules["max"], "violations": int(violations)},
                    )
                )
    return issues


def _check_date_columns(df: pd.DataFrame, source: DataSource) -> List[ValidationIssue]:
    issues = []
    date_cols = [f.name for f in source.fields if f.dtype == "date" and f.name in df.columns]

    for col_name in date_cols:
        col = df[col_name]
        try:
            parsed = pd.to_datetime(col, errors="coerce")
        except Exception:
            issues.append(
                ValidationIssue(
                    ERROR, "date_parse", f"Column '{col_name}' could not be parsed as dates"
                )
            )
            continue

        null_dates = parsed.isna().sum()
        if null_dates > 0:
            issues.append(
                ValidationIssue(
                    WARNING,
                    "date_parse",
                    f"Column '{col_name}' has {null_dates} unparseable dates",
                    {"column": col_name, "null_dates": int(null_dates)},
                )
            )

        parsed = parsed.dropna().sort_values()
        if len(parsed) >= 2:
            dups = parsed.duplicated().sum()
            if dups > 0:
                issues.append(
                    ValidationIssue(
                        ERROR,
                        "duplicate_dates",
                        f"Column '{col_name}' has {dups} duplicate date values",
                        {"column": col_name, "duplicates": int(dups)},
                    )
                )
            gaps = (parsed.diff().dropna() > pd.Timedelta(days=35)).sum()
            if source.frequency == "monthly" and gaps > 0:
                issues.append(
                    ValidationIssue(
                        WARNING,
                        "date_gaps",
                        f"Column '{col_name}' has {int(gaps)} gaps > 35 days in monthly series",
                        {"column": col_name, "gaps": int(gaps)},
                    )
                )

    if not issues:
        issues.append(ValidationIssue(VALID, "date_columns", "Date columns valid"))
    return issues


def _check_dtypes(df: pd.DataFrame, source: DataSource) -> List[ValidationIssue]:
    issues = []
    for f in source.fields:
        if f.name not in df.columns:
            continue
        expected_type = f.dtype
        col = df[f.name]
        if expected_type == "float":
            numeric = pd.to_numeric(col, errors="coerce")
            non_numeric = numeric.isna().sum() - col.isna().sum()
            if non_numeric > 0:
                issues.append(
                    ValidationIssue(
                        WARNING,
                        "dtype_check",
                        f"Column '{f.name}' should be float; {non_numeric} values non-numeric",
                        {"column": f.name, "non_numeric": int(non_numeric)},
                    )
                )
        elif expected_type == "int":
            numeric = pd.to_numeric(col, errors="coerce")
            non_integer = (numeric.dropna() % 1 != 0).sum()
            if non_integer > 0:
                issues.append(
                    ValidationIssue(
                        WARNING,
                        "dtype_check",
                        f"Column '{f.name}' should be int; "
                        f"{int(non_integer)} values are non-integer",
                        {"column": f.name, "non_integer": int(non_integer)},
                    )
                )
    return issues


def _check_empty(df: pd.DataFrame, source: DataSource) -> List[ValidationIssue]:
    issues = []
    if df.empty:
        issues.append(
            ValidationIssue(ERROR, "empty_dataset", f"DataFrame for '{source.name}' is empty")
        )
    return issues


# ──────────────────────────────────────────────────────────────
# Main validation entry point
# ──────────────────────────────────────────────────────────────
def validate_dataframe(
    df: pd.DataFrame,
    source: DataSource,
) -> ValidationResult:
    """
    Validate a DataFrame against its DataSource schema.

    Runs all validation checks and returns an aggregated result.
    The result.passed is True only if there are no ERROR-level issues.
    """
    result = ValidationResult(
        source_name=source.name,
        row_count=len(df),
        column_count=len(df.columns),
    )

    check_functions = [
        _check_empty,
        _check_columns,
        _check_missing,
        _check_dtypes,
        _check_ranges,
        _check_date_columns,
    ]

    for check_fn in check_functions:
        try:
            issues = check_fn(df, source)
            result.issues.extend(issues)
        except Exception as exc:
            logger.exception(
                "Validation check '%s' failed for '%s'", check_fn.__name__, source.name
            )
            result.issues.append(
                ValidationIssue(
                    ERROR,
                    "validation_error",
                    f"Check '{check_fn.__name__}' raised exception: {exc}",
                )
            )

    result.passed = result.n_errors == 0
    logger.info("Validation %s for '%s'", "PASSED" if result.passed else "FAILED", source.name)
    if result.n_warnings > 0 or result.n_errors > 0:
        logger.info(result.summary)

    return result


def validate_all(
    source_data: Dict[str, pd.DataFrame],
) -> Dict[str, ValidationResult]:
    """Validate all DataFrames against their source definitions."""
    from aot_stock_network.data.sources import SOURCE_REGISTRY

    results = {}
    for name, df in source_data.items():
        if name in SOURCE_REGISTRY:
            results[name] = validate_dataframe(df, SOURCE_REGISTRY[name])
        else:
            logger.warning("No source definition for '%s', skipping validation", name)
    return results
