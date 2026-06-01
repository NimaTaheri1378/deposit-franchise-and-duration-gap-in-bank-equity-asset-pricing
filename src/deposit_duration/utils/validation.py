from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ValidationResult:
    name: str
    passed: bool
    message: str


def assert_unique_keys(df: pd.DataFrame, keys: list[str], name: str) -> ValidationResult:
    missing = [k for k in keys if k not in df.columns]
    if missing:
        return ValidationResult(name, False, f"missing key columns: {missing}")
    dupes = int(df.duplicated(keys).sum())
    if dupes:
        return ValidationResult(name, False, f"{dupes} duplicate rows on {keys}")
    return ValidationResult(name, True, f"{len(df):,} unique rows on {keys}")


def require_columns(df: pd.DataFrame, columns: list[str], name: str) -> ValidationResult:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        return ValidationResult(name, False, f"missing columns: {missing}")
    return ValidationResult(name, True, "all required columns present")


def month_end(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s).dt.to_period("M").dt.to_timestamp("M")

