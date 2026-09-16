from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from pharma_forecasting.data import (
    DataValidationError,
    build_monthly_panel,
    load_sales_data,
    validate_required_columns,
)
from pharma_forecasting.features import (
    InsufficientHistoryError,
    build_prediction_features,
    build_supervised_features,
)

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "sample_pharmacy_sales.csv"


def test_synthetic_sample_loads_and_aggregates() -> None:
    raw = load_sales_data(SAMPLE)
    monthly = build_monthly_panel(raw)
    assert len(raw) == 96
    assert len(monthly) == 96
    assert raw["ItemCode"].nunique() == 2
    assert monthly["YearMonth"].min() == pd.Period("2023-01", freq="M")
    assert monthly["YearMonth"].max() == pd.Period("2024-12", freq="M")


def test_required_column_validation_reports_missing_columns() -> None:
    frame = pd.read_csv(SAMPLE).drop(columns=["SellingQuantity"])
    with pytest.raises(DataValidationError, match="SellingQuantity"):
        validate_required_columns(frame)


def test_supervised_features_shift_target_by_horizon() -> None:
    monthly = build_monthly_panel(load_sales_data(SAMPLE))
    features = build_supervised_features(monthly, horizons=(2,), min_history_months=3)
    first = features.sort_values(["item_code", "branch_code", "as_of_month"]).iloc[0]
    assert pd.Period(first["target_month"], freq="M") == (
        pd.Period(first["as_of_month"], freq="M") + 2
    )
    assert pd.Period(first["history_end_month"], freq="M") == pd.Period(
        first["as_of_month"], freq="M"
    )


def test_prediction_features_reject_future_cutoff() -> None:
    monthly = build_monthly_panel(load_sales_data(SAMPLE))
    with pytest.raises(ValueError, match="between"):
        build_prediction_features(monthly, horizon=1, as_of_date=date(2025, 1, 1))
    with pytest.raises(InsufficientHistoryError):
        build_prediction_features(monthly.head(2), horizon=1)
