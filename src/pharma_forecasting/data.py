from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = (
    "ItemCode",
    "ItemName",
    "Category",
    "BranchCode",
    "BranchName",
    "SellingQuantity",
    "CostPrice",
    "SalePrice",
    "Profit",
    "SaleDate",
    "CreatedDateTime",
    "PromotionApplied",
    "StockAvailable",
    "PreviousMonthSales",
    "Year",
    "Month",
    "Quarter",
    "DayOfWeek",
    "IsWeekend",
    "IsWinter",
)

NUMERIC_COLUMNS = (
    "SellingQuantity",
    "CostPrice",
    "SalePrice",
    "Profit",
    "StockAvailable",
    "PreviousMonthSales",
    "PromotionApplied",
    "Year",
    "Month",
    "Quarter",
    "DayOfWeek",
    "IsWeekend",
    "IsWinter",
)

BINARY_COLUMNS = ("PromotionApplied", "IsWeekend", "IsWinter")
NONNEGATIVE_COLUMNS = (
    "SellingQuantity",
    "CostPrice",
    "SalePrice",
    "StockAvailable",
    "PreviousMonthSales",
)


class DataValidationError(ValueError):
    """Raised when an input data set does not satisfy the forecasting contract."""


def validate_required_columns(frame: pd.DataFrame) -> None:
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise DataValidationError(f"Input CSV is missing required columns: {missing}")


def load_sales_data(path: str | Path) -> pd.DataFrame:
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"Input CSV does not exist: {source}")

    frame = pd.read_csv(source)
    frame.columns = frame.columns.str.strip()
    validate_required_columns(frame)

    frame["SaleDate"] = pd.to_datetime(frame["SaleDate"], errors="coerce")
    frame["CreatedDateTime"] = pd.to_datetime(frame["CreatedDateTime"], errors="coerce")
    invalid_dates = frame["SaleDate"].isna() | frame["CreatedDateTime"].isna()
    if invalid_dates.any():
        raise DataValidationError(f"Input CSV contains {int(invalid_dates.sum())} invalid dates")

    for column in NUMERIC_COLUMNS:
        converted = pd.to_numeric(frame[column], errors="coerce")
        values = converted.to_numpy(dtype=float, na_value=np.nan)
        if not np.isfinite(values).all():
            raise DataValidationError(f"Column {column} contains non-finite numeric values")
        frame[column] = converted.astype(float)

    for column in BINARY_COLUMNS:
        invalid = ~frame[column].isin([0, 1, 0.0, 1.0])
        if invalid.any():
            raise DataValidationError(f"Column {column} must contain only 0 or 1")

    for column in NONNEGATIVE_COLUMNS:
        invalid = frame[column] < 0
        if invalid.any():
            raise DataValidationError(f"Column {column} must be non-negative")

    if frame.empty:
        raise DataValidationError("Input CSV is empty")
    return frame


def build_monthly_panel(frame: pd.DataFrame) -> pd.DataFrame:
    validate_required_columns(frame)
    data = frame.copy()
    data["YearMonth"] = data["SaleDate"].dt.to_period("M")
    monthly = (
        data.groupby(
            ["YearMonth", "ItemCode", "ItemName", "Category", "BranchCode", "BranchName"],
            as_index=False,
            sort=True,
        )
        .agg(
            MonthlyQuantity=("SellingQuantity", "sum"),
            AvgCostPrice=("CostPrice", "mean"),
            AvgSalePrice=("SalePrice", "mean"),
            TotalProfit=("Profit", "sum"),
            AvgStockAvailable=("StockAvailable", "mean"),
            AvgPreviousMonthSales=("PreviousMonthSales", "mean"),
            PromotionRate=("PromotionApplied", "mean"),
            WeekendRate=("IsWeekend", "mean"),
            WinterRate=("IsWinter", "mean"),
        )
        .sort_values(["ItemCode", "BranchCode", "YearMonth"])
        .reset_index(drop=True)
    )
    return complete_monthly_panel(monthly)


def complete_monthly_panel(monthly: pd.DataFrame) -> pd.DataFrame:
    panels: list[pd.DataFrame] = []
    group_columns = ["ItemCode", "ItemName", "Category", "BranchCode", "BranchName"]
    for _keys, group in monthly.groupby(group_columns, as_index=False, sort=True):
        group = group.sort_values("YearMonth").set_index("YearMonth")
        periods = pd.period_range(group.index.min(), group.index.max(), freq="M")
        group = group.reindex(periods)
        group.index.name = "YearMonth"
        group["MonthlyQuantity"] = group["MonthlyQuantity"].fillna(0.0)
        numeric_columns = [
            "AvgCostPrice",
            "AvgSalePrice",
            "TotalProfit",
            "AvgStockAvailable",
            "AvgPreviousMonthSales",
            "PromotionRate",
            "WeekendRate",
            "WinterRate",
        ]
        group[numeric_columns] = group[numeric_columns].ffill().bfill().fillna(0.0)
        group[group_columns] = group[group_columns].ffill().bfill()
        group = group.reset_index()
        panels.append(group)
    if not panels:
        return monthly.copy()
    return pd.concat(panels, ignore_index=True).sort_values(
        ["ItemCode", "BranchCode", "YearMonth"]
    ).reset_index(drop=True)


def summarize_data(raw: pd.DataFrame, monthly: pd.DataFrame, source_file: str) -> dict[str, Any]:
    monthly_trend = (
        monthly.groupby("YearMonth", as_index=False)["MonthlyQuantity"]
        .sum()
        .sort_values("YearMonth")
        .tail(12)
    )
    top_medicines = (
        raw.groupby(["ItemCode", "ItemName"], as_index=False)["SellingQuantity"]
        .sum()
        .sort_values("SellingQuantity", ascending=False)
        .head(5)
    )
    top_branches = (
        raw.groupby(["BranchCode", "BranchName"], as_index=False)["SellingQuantity"]
        .sum()
        .sort_values("SellingQuantity", ascending=False)
        .head(5)
    )
    return {
        "source_file": source_file,
        "raw_records": int(len(raw)),
        "monthly_records": int(len(monthly)),
        "medicine_count": int(raw["ItemCode"].nunique()),
        "branch_count": int(raw["BranchCode"].nunique()),
        "category_count": int(raw["Category"].nunique()),
        "date_range": {
            "start": raw["SaleDate"].min().date().isoformat(),
            "end": raw["SaleDate"].max().date().isoformat(),
        },
        "total_quantity": float(raw["SellingQuantity"].sum()),
        "total_profit": float(round(raw["Profit"].sum(), 2)),
        "avg_sale_price": float(round(raw["SalePrice"].mean(), 2)),
        "avg_cost_price": float(round(raw["CostPrice"].mean(), 2)),
        "top_medicines": top_medicines.to_dict(orient="records"),
        "top_branches": top_branches.to_dict(orient="records"),
        "monthly_trend": [
            {"year_month": str(row.YearMonth), "quantity": float(row.MonthlyQuantity)}
            for row in monthly_trend.itertuples(index=False)
        ],
    }
