from __future__ import annotations

from datetime import date, datetime
from typing import Any, Iterable

import pandas as pd

FEATURE_COLUMNS = (
    "prev_month_quantity",
    "avg_last_3m_quantity",
    "avg_last_6m_quantity",
    "max_last_6m_quantity",
    "min_last_6m_quantity",
    "trend_quantity",
    "avg_cost_price_6m",
    "avg_sale_price_6m",
    "avg_profit_6m",
    "avg_stock_available_6m",
    "avg_previous_month_sales_6m",
    "promotion_rate_6m",
    "winter_rate_6m",
    "weekend_rate_6m",
    "month_of_year",
    "quarter_of_year",
    "forecast_horizon",
    "branch_code_encoded",
)

METADATA_COLUMNS = (
    "item_code",
    "item_name",
    "category",
    "branch_code",
    "as_of_month",
    "target_month",
    "history_start_month",
    "history_end_month",
    "horizon",
)


class InsufficientHistoryError(ValueError):
    """Raised when a forecast cannot be built from the requested history window."""


def add_months(period: pd.Period, months: int) -> pd.Period:
    if months < 1:
        raise ValueError("months must be at least 1")
    return period + months


def coerce_period(value: date | datetime | pd.Period | str | None) -> pd.Period | None:
    if value is None:
        return None
    if isinstance(value, pd.Period):
        return value
    if isinstance(value, (date, datetime)):
        return pd.Period(value, freq="M")
    return pd.Period(value, freq="M")


def resolve_cutoff(
    monthly: pd.DataFrame,
    as_of_date: date | datetime | pd.Period | str | None,
) -> pd.Period:
    if monthly.empty or "YearMonth" not in monthly:
        raise ValueError("Monthly data is empty")
    minimum = monthly["YearMonth"].min()
    maximum = monthly["YearMonth"].max()
    if as_of_date is None:
        return maximum
    cutoff = coerce_period(as_of_date)
    if cutoff is None or cutoff < minimum or cutoff > maximum:
        raise ValueError(f"as_of_date must be between {minimum} and {maximum}")
    return cutoff


def _feature_row(
    history: pd.DataFrame,
    target_period: pd.Period,
    branch_code: str,
) -> dict[str, Any]:
    quantities = history["MonthlyQuantity"].to_numpy(dtype=float)
    last = float(quantities[-1])
    previous = float(quantities[-2]) if len(quantities) >= 2 else last
    return {
        "prev_month_quantity": last,
        "avg_last_3m_quantity": float(history["MonthlyQuantity"].tail(3).mean()),
        "avg_last_6m_quantity": float(history["MonthlyQuantity"].tail(6).mean()),
        "max_last_6m_quantity": float(history["MonthlyQuantity"].tail(6).max()),
        "min_last_6m_quantity": float(history["MonthlyQuantity"].tail(6).min()),
        "trend_quantity": last - previous,
        "avg_cost_price_6m": float(history["AvgCostPrice"].tail(6).mean()),
        "avg_sale_price_6m": float(history["AvgSalePrice"].tail(6).mean()),
        "avg_profit_6m": float(history["TotalProfit"].tail(6).mean()),
        "avg_stock_available_6m": float(history["AvgStockAvailable"].tail(6).mean()),
        "avg_previous_month_sales_6m": float(history["AvgPreviousMonthSales"].tail(6).mean()),
        "promotion_rate_6m": float(history["PromotionRate"].tail(6).mean()),
        "winter_rate_6m": float(history["WinterRate"].tail(6).mean()),
        "weekend_rate_6m": float(history["WeekendRate"].tail(6).mean()),
        "month_of_year": int(target_period.month),
        "quarter_of_year": int(target_period.quarter),
        "branch_code": branch_code,
    }


def _validate_horizons(horizons: Iterable[int]) -> tuple[int, ...]:
    values = tuple(sorted({int(value) for value in horizons}))
    if not values or any(value < 1 for value in values):
        raise ValueError("horizons must contain positive integers")
    return values


def build_supervised_features(
    monthly: pd.DataFrame,
    lookback_months: int = 6,
    horizons: Iterable[int] = (1,),
    min_history_months: int = 3,
) -> pd.DataFrame:
    if lookback_months < 1:
        raise ValueError("lookback_months must be at least 1")
    if min_history_months < 1 or min_history_months > lookback_months:
        raise ValueError("min_history_months must be between 1 and lookback_months")
    horizon_values = _validate_horizons(horizons)

    records: list[dict[str, Any]] = []
    group_columns = ["ItemCode", "BranchCode"]
    for (item_code, branch_code), group in monthly.groupby(group_columns, sort=True):
        group = group.sort_values("YearMonth").reset_index(drop=True)
        for horizon in horizon_values:
            for index in range(min_history_months - 1, len(group) - horizon):
                history = group.iloc[max(0, index - lookback_months + 1) : index + 1]
                target = group.iloc[index + horizon]
                row = _feature_row(history, target["YearMonth"], str(branch_code))
                row.update(
                    {
                        "item_code": str(item_code),
                        "item_name": str(group.iloc[0]["ItemName"]),
                        "category": str(group.iloc[0]["Category"]),
                        "branch_code": str(branch_code),
                        "as_of_month": str(group.iloc[index]["YearMonth"]),
                        "target_month": str(target["YearMonth"]),
                        "history_start_month": str(history["YearMonth"].iloc[0]),
                        "history_end_month": str(history["YearMonth"].iloc[-1]),
                        "horizon": int(horizon),
                        "forecast_horizon": int(horizon),
                        "target_quantity": float(target["MonthlyQuantity"]),
                    }
                )
                records.append(row)
    if not records:
        return pd.DataFrame(columns=[*FEATURE_COLUMNS, *METADATA_COLUMNS, "target_quantity"])
    return pd.DataFrame(records)


def build_prediction_features(
    monthly: pd.DataFrame,
    horizon: int,
    lookback_months: int = 6,
    min_history_months: int = 3,
    as_of_date: date | datetime | pd.Period | str | None = None,
    previous_month_quantity: float | None = None,
) -> pd.DataFrame:
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    cutoff = resolve_cutoff(monthly, as_of_date)
    history = monthly[monthly["YearMonth"] <= cutoff].sort_values("YearMonth").tail(lookback_months)
    if len(history) < min_history_months:
        raise InsufficientHistoryError(
            f"At least {min_history_months} monthly observations are required"
        )
    history = history.copy()
    if previous_month_quantity is not None:
        if previous_month_quantity < 0:
            raise ValueError("previous_month_quantity must be non-negative")
        history.iloc[-1, history.columns.get_loc("MonthlyQuantity")] = previous_month_quantity

    row = _feature_row(history, add_months(cutoff, horizon), str(history["BranchCode"].iloc[-1]))
    row.update(
        {
            "item_code": str(history["ItemCode"].iloc[-1]),
            "item_name": str(history["ItemName"].iloc[-1]),
            "category": str(history["Category"].iloc[-1]),
            "branch_code": str(history["BranchCode"].iloc[-1]),
            "as_of_month": str(cutoff),
            "target_month": str(add_months(cutoff, horizon)),
            "history_start_month": str(history["YearMonth"].iloc[0]),
            "history_end_month": str(history["YearMonth"].iloc[-1]),
            "horizon": int(horizon),
            "forecast_horizon": int(horizon),
        }
    )
    return pd.DataFrame([row])
