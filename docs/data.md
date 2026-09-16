# Data contract

## Required columns

The loader validates these columns before aggregation:

- Keys and labels: `ItemCode`, `ItemName`, `Category`, `BranchCode`, `BranchName`
- Demand and financial values: `SellingQuantity`, `CostPrice`, `SalePrice`, `Profit`, `StockAvailable`, `PreviousMonthSales`
- Dates: `SaleDate`, `CreatedDateTime`
- Calendar and operating flags: `Year`, `Month`, `Quarter`, `DayOfWeek`, `IsWeekend`, `IsWinter`, `PromotionApplied`

Numeric values must be finite. Demand, prices, stock, and previous-month sales must be non-negative. `PromotionApplied`, `IsWeekend`, and `IsWinter` must be binary `0` or `1`.

## Monthly aggregation

Rows are grouped by calendar month, medicine, and branch. Missing calendar months inside an observed range are filled with zero demand; price, stock, and rate fields are forward-filled and then zero-filled when no prior value exists. This makes horizon arithmetic calendar-based instead of row-based.

## Date and horizon semantics

`as_of_date` is a cutoff month and must fall within the observed data range. A forecast with `horizon=1` targets the month after the cutoff; `horizon=3` targets three calendar months after it. Training features use only months at or before the feature month, and the target is shifted forward by the horizon. This prevents target leakage.

## Synthetic sample

`data/sample_pharmacy_sales.csv` is explicitly synthetic and contains only fictional codes and deterministic values. It is small enough to train a local smoke-test model quickly. It is not suitable for evaluating a production forecasting system.
