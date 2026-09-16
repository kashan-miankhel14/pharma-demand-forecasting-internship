# Model notes

The trainer uses one LightGBM regression model per medicine. Branch codes are encoded with a per-medicine `LabelEncoder`; the metadata stores only artifact filenames, never absolute host paths.

Features are rolling demand, price, profit, stock, previous-sales, promotion, weekend, winter, calendar, and forecast-horizon features. Training rows cover every horizon through `MAX_FORECAST_HORIZON`, so requests can target different future months without retraining. Training scores are in-sample scores and should not be interpreted as production accuracy.

The sample data is intentionally tiny. A real deployment needs chronological validation, backtesting, calibration, monitoring, retraining rules, and approval from the data owner.
