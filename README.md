# Pharma Demand Forecasting Internship

A clean, local pharmacy demand-forecasting repository based on the supplied stock-prediction project. It provides validated monthly aggregation, leakage-aware multi-horizon LightGBM training, a FastAPI forecasting endpoint, synthetic test data, and security-focused defaults.

## Project layout

```text
src/pharma_forecasting/  package code
scripts/                 command wrappers
data/                    synthetic sample and data policy
models/.gitkeep          runtime model directory
tests/                   validation, training, and API tests
docs/                    data, model, and security notes
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Copy `.env.example` to `.env` when local overrides are needed. The default configuration uses the synthetic sample, writes runtime artifacts to `models/`, binds to localhost, and disables API docs.

## Commands

Validate and summarize data:

```bash
python scripts/check_data.py
```

Train local models:

```bash
python scripts/train.py
```

Run the API:

```bash
python scripts/run_api.py
```

The default health endpoint is `http://127.0.0.1:8000/health`. `FORECAST_HORIZON` sets the default horizon for omitted requests, while `MAX_FORECAST_HORIZON` bounds accepted horizons. Interactive docs are intentionally disabled. To enable them for local development, set `API_DOCS_ENABLED=true`; to bind publicly, set both `API_HOST=0.0.0.0` and `ALLOW_PUBLIC_BIND=true` after reviewing the security notes.

Prediction example:

```bash
curl -s http://127.0.0.1:8000/predict   -H 'content-type: application/json'   -d '{"medicine_code":"MED001","branch_code":"BR001","horizon":1}'
```

## Tests and smoke checks

```bash
python -m compileall -q src scripts tests
python -m pytest
```

The tests use the synthetic sample and temporary model directories. No raw commercial CSV or trained binary is committed.

## Data and security

The original project's CSV files were not copied because their provenance and commercial sensitivity could not be verified. The checked-in CSV is a tiny, clearly synthetic sample. See [docs/data.md](docs/data.md), [docs/security.md](docs/security.md), and [docs/model.md](docs/model.md) for the schema, date/horizon rules, and operational caveats.

This is an internship demonstration, not a production inventory or clinical decision system.
