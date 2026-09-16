from __future__ import annotations

import argparse
import json

import uvicorn

from .api import create_app
from .config import Settings, resolve_path
from .data import build_monthly_panel, load_sales_data, summarize_data
from .training import train_models


def _settings(args: argparse.Namespace) -> Settings:
    settings = Settings.from_env()
    changes: dict[str, object] = {}
    if getattr(args, "data", None):
        changes["data_path"] = resolve_path(args.data)
    if getattr(args, "models_dir", None):
        changes["models_dir"] = resolve_path(args.models_dir)
    for name in (
        "host",
        "port",
        "docs_enabled",
        "forecast_horizon",
        "max_forecast_horizon",
        "lookback_months",
        "min_history_months",
        "random_state",
        "n_estimators",
        "max_depth",
        "learning_rate",
        "n_jobs",
    ):
        value = getattr(args, name, None)
        if value is not None:
            changes[name] = value
    return Settings(**{**settings.__dict__, **changes})


def _add_training_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data", help="Path to a validated input CSV")
    parser.add_argument("--models-dir", help="Directory for generated model artifacts")
    parser.add_argument("--max-forecast-horizon", type=int)
    parser.add_argument("--forecast-horizon", type=int)
    parser.add_argument("--lookback-months", type=int)
    parser.add_argument("--min-history-months", type=int)
    parser.add_argument("--n-estimators", type=int)
    parser.add_argument("--max-depth", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--random-state", type=int)
    parser.add_argument("--n-jobs", type=int)


def _add_api_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data")
    parser.add_argument("--models-dir")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--docs", action="store_true", help="Enable interactive API docs")
    parser.add_argument("--allow-public-bind", action="store_true")


def _train(args: argparse.Namespace) -> int:
    settings = _settings(args)
    raw = load_sales_data(settings.data_path)
    monthly = build_monthly_panel(raw)
    metadata = train_models(monthly, settings.models_dir, settings)
    print(
        json.dumps(
            {"trained_medicines": len(metadata["models"]), "models_dir": str(settings.models_dir)}
        )
    )
    return 0


def _check_data(args: argparse.Namespace) -> int:
    settings = _settings(args)
    raw = load_sales_data(settings.data_path)
    monthly = build_monthly_panel(raw)
    print(json.dumps(summarize_data(raw, monthly, str(settings.data_path)), indent=2))
    return 0


def _api(args: argparse.Namespace) -> int:
    settings = _settings(args)
    app = create_app(settings)
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pharma-demand-forecasting")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train LightGBM demand models")
    _add_training_arguments(train_parser)
    train_parser.set_defaults(handler=_train)

    check_parser = subparsers.add_parser("check-data", help="Validate and summarize a CSV")
    check_parser.add_argument("--data")
    check_parser.set_defaults(handler=_check_data)

    api_parser = subparsers.add_parser("api", help="Run the local forecasting API")
    _add_api_arguments(api_parser)
    api_parser.set_defaults(handler=_api)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))
