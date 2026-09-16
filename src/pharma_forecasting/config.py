from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env_bool(key: str, default: bool) -> bool:
    value = os.getenv(key)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(key: str, default: int) -> int:
    value = os.getenv(key)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    value = os.getenv(key)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def resolve_path(value: str | Path, base: Path = PROJECT_ROOT) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else base / path


@dataclass(frozen=True)
class Settings:
    data_path: Path = PROJECT_ROOT / "data" / "sample_pharmacy_sales.csv"
    models_dir: Path = PROJECT_ROOT / "models"
    host: str = "127.0.0.1"
    port: int = 8000
    docs_enabled: bool = False
    allow_public_bind: bool = False
    forecast_horizon: int = 1
    max_forecast_horizon: int = 24
    lookback_months: int = 6
    min_history_months: int = 3
    random_state: int = 42
    n_estimators: int = 80
    max_depth: int = 4
    learning_rate: float = 0.08
    n_jobs: int = 1

    @classmethod
    def from_env(cls, dotenv_path: Path | None = None) -> "Settings":
        path = dotenv_path or PROJECT_ROOT / ".env"
        if path.exists():
            load_dotenv(dotenv_path=path)
        settings = cls(
            data_path=resolve_path(os.getenv("DATA_PATH", "data/sample_pharmacy_sales.csv")),
            models_dir=resolve_path(os.getenv("MODELS_DIR", "models")),
            host=os.getenv("API_HOST", "127.0.0.1").strip(),
            port=_env_int("API_PORT", 8000),
            docs_enabled=_env_bool("API_DOCS_ENABLED", False),
            allow_public_bind=_env_bool("ALLOW_PUBLIC_BIND", False),
            forecast_horizon=_env_int("FORECAST_HORIZON", 1),
            max_forecast_horizon=_env_int("MAX_FORECAST_HORIZON", 24),
            lookback_months=_env_int("LOOKBACK_MONTHS", 6),
            min_history_months=_env_int("MIN_HISTORY_MONTHS", 3),
            random_state=_env_int("LIGHTGBM_RANDOM_STATE", 42),
            n_estimators=_env_int("LIGHTGBM_N_ESTIMATORS", 80),
            max_depth=_env_int("LIGHTGBM_MAX_DEPTH", 4),
            learning_rate=_env_float("LIGHTGBM_LEARNING_RATE", 0.08),
            n_jobs=_env_int("LIGHTGBM_N_JOBS", 1),
        )
        if settings.forecast_horizon < 1:
            raise ValueError("FORECAST_HORIZON must be at least 1")
        if settings.max_forecast_horizon < settings.forecast_horizon:
            raise ValueError("MAX_FORECAST_HORIZON must be at least FORECAST_HORIZON")
        if settings.lookback_months < 1:
            raise ValueError("LOOKBACK_MONTHS must be at least 1")
        if (
            settings.min_history_months < 1
            or settings.min_history_months > settings.lookback_months
        ):
            raise ValueError("MIN_HISTORY_MONTHS must be between 1 and LOOKBACK_MONTHS")
        if settings.port < 1 or settings.port > 65535:
            raise ValueError("API_PORT must be between 1 and 65535")
        if settings.host in {"0.0.0.0", "::"} and not settings.allow_public_bind:
            raise ValueError("Public API binding requires ALLOW_PUBLIC_BIND=true")
        return settings
