from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .config import Settings
from .features import FEATURE_COLUMNS, build_supervised_features


class TrainingError(ValueError):
    """Raised when a model cannot be trained from validated monthly data."""


def safe_model_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    if not slug or slug in {".", ".."}:
        raise TrainingError(f"Unsafe medicine code: {value!r}")
    return slug[:100]


def _json_default(value: Any) -> str:
    if isinstance(value, (pd.Period, datetime)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def train_models(
    monthly: pd.DataFrame,
    output_dir: str | Path,
    settings: Settings,
) -> dict[str, Any]:
    supervised = build_supervised_features(
        monthly,
        lookback_months=settings.lookback_months,
        horizons=range(1, settings.max_forecast_horizon + 1),
        min_history_months=settings.min_history_months,
    )
    if supervised.empty:
        raise TrainingError("Not enough monthly history to create supervised training rows")

    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    registry: dict[str, Any] = {}
    used_slugs: set[str] = set()

    for medicine_code, medicine_rows in supervised.groupby("item_code", sort=True):
        slug = safe_model_slug(str(medicine_code))
        if slug in used_slugs:
            raise TrainingError(
                f"Medicine codes collide after path sanitization: {medicine_code!r}"
            )
        used_slugs.add(slug)
        branches = sorted({str(value) for value in medicine_rows["branch_code"]})
        encoder = LabelEncoder()
        encoder.fit(branches)
        source_features = [column for column in FEATURE_COLUMNS if column != "branch_code_encoded"]
        branch_codes = medicine_rows["branch_code"].astype(str)
        features = medicine_rows[source_features].copy()
        features["branch_code_encoded"] = encoder.transform(branch_codes).astype(int)
        target = medicine_rows["target_quantity"].to_numpy(dtype=float)
        if len(features) < 2:
            continue

        model = lgb.LGBMRegressor(
            n_estimators=settings.n_estimators,
            max_depth=settings.max_depth,
            learning_rate=settings.learning_rate,
            random_state=settings.random_state,
            n_jobs=settings.n_jobs,
            verbosity=-1,
        )
        model.fit(features, target)
        score = float(model.score(features, target))
        if not np.isfinite(score):
            score = 0.0

        model_file = f"model_{slug}.joblib"
        encoder_file = f"encoder_{slug}.joblib"
        joblib.dump(model, destination / model_file)
        joblib.dump(encoder, destination / encoder_file)

        importances = sorted(
            (
                {"feature": str(name), "importance": float(value)}
                for name, value in zip(FEATURE_COLUMNS, model.feature_importances_, strict=True)
            ),
            key=lambda item: item["importance"],
            reverse=True,
        )
        registry[str(medicine_code)] = {
            "model_file": model_file,
            "encoder_file": encoder_file,
            "medicine_name": str(medicine_rows["item_name"].iloc[0]),
            "category": str(medicine_rows["category"].iloc[0]),
            "branches": branches,
            "r2_score": score,
            "training_samples": int(len(features)),
            "feature_columns": list(FEATURE_COLUMNS),
            "feature_importances": importances,
            "training_target_range": {
                "start": str(medicine_rows["target_month"].min()),
                "end": str(medicine_rows["target_month"].max()),
            },
        }

    if not registry:
        raise TrainingError("No medicine had enough observations to train a model")

    metadata = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lookback_months": settings.lookback_months,
        "forecast_horizons": list(range(1, settings.max_forecast_horizon + 1)),
        "models": registry,
    }
    (destination / "models_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    summary = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "monthly_records": int(len(monthly)),
        "medicine_count": int(monthly["ItemCode"].nunique()),
        "branch_count": int(monthly["BranchCode"].nunique()),
        "date_range": {
            "start": str(monthly["YearMonth"].min()),
            "end": str(monthly["YearMonth"].max()),
        },
    }
    (destination / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return metadata


def load_model_registry(models_dir: str | Path) -> dict[str, dict[str, Any]]:
    directory = Path(models_dir).expanduser().resolve()
    metadata_path = directory / "models_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Model metadata does not exist: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    registry_data = metadata.get("models", {})
    if not isinstance(registry_data, dict):
        raise TrainingError("models_metadata.json has an invalid models object")

    loaded: dict[str, dict[str, Any]] = {}
    for medicine_code, info in registry_data.items():
        for key in ("model_file", "encoder_file"):
            filename = info.get(key)
            if not isinstance(filename, str) or Path(filename).name != filename:
                raise TrainingError(f"Unsafe artifact path in metadata for {medicine_code}: {key}")
            artifact = directory / filename
            if not artifact.is_file():
                raise FileNotFoundError(f"Model artifact does not exist: {artifact}")
        loaded[str(medicine_code)] = {
            "model": joblib.load(directory / info["model_file"]),
            "encoder": joblib.load(directory / info["encoder_file"]),
            "info": info,
        }
    return loaded
