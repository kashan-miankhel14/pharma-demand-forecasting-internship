from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest

from pharma_forecasting.api import create_app
from pharma_forecasting.config import Settings
from pharma_forecasting.data import build_monthly_panel, load_sales_data
from pharma_forecasting.training import load_model_registry, train_models

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "sample_pharmacy_sales.csv"


@pytest.fixture(scope="module")
def monthly():
    return build_monthly_panel(load_sales_data(SAMPLE))


@pytest.fixture(scope="module")
def model_directory(tmp_path_factory, monthly):
    destination = tmp_path_factory.mktemp("models")
    settings = Settings(
        models_dir=destination,
        max_forecast_horizon=3,
        n_estimators=20,
        max_depth=3,
        learning_rate=0.1,
        n_jobs=1,
    )
    train_models(monthly, destination, settings)
    return destination


def test_training_writes_only_relative_artifact_names(model_directory) -> None:
    registry = load_model_registry(model_directory)
    assert set(registry) == {"MED001", "MED002"}
    for item in registry.values():
        assert Path(item["info"]["model_file"]).name == item["info"]["model_file"]
        assert Path(item["info"]["encoder_file"]).name == item["info"]["encoder_file"]


def test_api_is_local_by_default_and_forecasts_horizon(model_directory) -> None:
    async def run_checks() -> None:
        settings = Settings(
            data_path=SAMPLE,
            models_dir=model_directory,
            host="127.0.0.1",
            docs_enabled=False,
            max_forecast_horizon=3,
        )
        app = create_app(settings)
        assert app.docs_url is None
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            health = await client.get("/health")
            assert health.status_code == 200
            assert health.json()["models_loaded"] == 2
            response = await client.post(
                "/predict",
                json={"medicine_code": "MED001", "branch_code": "BR001", "horizon": 2},
            )
            assert response.status_code == 200
            assert response.json()["forecast_month"] == "2025-02"
            assert response.json()["predicted_quantity"] >= 0
            docs = await client.get("/docs")
            assert docs.status_code == 404
            future = await client.post(
                "/predict",
                json={
                    "medicine_code": "MED001",
                    "branch_code": "BR001",
                    "horizon": 1,
                    "as_of_date": "2025-01-01",
                },
            )
            assert future.status_code == 422

    asyncio.run(run_checks())
