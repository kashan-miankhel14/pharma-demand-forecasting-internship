from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from .config import Settings
from .data import build_monthly_panel, load_sales_data, summarize_data
from .features import (
    FEATURE_COLUMNS,
    InsufficientHistoryError,
    add_months,
    build_prediction_features,
    resolve_cutoff,
)
from .training import load_model_registry


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    medicine_code: str = Field(min_length=1, max_length=100)
    branch_code: str = Field(min_length=1, max_length=100)
    horizon: int | None = Field(default=None, ge=1)
    as_of_date: date | None = None
    prev_month_quantity: float | None = Field(default=None, ge=0)


class PredictionResponse(BaseModel):
    medicine_code: str
    medicine_name: str
    branch_code: str
    predicted_quantity: float
    forecast_month: str
    as_of_month: str
    horizon: int
    confidence: str
    timestamp: str
    model_info: dict[str, Any]


class BatchPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    predictions: list[PredictionRequest] = Field(min_length=1, max_length=100)


class BatchPredictionResponse(BaseModel):
    predictions: list[dict[str, Any]]
    total_count: int
    successful: int
    failed: int


def _confidence(score: float) -> str:
    if score > 0.75:
        return "HIGH"
    if score > 0.5:
        return "MEDIUM"
    return "LOW"


def create_app(
    settings: Settings | None = None,
    data_path: str | Path | None = None,
    models_dir: str | Path | None = None,
) -> FastAPI:
    active_settings = settings or Settings.from_env()
    source = Path(data_path).expanduser() if data_path is not None else active_settings.data_path
    model_directory = (
        Path(models_dir).expanduser() if models_dir is not None else active_settings.models_dir
    )
    raw = load_sales_data(source)
    monthly = build_monthly_panel(raw)
    registry = load_model_registry(model_directory)
    summary = summarize_data(raw, monthly, str(source))

    docs_url = "/docs" if active_settings.docs_enabled else None
    redoc_url = "/redoc" if active_settings.docs_enabled else None
    openapi_url = "/openapi.json" if active_settings.docs_enabled else None
    app = FastAPI(
        title="Pharmacy Demand Forecasting API",
        description="Local, validated demand forecasting for pharmacy inventory planning.",
        version="0.1.0",
        docs_url=docs_url,
        redoc_url=redoc_url,
        openapi_url=openapi_url,
    )
    app.state.settings = active_settings
    app.state.raw_data = raw
    app.state.monthly_data = monthly
    app.state.model_registry = registry
    app.state.summary = summary

    @app.get("/health")
    def health_check() -> dict[str, Any]:
        return {
            "status": "healthy",
            "models_loaded": len(registry),
            "dataset_loaded": True,
            "dataset_records": int(len(raw)),
        }

    @app.get("/analysis/summary")
    def analysis_summary() -> dict[str, Any]:
        scores = [item["info"]["r2_score"] for item in registry.values()]
        return {
            "summary": summary,
            "model_overview": {
                "medicines_trained": len(registry),
                "average_r2": float(sum(scores) / len(scores)) if scores else 0.0,
                "best_model": (
                    max(registry, key=lambda code: registry[code]["info"]["r2_score"])
                    if registry
                    else None
                ),
            },
        }

    @app.get("/analysis/medicine/{medicine_code}")
    def analysis_medicine(medicine_code: str) -> dict[str, Any]:
        if medicine_code not in registry:
            raise HTTPException(status_code=404, detail="Medicine not found")
        medicine_data = monthly[monthly["ItemCode"] == medicine_code]
        if medicine_data.empty:
            raise HTTPException(status_code=404, detail="No records found for medicine")
        monthly_trend = (
            medicine_data.groupby("YearMonth", as_index=False)["MonthlyQuantity"]
            .sum()
            .sort_values("YearMonth")
            .tail(12)
        )
        branches = (
            raw[raw["ItemCode"] == medicine_code]
            .groupby(["BranchCode", "BranchName"], as_index=False)["SellingQuantity"]
            .sum()
            .sort_values("SellingQuantity", ascending=False)
            .head(5)
        )
        category_values = medicine_data["Category"].dropna()
        category = str(category_values.mode().iloc[0]) if not category_values.empty else None
        info = registry[medicine_code]["info"]
        return {
            "medicine_code": medicine_code,
            "medicine_name": info.get("medicine_name", medicine_code),
            "category": category,
            "records": int(len(medicine_data)),
            "date_range": {
                "start": str(medicine_data["YearMonth"].min()),
                "end": str(medicine_data["YearMonth"].max()),
            },
            "monthly_trend": [
                {"month": str(row.YearMonth), "quantity": float(row.MonthlyQuantity)}
                for row in monthly_trend.itertuples(index=False)
            ],
            "top_branches": branches.to_dict(orient="records"),
            "r2_score": float(round(float(info["r2_score"]), 4)),
            "feature_importances": info.get("feature_importances", [])[:10],
        }

    @app.get("/medicines")
    def list_medicines() -> dict[str, Any]:
        medicines = [
            {
                "code": code,
                "name": item["info"].get("medicine_name", code),
                "category": item["info"].get("category"),
                "r2_score": round(float(item["info"]["r2_score"]), 4),
                "branches": item["info"].get("branches", []),
                "training_samples": int(item["info"].get("training_samples", 0)),
            }
            for code, item in registry.items()
        ]
        return {"medicines": medicines, "count": len(medicines)}

    @app.get("/branches/{medicine_code}")
    def get_branches(medicine_code: str) -> dict[str, Any]:
        if medicine_code not in registry:
            raise HTTPException(status_code=404, detail="Medicine not found")
        branches = registry[medicine_code]["info"].get("branches", [])
        return {
            "medicine_code": medicine_code,
            "medicine_name": registry[medicine_code]["info"].get("medicine_name", medicine_code),
            "branches": branches,
            "count": len(branches),
        }

    @app.post("/predict", response_model=PredictionResponse)
    def predict_stock(request: PredictionRequest) -> PredictionResponse:
        if request.medicine_code not in registry:
            raise HTTPException(status_code=404, detail="Medicine not found in trained models")
        horizon = request.horizon or active_settings.forecast_horizon
        if horizon > active_settings.max_forecast_horizon:
            raise HTTPException(
                status_code=422,
                detail=f"horizon must be at most {active_settings.max_forecast_horizon}",
            )
        medicine = registry[request.medicine_code]
        branches = medicine["info"].get("branches", [])
        if request.branch_code not in branches:
            raise HTTPException(status_code=400, detail="Branch not found for medicine")

        medicine_monthly = monthly[
            (monthly["ItemCode"] == request.medicine_code)
            & (monthly["BranchCode"] == request.branch_code)
        ].sort_values("YearMonth")
        if medicine_monthly.empty:
            raise HTTPException(
                status_code=400, detail="No historical data for medicine and branch"
            )
        try:
            cutoff = resolve_cutoff(medicine_monthly, request.as_of_date)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        history = medicine_monthly[medicine_monthly["YearMonth"] <= cutoff]
        if len(history) < active_settings.min_history_months:
            raise HTTPException(status_code=400, detail="Not enough monthly history")

        try:
            feature_frame = build_prediction_features(
                history,
                horizon=horizon,
                lookback_months=active_settings.lookback_months,
                min_history_months=active_settings.min_history_months,
                as_of_date=cutoff,
                previous_month_quantity=request.prev_month_quantity,
            )
        except InsufficientHistoryError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        source_features = [column for column in FEATURE_COLUMNS if column != "branch_code_encoded"]
        feature_frame = feature_frame[source_features]
        encoded_branch = int(medicine["encoder"].transform([request.branch_code])[0])
        feature_frame["branch_code_encoded"] = encoded_branch
        predicted = max(float(medicine["model"].predict(feature_frame)[0]), 0.0)
        score = float(medicine["info"]["r2_score"])
        return PredictionResponse(
            medicine_code=request.medicine_code,
            medicine_name=str(medicine["info"].get("medicine_name", request.medicine_code)),
            branch_code=request.branch_code,
            predicted_quantity=round(predicted, 2),
            forecast_month=str(add_months(cutoff, horizon)),
            as_of_month=str(cutoff),
            horizon=horizon,
            confidence=_confidence(score),
            timestamp=pd.Timestamp.now().isoformat(),
            model_info={
                "r2_score": round(score, 4),
                "training_samples": int(medicine["info"].get("training_samples", len(history))),
                "historical_avg": round(float(history["MonthlyQuantity"].mean()), 2),
                "last_month_actual": round(float(history["MonthlyQuantity"].iloc[-1]), 2),
                "feature_columns": list(FEATURE_COLUMNS),
            },
        )

    @app.post("/predict-batch", response_model=BatchPredictionResponse)
    def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
        results: list[dict[str, Any]] = []
        successful = 0
        failed = 0
        for prediction_request in request.predictions:
            try:
                result = predict_stock(prediction_request)
                results.append(result.model_dump())
                successful += 1
            except HTTPException as exc:
                failed += 1
                results.append(
                    {
                        "medicine_code": prediction_request.medicine_code,
                        "branch_code": prediction_request.branch_code,
                        "error": exc.detail,
                    }
                )
        return BatchPredictionResponse(
            predictions=results,
            total_count=len(request.predictions),
            successful=successful,
            failed=failed,
        )

    @app.get("/")
    def root() -> dict[str, Any]:
        quick_start = {
            "check_health": "GET /health",
            "dataset_summary": "GET /analysis/summary",
            "list_medicines": "GET /medicines",
            "predict": "POST /predict",
        }
        if active_settings.docs_enabled:
            quick_start["interactive_docs"] = "GET /docs"
        return {
            "api": "Pharmacy Demand Forecasting API",
            "version": "0.1.0",
            "status": "running",
            "models_ready": len(registry),
            "quick_start": quick_start,
        }

    return app
