"""Pydantic request/response models for the forecasting API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


ModelChoice = Literal["single", "ensemble"]


class HealthResponse(BaseModel):
    ok: bool
    models: list[str]
    n_test_windows: int


class TestWindowItem(BaseModel):
    id: int
    t: int
    start_time: str
    first_pred_time: str
    last_pred_time: str


class TestWindowsResponse(BaseModel):
    total: int
    offset: int
    limit: int
    items: list[TestWindowItem]


class MetaResponse(BaseModel):
    meter: str
    time_start: str
    time_end: str
    window: int
    horizon: int
    n_rows_hourly: int
    n_test_windows: int
    test_first_pred_time: str
    test_last_pred_time: str
    available_models: list[str]
    metrics: dict
    manifest: dict | None
    config: dict


class PredictRequest(BaseModel):
    t: int = Field(..., description="Test-window start index (first predicted hour).")
    model: ModelChoice = "ensemble"


class HistoryBlock(BaseModel):
    times: list[str]
    load_kw: list[float]


class ForecastBlock(BaseModel):
    times: list[str]
    pred_kw: list[float]
    actual_kw: list[float]
    naive_kw: list[float]


class ErrorBlock(BaseModel):
    lstm_abs: list[float]
    naive_abs: list[float]


class SummaryBlock(BaseModel):
    mae_kw: float
    rmse_kw: float
    smape_pct: float
    naive_mae_kw: float
    naive_rmse_kw: float
    skill_score: float


class PredictResponse(BaseModel):
    t: int
    first_pred_time: str
    history: HistoryBlock
    forecast: ForecastBlock
    errors: ErrorBlock
    summary: SummaryBlock
    model_used: ModelChoice
    n_members: int
