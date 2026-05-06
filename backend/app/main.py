"""FastAPI app: serves the trained LSTM load forecaster over HTTP."""

from __future__ import annotations

import os
import random
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from src import config

from app import inference, schemas

_state: inference.LoadedState | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _state
    _state = inference.load_state()
    yield
    _state = None


app = FastAPI(
    title="Electricity Load Forecasting API",
    version="1.0.0",
    lifespan=lifespan,
)

_cors_extra = os.environ.get("EXTRA_CORS_ORIGINS", "").split(",")
_cors_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    *[o.strip() for o in _cors_extra if o.strip()],
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _require_state() -> inference.LoadedState:
    if _state is None:
        raise HTTPException(status_code=503, detail="Model state not loaded yet.")
    return _state


def _model_labels(state: inference.LoadedState) -> list[str]:
    labels = ["single"]
    if len(state.ensemble_models) > 1 and state.ensemble_models[0] is not state.single_model:
        labels.append("ensemble")
    elif len(state.ensemble_models) > 1:
        labels.append("ensemble")
    return labels


@app.get("/api/health", response_model=schemas.HealthResponse)
def health() -> schemas.HealthResponse:
    state = _require_state()
    return schemas.HealthResponse(
        ok=True,
        models=_model_labels(state),
        n_test_windows=int(state.test_indices.size),
    )


@app.get("/api/meta", response_model=schemas.MetaResponse)
def meta() -> schemas.MetaResponse:
    state = _require_state()
    test_idx = state.test_indices
    if test_idx.size == 0:
        raise HTTPException(status_code=500, detail="No test windows available.")
    first_t = int(test_idx[0])
    last_t = int(test_idx[-1])
    return schemas.MetaResponse(
        meter=state.splits.get("meter", state.config.get("meter", "unknown")),
        time_start=str(state.splits.get("time_start", "")),
        time_end=str(state.splits.get("time_end", "")),
        window=config.WINDOW,
        horizon=config.HORIZON,
        n_rows_hourly=int(state.splits.get("n_rows_hourly", len(state.df))),
        n_test_windows=int(test_idx.size),
        test_first_pred_time=state.times_iso[first_t],
        test_last_pred_time=state.times_iso[last_t],
        available_models=_model_labels(state),
        metrics=state.metrics,
        manifest=state.manifest,
        config=state.config,
    )


@app.get("/api/test-windows", response_model=schemas.TestWindowsResponse)
def test_windows(
    limit: int = Query(200, ge=1, le=2000),
    offset: int = Query(0, ge=0),
    q: str | None = Query(None, description="Filter by ISO date prefix on first_pred_time, e.g. 2014-08."),
    random_one: bool = Query(False, alias="random"),
) -> schemas.TestWindowsResponse:
    state = _require_state()
    test_idx = state.test_indices
    if test_idx.size == 0:
        return schemas.TestWindowsResponse(total=0, offset=0, limit=limit, items=[])

    indices = test_idx.tolist()

    if random_one:
        t = int(random.choice(indices))
        item = _to_window_item(state, t, indices.index(t))
        return schemas.TestWindowsResponse(total=1, offset=0, limit=1, items=[item])

    if q:
        q_norm = q.strip()
        filtered = [
            (i, t) for i, t in enumerate(indices)
            if state.times_iso[t].startswith(q_norm)
        ]
    else:
        filtered = list(enumerate(indices))

    total = len(filtered)
    page = filtered[offset : offset + limit]
    items = [_to_window_item(state, int(t), int(i)) for (i, t) in page]
    return schemas.TestWindowsResponse(total=total, offset=offset, limit=limit, items=items)


def _to_window_item(state: inference.LoadedState, t: int, idx_id: int) -> schemas.TestWindowItem:
    H = config.HORIZON
    W = config.WINDOW
    return schemas.TestWindowItem(
        id=idx_id,
        t=t,
        start_time=state.times_iso[t - W],
        first_pred_time=state.times_iso[t],
        last_pred_time=state.times_iso[t + H - 1],
    )


@app.post("/api/predict", response_model=schemas.PredictResponse)
def predict(req: schemas.PredictRequest) -> schemas.PredictResponse:
    state = _require_state()
    if not inference.is_valid_t(state, req.t):
        raise HTTPException(
            status_code=400,
            detail=f"t={req.t} is not a valid test-window index. "
                   f"Valid range: [{int(state.test_indices[0])}, {int(state.test_indices[-1])}].",
        )
    if req.model == "ensemble" and len(state.ensemble_models) <= 1:
        # Allowed; will simply use the single model.
        pass
    payload = inference.predict_window(state, int(req.t), req.model)
    return schemas.PredictResponse(**payload)
