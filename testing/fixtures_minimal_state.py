"""Synthetic LoadedState for API tests (no real artifacts on disk)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler

from app.inference import LoadedState
from src import config
from src.models import LSTMForecaster
from src.windows import build_feature_arrays, split_indices


def build_minimal_loaded_state(*, seed: int = 42) -> LoadedState:
    """Build a small in-memory state matching production field layout (`inference.load_state`)."""
    rng = np.random.default_rng(seed)
    W, H = config.WINDOW, config.HORIZON
    n = 4000
    idx = pd.date_range("2011-01-01", periods=n, freq="h", tz="UTC")
    load_kw = (
        80.0
        + 15.0 * np.sin(np.arange(n, dtype=np.float64) * 2 * np.pi / 168.0)
        + rng.normal(0.0, 2.0, n)
    )
    load_kw = np.clip(load_kw, 0.5, None)

    df = pd.DataFrame(
        {
            "load_kw": load_kw,
            "is_march_dst_short": np.zeros(n, dtype=np.float32),
            "is_october_dst_long": np.zeros(n, dtype=np.float32),
            "is_dst_window": np.zeros(n, dtype=np.float32),
        },
        index=idx,
    )

    val_start = 1200
    test_start = 2800
    scaler = RobustScaler()
    scaler.fit(load_kw[:val_start].reshape(-1, 1))
    load_scaled, feat_rest = build_feature_arrays(df, scaler)
    idx_map = split_indices(val_start, test_start, n, W, H)
    times_iso = [t.isoformat() for t in df.index]

    splits = {
        "meter": "MT_TEST",
        "time_start": times_iso[0],
        "time_end": times_iso[-1],
        "n_rows_hourly": n,
        "val_start": val_start,
        "test_start": test_start,
    }

    model = LSTMForecaster(hidden=32, layers=1, dropout=0.0)
    model.eval()
    device = torch.device("cpu")

    return LoadedState(
        df=df,
        load_kw=load_kw,
        load_scaled=load_scaled,
        feat_rest=feat_rest,
        times_iso=times_iso,
        test_indices=idx_map["test"],
        splits=splits,
        config={"meter": "MT_TEST", "feature_dim": config.FEATURE_DIM},
        metrics={"test_mae_kw_lstm": 1.23, "test_mae_kw_naive": 1.5},
        manifest=None,
        scaler=scaler,
        single_model=model,
        ensemble_models=[model],
        device=device,
    )
