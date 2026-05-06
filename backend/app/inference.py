"""Model loading and per-window prediction (single + ensemble)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from src import config
from src.data import load_processed_parquet
from src.models import LSTMForecaster
from src.utils import read_json
from src.windows import build_feature_arrays, split_indices

from app import paths


def seasonal_naive_forecast(load_kw: np.ndarray, t: int, H: int) -> np.ndarray:
    """Same semantics as src.eval.seasonal_naive_forecast (avoid importing matplotlib via src.eval)."""
    out = np.empty(H, dtype=np.float64)
    for k in range(H):
        i = t + k
        j = i - 168
        if j >= 0:
            out[k] = load_kw[j]
        else:
            j2 = i - 24
            out[k] = load_kw[j2] if j2 >= 0 else np.nan
    return out


def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    denom = np.abs(y_true) + np.abs(y_pred) + eps
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom) * 100.0)


@dataclass
class LoadedState:
    """Everything kept in memory between requests."""

    df: pd.DataFrame
    load_kw: np.ndarray
    load_scaled: np.ndarray
    feat_rest: np.ndarray
    times_iso: list[str]
    test_indices: np.ndarray
    splits: dict
    config: dict
    metrics: dict
    manifest: dict | None
    scaler: object
    single_model: torch.nn.Module | None
    ensemble_models: list[torch.nn.Module] = field(default_factory=list)
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))


def _load_ckpt(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _build_model_from_ckpt(ckpt: dict, device: torch.device) -> torch.nn.Module:
    cfg = ckpt.get("config", {})
    m = LSTMForecaster(
        in_dim=cfg.get("feature_dim", config.FEATURE_DIM),
        hidden=cfg.get("hidden_size", config.HIDDEN_SIZE),
        layers=cfg.get("lstm_layers", config.LSTM_LAYERS),
        horizon=cfg.get("horizon", config.HORIZON),
        dropout=cfg.get("dropout", config.DROPOUT),
        use_sequence_pool=cfg.get("use_sequence_pool", config.USE_SEQUENCE_POOL),
    ).to(device)
    m.load_state_dict(ckpt["state_dict"])
    m.eval()
    return m


def load_state() -> LoadedState:
    """Load parquet, scaler, splits, and model checkpoints once at startup."""
    parquet_path = paths.resolve_processed_parquet()
    if parquet_path is None:
        searched = "\n".join(f"  - {p}" for p in paths.processed_parquet_search_paths())
        default_dest = paths.ARTIFACTS_BEST_DIR / "processed" / "electricity_hourly.parquet"
        raise FileNotFoundError(
            "Processed hourly parquet not found. Checked (in order):\n"
            f"{searched}\n\n"
            "Either set ELECTRICITY_PROCESSED_PARQUET to the file path, or download:\n\n"
            f"  modal volume get electricity-data /processed/electricity_hourly.parquet \\\n"
            f"    {default_dest} --force\n"
        )
    if not paths.BEST_SCALER.is_file():
        raise FileNotFoundError(f"Scaler missing at {paths.BEST_SCALER}")
    if not paths.BEST_SPLITS.is_file():
        raise FileNotFoundError(f"Splits missing at {paths.BEST_SPLITS}")
    if not paths.BEST_CHECKPOINT.is_file():
        raise FileNotFoundError(f"Best checkpoint missing at {paths.BEST_CHECKPOINT}")

    device = torch.device("cpu")
    df = load_processed_parquet(parquet_path)
    splits = read_json(paths.BEST_SPLITS)
    scaler = joblib.load(paths.BEST_SCALER)
    cfg = read_json(paths.BEST_CONFIG) if paths.BEST_CONFIG.is_file() else {}
    metrics = read_json(paths.BEST_METRICS) if paths.BEST_METRICS.is_file() else {}
    manifest = read_json(paths.BEST_MANIFEST) if paths.BEST_MANIFEST.is_file() else None

    n = len(df)
    val_start = int(splits["val_start"])
    test_start = int(splits["test_start"])
    load_scaled, feat_rest = build_feature_arrays(df, scaler)
    idx_map = split_indices(val_start, test_start, n, config.WINDOW, config.HORIZON)

    single_ckpt = _load_ckpt(paths.BEST_CHECKPOINT, device)
    single_model = _build_model_from_ckpt(single_ckpt, device)

    ensemble_models: list[torch.nn.Module] = []
    member_paths = paths.ensemble_member_paths()
    for p in member_paths:
        ck = _load_ckpt(p, device)
        ensemble_models.append(_build_model_from_ckpt(ck, device))
    if not ensemble_models:
        ensemble_models = [single_model]

    times_iso = [t.isoformat() for t in df.index]
    load_kw = df["load_kw"].to_numpy(dtype=np.float64)

    return LoadedState(
        df=df,
        load_kw=load_kw,
        load_scaled=load_scaled,
        feat_rest=feat_rest,
        times_iso=times_iso,
        test_indices=idx_map["test"],
        splits=splits,
        config=cfg,
        metrics=metrics,
        manifest=manifest,
        scaler=scaler,
        single_model=single_model,
        ensemble_models=ensemble_models,
        device=device,
    )


def is_valid_t(state: LoadedState, t: int) -> bool:
    if state.test_indices.size == 0:
        return False
    return bool(t in state.test_indices)


def _build_input(state: LoadedState, t: int) -> torch.Tensor:
    W = config.WINDOW
    load_win = state.load_scaled[t - W : t][:, None]
    rest_win = state.feat_rest[t - W : t]
    x = np.concatenate([load_win, rest_win], axis=1).astype(np.float32)
    return torch.from_numpy(x).unsqueeze(0).to(state.device)


def predict_window(state: LoadedState, t: int, model_choice: str) -> dict:
    """Run inference for one test window and assemble the response payload."""
    H = config.HORIZON
    W = config.WINDOW

    models = state.ensemble_models if model_choice == "ensemble" else [state.single_model]
    n_members = len(models)

    x = _build_input(state, t)
    with torch.no_grad():
        acc = torch.zeros((1, H), dtype=torch.float32, device=state.device)
        for m in models:
            acc += m(x)
        pred_scaled = (acc / max(1, n_members)).cpu().numpy().reshape(-1, 1)

    pred_kw = state.scaler.inverse_transform(pred_scaled).ravel()
    actual_kw = state.load_kw[t : t + H]
    naive_kw = seasonal_naive_forecast(state.load_kw, t, H)

    err_lstm = pred_kw - actual_kw
    err_naive = naive_kw - actual_kw

    mae = float(np.mean(np.abs(err_lstm)))
    rmse = float(np.sqrt(np.mean(err_lstm ** 2)))
    sm = smape(actual_kw, pred_kw)
    naive_mae = float(np.mean(np.abs(err_naive)))
    naive_rmse = float(np.sqrt(np.mean(err_naive ** 2)))
    skill = float(1.0 - mae / (naive_mae + 1e-12))

    history_times = state.times_iso[t - W : t]
    forecast_times = state.times_iso[t : t + H]
    history_kw = state.load_kw[t - W : t].tolist()

    return {
        "t": int(t),
        "first_pred_time": state.times_iso[t],
        "history": {
            "times": history_times,
            "load_kw": [float(v) for v in history_kw],
        },
        "forecast": {
            "times": forecast_times,
            "pred_kw": [float(v) for v in pred_kw],
            "actual_kw": [float(v) for v in actual_kw],
            "naive_kw": [float(v) for v in naive_kw],
        },
        "errors": {
            "lstm_abs": [float(abs(v)) for v in err_lstm],
            "naive_abs": [float(abs(v)) for v in err_naive],
        },
        "summary": {
            "mae_kw": mae,
            "rmse_kw": rmse,
            "smape_pct": sm,
            "naive_mae_kw": naive_mae,
            "naive_rmse_kw": naive_rmse,
            "skill_score": skill,
        },
        "model_used": model_choice,
        "n_members": n_members,
    }
