"""Test metrics vs seasonal naive; optional multi-member ensemble averaged in scaled space."""

from __future__ import annotations

import shutil
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src import config
from src.data import load_processed_parquet
from src.models import LSTMForecaster
from src.utils import read_json, write_json
from src.windows import load_datasets_from_volume


def seasonal_naive_forecast(load_kw: np.ndarray, t: int, H: int) -> np.ndarray:
    """For window ending at t (first pred is index t), return H kW values."""
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


def _load_ckpt(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _checkpoint_paths(models_dir: Path) -> tuple[list[Path], int]:
    """Return list of checkpoints to ensemble and nominal member count hint."""
    models_dir = Path(models_dir)
    manifest_path = models_dir / "ensemble_manifest.json"
    if manifest_path.is_file():
        mf = read_json(manifest_path)
        files = mf.get("member_files") or []
        paths = [models_dir / f for f in files]
        paths = [p for p in paths if p.is_file()]
        if paths:
            return paths, len(paths)
        lone = models_dir / "lstm_load.pt"
        return ([lone] if lone.is_file() else []), 1

    lone = models_dir / "lstm_load.pt"
    if lone.is_file():
        return [lone], 1
    members = sorted(models_dir.glob("ensemble_member_seed*.pt"))
    if members:
        return members, len(members)
    return ([lone] if lone.exists() else []), 0


def _build_models(ckpts: list[dict], device: torch.device) -> list[torch.nn.Module]:
    models: list[torch.nn.Module] = []
    ref = ckpts[0]["config"]
    for ck in ckpts:
        cfg = ck["config"]
        m = LSTMForecaster(
            in_dim=cfg.get("feature_dim", ref.get("feature_dim", config.FEATURE_DIM)),
            hidden=cfg.get("hidden_size", ref.get("hidden_size", config.HIDDEN_SIZE)),
            layers=cfg.get("lstm_layers", ref.get("lstm_layers", config.LSTM_LAYERS)),
            horizon=cfg.get("horizon", ref.get("horizon", config.HORIZON)),
            dropout=cfg.get("dropout", ref.get("dropout", config.DROPOUT)),
            use_sequence_pool=cfg.get("use_sequence_pool", ref.get("use_sequence_pool", False)),
        ).to(device)
        m.load_state_dict(ck["state_dict"])
        m.eval()
        models.append(m)
    return models


def run_eval(
    parquet_path: Path,
    splits_json_path: Path,
    scaler_path: Path,
    models_dir: Path,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = joblib.load(scaler_path)
    splits = read_json(splits_json_path)
    df = load_processed_parquet(Path(parquet_path))
    load_kw = df["load_kw"].to_numpy(dtype=np.float64)

    dsets, _ = load_datasets_from_volume(parquet_path, splits_json_path, scaler_path)
    test_loader = DataLoader(dsets["test"], batch_size=config.BATCH_SIZE, shuffle=False)

    models_dir = Path(models_dir)
    paths, _ = _checkpoint_paths(models_dir)
    if not paths:
        raise FileNotFoundError(
            f"No checkpoints found in {models_dir}. Expected lstm_load.pt, ensemble_manifest.json, or ensemble_member_seed*.pt"
        )
    ckpts = [_load_ckpt(p, device) for p in paths]
    n_members_eff = len(ckpts)
    models = _build_models(ckpts, device)

    all_pred = []
    all_true = []
    all_naive = []
    test_indices = dsets["test"].indices

    idx_ptr = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            y_np = yb.numpy()
            acc = torch.zeros((xb.shape[0], config.HORIZON), device=device, dtype=torch.float32)
            for m in models:
                acc += m(xb)
            pred_s = (acc / max(1, len(models))).cpu().numpy()
            bs = pred_s.shape[0]
            for b in range(bs):
                t = int(test_indices[idx_ptr + b])
                nv = seasonal_naive_forecast(load_kw, t, config.HORIZON)
                all_naive.append(nv)
            all_pred.append(pred_s)
            all_true.append(y_np)
            idx_ptr += bs

    pred_scaled = np.vstack(all_pred)
    true_scaled = np.vstack(all_true)
    naive_kw = np.stack(all_naive, axis=0)

    pred_kw = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).reshape(pred_scaled.shape)
    true_kw = scaler.inverse_transform(true_scaled.reshape(-1, 1)).reshape(true_scaled.shape)

    err_lstm = pred_kw - true_kw
    err_naive = naive_kw - true_kw

    mae_lstm = float(np.mean(np.abs(err_lstm)))
    rmse_lstm = float(np.sqrt(np.mean(err_lstm**2)))
    mae_naive = float(np.mean(np.abs(err_naive)))
    rmse_naive = float(np.sqrt(np.mean(err_naive**2)))
    skill = float(1.0 - mae_lstm / (mae_naive + 1e-12))
    sm = smape(true_kw.ravel(), pred_kw.ravel())
    mean_abs_load = float(np.mean(np.abs(true_kw)))
    nmae_pct = float(100.0 * mae_lstm / (mean_abs_load + 1e-12))

    label = "lstm_ensemble_mean" if n_members_eff > 1 else "lstm_single"

    metrics = {
        "test_mae_kw_lstm": mae_lstm,
        "test_rmse_kw_lstm": rmse_lstm,
        "test_smape_pct_lstm": sm,
        "test_mae_kw_naive": mae_naive,
        "test_rmse_kw_naive": rmse_naive,
        "skill_score_vs_naive": skill,
        "n_test_windows": int(pred_kw.shape[0]),
        "test_mean_abs_load_kw": mean_abs_load,
        "test_nmae_pct_of_mean_load_lstm": nmae_pct,
        "ensemble_n_members": int(n_members_eff),
        "eval_label": label,
    }
    out_dir = Path(models_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "metrics.json", metrics)

    shutil.copy2(scaler_path, out_dir / "scaler.joblib")
    shutil.copy2(splits_json_path, out_dir / "splits.json")

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    week_h = 7 * 24
    m = min(week_h, pred_kw.shape[0])
    h_axis = np.arange(m)
    ttl = f"Test: first week ({label}, n={n_members_eff})"
    plt.figure(figsize=(14, 4))
    plt.plot(h_axis, true_kw[:m, 0], label="actual kW", alpha=0.9)
    plt.plot(h_axis, pred_kw[:m, 0], label="pred kW (1-step)", alpha=0.9)
    plt.plot(h_axis, naive_kw[:m, 0], label="seasonal naive kW (1-step)", alpha=0.9)
    plt.legend()
    plt.xlabel("hours from first test window (contiguous)")
    plt.ylabel("kW")
    plt.title(ttl)
    plt.tight_layout()
    plt.savefig(fig_dir / "forecast_week.png", dpi=150)
    plt.close()

    hours_list: list[int] = []
    resids: list[float] = []
    for wi, t in enumerate(test_indices):
        for k in range(config.HORIZON):
            hours_list.append(int(df.index[t + k].hour))
            resids.append(float(pred_kw[wi, k] - true_kw[wi, k]))
    hres = np.array(hours_list)
    res = np.array(resids)
    plt.figure(figsize=(10, 4))
    plt.boxplot(
        [res[hres == h] for h in range(24)],
        labels=[str(h) for h in range(24)],
        showfliers=False,
    )
    plt.xlabel("hour of day")
    plt.ylabel("residual kW (pred - actual)")
    plt.title(f"Residuals by hour ({label})")
    plt.tight_layout()
    plt.savefig(fig_dir / "residuals_by_hour.png", dpi=150)
    plt.close()

    return metrics
