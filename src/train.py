"""Train LSTM: horizon-weighted L1 + OneCycleLR; optional multi-seed ensemble checkpoints."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src import config
from src.models import LSTMForecaster
from src.utils import seed_all, write_json
from src.windows import load_datasets_from_volume


def _horizon_weights(device: torch.device) -> torch.Tensor:
    return torch.linspace(
        config.HORIZON_LOSS_WEIGHT_START,
        config.HORIZON_LOSS_WEIGHT_END,
        config.HORIZON,
        device=device,
        dtype=torch.float32,
    )


def _weighted_l1(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return ((pred - target).abs() * weights.view(1, -1)).mean()


def _mae_kw_scaled_batch(
    y_hat: torch.Tensor,
    y: torch.Tensor,
    scaler,
    device: torch.device,
) -> float:
    y_hat_np = y_hat.detach().cpu().numpy().reshape(-1, 1)
    y_np = y.detach().cpu().numpy().reshape(-1, 1)
    pred_kw = scaler.inverse_transform(y_hat_np).ravel()
    true_kw = scaler.inverse_transform(y_np).ravel()
    return float(np.mean(np.abs(pred_kw - true_kw)))


def _train_one_seed(
    seed: int,
    models_dir: Path,
    device: torch.device,
    scaler: Any,
    dsets: dict,
    splits: dict,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    seed_all(seed)
    train_loader = DataLoader(
        dsets["train"],
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )
    val_loader = DataLoader(dsets["val"], batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

    model = LSTMForecaster(use_sequence_pool=config.USE_SEQUENCE_POOL).to(device)
    opt = AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * config.MAX_EPOCHS
    sched = OneCycleLR(
        opt,
        max_lr=config.LR,
        total_steps=int(total_steps),
        pct_start=config.ONE_CYCLE_PCT_START,
        div_factor=25.0,
        final_div_factor=1e4,
    )

    h_w = _horizon_weights(device)

    best_val = float("inf")
    best_epoch = -1
    patience_left = config.EARLY_STOP_PATIENCE
    best_state: dict[str, Any] | None = None

    for epoch in range(1, config.MAX_EPOCHS + 1):
        model.train()
        for xb, yb in tqdm(train_loader, desc=f"seed={seed} ep {epoch}", leave=False):
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = _weighted_l1(pred, yb, h_w)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            opt.step()
            sched.step()

        model.eval()
        val_maes = []
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                val_losses.append(_weighted_l1(pred, yb, h_w).item())
                val_maes.append(_mae_kw_scaled_batch(pred, yb, scaler, device))

        val_mae = float(np.mean(val_maes))
        _ = float(np.mean(val_losses))

        if val_mae < best_val:
            best_val = val_mae
            best_epoch = epoch
            patience_left = config.EARLY_STOP_PATIENCE
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    cfg_out = {
        "window": config.WINDOW,
        "horizon": config.HORIZON,
        "feature_dim": config.FEATURE_DIM,
        "hidden_size": config.HIDDEN_SIZE,
        "lstm_layers": config.LSTM_LAYERS,
        "dropout": config.DROPOUT,
        "use_sequence_pool": config.USE_SEQUENCE_POOL,
        "seed": seed,
        "best_epoch": best_epoch,
        "val_mae_kw": best_val,
        "meter": splits.get("meter"),
        "n_rows_hourly": splits.get("n_rows_hourly"),
    }

    ckpt = {
        "state_dict": best_state,
        "config": cfg_out,
        "val_mae_kw": best_val,
        "best_epoch": best_epoch,
    }
    member_path = models_dir / f"ensemble_member_seed{seed}.pt"
    torch.save(ckpt, member_path)
    return best_state, cfg_out


def run_training(
    parquet_path: Path,
    splits_json_path: Path,
    scaler_path: Path,
    models_dir: Path,
    seed: int | None = None,
) -> dict[str, Any]:
    """
    Trains each seed in ENSEMBLE_SEEDS, saves member checkpoints + manifest.
    Best member (by val MAE) is also mirrored to lstm_load.pt for single-member inference.
    If ``seed`` is set, only that seed runs (backward compat — rarely needed).
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dsets, splits = load_datasets_from_volume(parquet_path, splits_json_path, scaler_path)
    scaler = joblib.load(scaler_path)

    seeds = (seed,) if seed is not None else tuple(config.ENSEMBLE_SEEDS)
    summaries = []
    for s in seeds:
        _, cfg_one = _train_one_seed(s, models_dir, device, scaler, dsets, splits)
        summaries.append(cfg_one)

    val_maes = [float(x["val_mae_kw"]) for x in summaries]
    best_local = int(np.argmin(val_maes))
    best_seed = summaries[best_local]["seed"]
    best_cfg = summaries[best_local]

    manifest = {
        "ensemble_seeds": list(seeds),
        "member_files": [f"ensemble_member_seed{s}.pt" for s in seeds],
        "member_val_mae_kw": val_maes,
        "best_seed": best_seed,
    }
    write_json(models_dir / "ensemble_manifest.json", manifest)

    best_member = models_dir / f"ensemble_member_seed{best_seed}.pt"
    shutil.copy2(best_member, models_dir / "lstm_load.pt")

    aggregate_cfg = {
        **best_cfg,
        "ensemble_seeds": list(seeds),
        "ensemble_member_val_mae_kw": val_maes,
        "ensemble_best_seed": best_seed,
        "trainer": "onecycle_weighted_l1_multi_seed",
    }
    write_json(models_dir / "config.json", aggregate_cfg)
    return aggregate_cfg
