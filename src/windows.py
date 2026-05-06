"""Sliding-window dataset: (batch, W, 8) -> target (batch, H)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src import config
from src.data import load_processed_parquet
from src.utils import read_json


SplitName = Literal["train", "val", "test"]


def build_feature_arrays(df: pd.DataFrame, scaler) -> tuple[np.ndarray, np.ndarray]:
    """Full-length arrays: load_scaled (n,), feat_rest (n, 7) -> concat to (n,8) in dataset."""
    load = df["load_kw"].to_numpy(dtype=np.float64).reshape(-1, 1)
    load_scaled = scaler.transform(load).astype(np.float32).ravel()
    idx = df.index
    hour = idx.hour.to_numpy()
    dow = idx.dayofweek.to_numpy()
    sin_h = np.sin(2 * np.pi * hour / 24.0).astype(np.float32)
    cos_h = np.cos(2 * np.pi * hour / 24.0).astype(np.float32)
    sin_d = np.sin(2 * np.pi * dow / 7.0).astype(np.float32)
    cos_d = np.cos(2 * np.pi * dow / 7.0).astype(np.float32)
    dst = df[["is_march_dst_short", "is_october_dst_long", "is_dst_window"]].to_numpy(
        dtype=np.float32
    )
    rest = np.column_stack([sin_h, cos_h, sin_d, cos_d, dst])
    return load_scaled, rest


class WindowDataset(Dataset):
    def __init__(
        self,
        load_scaled: np.ndarray,
        feat_rest: np.ndarray,
        indices: np.ndarray,
        W: int = config.WINDOW,
        H: int = config.HORIZON,
    ):
        self.load_scaled = load_scaled
        self.feat_rest = feat_rest
        self.indices = indices
        self.W = W
        self.H = H

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        t = int(self.indices[i])
        sl = slice(t - self.W, t)
        load_win = self.load_scaled[sl][:, None]
        rest_win = self.feat_rest[sl]
        x = np.concatenate([load_win, rest_win], axis=1).astype(np.float32)
        y = self.load_scaled[t : t + self.H].astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)


def split_indices(val_start: int, test_start: int, n: int, W: int, H: int) -> dict[str, np.ndarray]:
    """Valid window end indices t (first predicted hour) per split."""
    train_t = np.arange(W, val_start - H + 1, dtype=np.int64)
    val_t = np.arange(max(W, val_start), test_start - H + 1, dtype=np.int64)
    test_t = np.arange(max(W, test_start), n - H + 1, dtype=np.int64)
    if len(train_t) == 0 or len(val_t) == 0 or len(test_t) == 0:
        raise ValueError(
            f"Empty split: train={len(train_t)} val={len(val_t)} test={len(test_t)} "
            f"(n={n}, val_start={val_start}, test_start={test_start}, W={W}, H={H})"
        )
    return {"train": train_t, "val": val_t, "test": test_t}


def load_datasets_from_volume(
    parquet_path: Path,
    splits_json_path: Path,
    scaler_path: Path,
) -> tuple[dict[str, WindowDataset], dict]:
    df = load_processed_parquet(Path(parquet_path))
    splits = read_json(splits_json_path)
    scaler = joblib.load(scaler_path)
    n = splits["n_rows_hourly"]
    if len(df) != n:
        # tolerate mismatch if parquet row count differs slightly
        n = len(df)
    val_start = int(splits["val_start"])
    test_start = int(splits["test_start"])
    load_scaled, feat_rest = build_feature_arrays(df, scaler)
    idx_map = split_indices(val_start, test_start, n, config.WINDOW, config.HORIZON)
    dsets = {
        name: WindowDataset(load_scaled, feat_rest, idx_map[name])
        for name in ("train", "val", "test")
    }
    return dsets, splits
