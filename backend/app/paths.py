"""Resolve repo-root artifact locations regardless of CWD."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

ARTIFACTS_BEST_DIR = REPO_ROOT / "artifacts_best"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"

PROCESSED_PARQUET = ARTIFACTS_BEST_DIR / "processed" / "electricity_hourly.parquet"


def processed_parquet_search_paths() -> list[Path]:
    """Ordered list of places we look for `electricity_hourly.parquet`."""
    env = os.environ.get("ELECTRICITY_PROCESSED_PARQUET", "").strip()
    base = [
        Path(env).expanduser().resolve() if env else None,
        ARTIFACTS_BEST_DIR / "processed" / "electricity_hourly.parquet",
        ARTIFACTS_DIR / "processed" / "electricity_hourly.parquet",
        REPO_ROOT / "data" / "processed" / "electricity_hourly.parquet",
    ]
    seen: set[Path] = set()
    out: list[Path] = []
    for p in base:
        if p is None:
            continue
        try:
            r = Path(p).resolve()
        except OSError:
            continue
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def resolve_processed_parquet() -> Path | None:
    """Return first existing processed hourly parquet path, or None."""
    for candidate in processed_parquet_search_paths():
        if candidate.is_file():
            return candidate
    return None

BEST_MODELS_DIR = ARTIFACTS_BEST_DIR / "models"
BEST_CHECKPOINT = BEST_MODELS_DIR / "lstm_load.pt"
BEST_SCALER = BEST_MODELS_DIR / "scaler.joblib"
BEST_SPLITS = BEST_MODELS_DIR / "splits.json"
BEST_CONFIG = BEST_MODELS_DIR / "config.json"
BEST_METRICS = BEST_MODELS_DIR / "metrics.json"
BEST_MANIFEST = BEST_MODELS_DIR / "ensemble_manifest.json"

ENSEMBLE_MODELS_DIR = ARTIFACTS_DIR / "models"


def ensemble_member_paths() -> list[Path]:
    """Return ensemble member checkpoint paths sorted by seed (matches eval.py logic)."""
    if not ENSEMBLE_MODELS_DIR.is_dir():
        return []
    return sorted(ENSEMBLE_MODELS_DIR.glob("ensemble_member_seed*.pt"))
