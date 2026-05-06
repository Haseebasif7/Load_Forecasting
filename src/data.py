"""
Load UCI electricity CSV from zip, hourly resample, trim leading zeros,
DST flags (Europe/Lisbon), chronological splits, RobustScaler on train load only.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from src import config
from src.utils import write_json


def _mt_columns(df: pd.DataFrame) -> list[str]:
    return sorted([c for c in df.columns if str(c).startswith("MT_")])


def _first_active_index(series: pd.Series, eps: float) -> int | None:
    s = pd.to_numeric(series, errors="coerce").fillna(0.0).to_numpy()
    m = s > eps
    if not m.any():
        return None
    return int(np.argmax(m))


def _max_zero_run_hours_after(series: pd.Series, start_i: int, eps: float) -> int:
    """Longest consecutive stretch of near-zero load (in rows = hours) after start_i."""
    s = pd.to_numeric(series.iloc[start_i:], errors="coerce").fillna(0.0)
    is_zero = (s <= eps).astype(int).to_numpy()
    if is_zero.size == 0:
        return 0
    best = cur = 0
    for v in is_zero:
        if v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def auto_pick_meter(df: pd.DataFrame, time_col: str, eps: float = config.EPS_LOAD) -> str:
    """
    First MT_* whose first activity is on/before 2011-04-01 and no >24h consecutive
    near-zero runs after activation.
    """
    t = pd.to_datetime(df[time_col])
    cutoff = pd.Timestamp("2011-04-01 23:59:59")
    for col in _mt_columns(df):
        fi = _first_active_index(df[col], eps)
        if fi is None:
            continue
        if t.iloc[fi] > cutoff:
            continue
        if _max_zero_run_hours_after(df[col], fi, eps) > 24:
            continue
        return col
    # Fallback: first column with any activity
    for col in _mt_columns(df):
        if _first_active_index(df[col], eps) is not None:
            return col
    raise ValueError("No MT_* column with non-zero load found.")


def _normalize_header_names(df: pd.DataFrame) -> pd.DataFrame:
    """First CSV column is always wall time; normalize to ``time``."""
    df = df.copy()
    names = [str(c).strip().strip('"') for c in df.columns]
    names[0] = "time"
    df.columns = names
    return df


def _meter_column_index(zip_path: Path, member: str, meter: str) -> int:
    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open(member, "r") as raw:
            hdr = pd.read_csv(raw, sep=";", decimal=",", nrows=0)
    strip = [str(c).strip().strip('"') for c in hdr.columns]
    try:
        return int(strip.index(meter))
    except ValueError as e:
        raise ValueError(f"Meter {meter!r} not found in zip header") from e


def _read_probe_from_zip(
    zip_path: Path,
    member: str = config.ZIP_MEMBER,
    nrows: int = 200_000,
) -> pd.DataFrame:
    """Read first nrows for meter auto-pick (avoids loading all 370 series)."""
    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open(member, "r") as raw:
            df = pd.read_csv(
                raw,
                sep=";",
                decimal=",",
                nrows=nrows,
                low_memory=False,
            )
    return _normalize_header_names(df)


def _read_series_from_zip(
    zip_path: Path,
    member: str,
    meter: str,
) -> pd.DataFrame:
    """Read only timestamp + one MT_* column (memory-safe)."""
    zip_path = Path(zip_path)
    mt_idx = _meter_column_index(zip_path, member, meter)
    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open(member, "r") as raw:
            df = pd.read_csv(
                raw,
                sep=";",
                decimal=",",
                usecols=[0, mt_idx],
                low_memory=False,
            )
    df.columns = ["time", "load_kw"]
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time")
    df = df.drop_duplicates(subset=["time"], keep="first")
    return df.reset_index(drop=True)


def add_dst_flags(index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Three binary columns aligned to naive Portuguese wall-clock times in the CSV.
    Uses tz_localize('Europe/Lisbon').

    Fall DST repeats a wall-clock hour; ``ambiguous=False`` picks the standard-time
    (winter) interpretation so localization is deterministic (``infer`` can raise
    AmbiguousTimeError on e.g. 2011-10-30 01:00). Spring gap uses ``shift_forward``.

    is_march_dst_short: hour where UTC offset jumps forward (spring forward).
    is_october_dst_long: hour where UTC offset jumps backward (fall back).
    is_dst_window: union of the two transition-hour markers.
    """
    loc = index.tz_localize(
        "Europe/Lisbon",
        ambiguous=False,
        nonexistent="shift_forward",
    )
    off_sec = pd.Series(
        [t.utcoffset().total_seconds() for t in loc],
        index=index,
        dtype="float64",
    )
    prev = off_sec.shift(1)
    change = (off_sec != prev) & prev.notna()
    spring = change & (off_sec > prev)
    fall = change & (off_sec < prev)
    out = pd.DataFrame(
        {
            "is_march_dst_short": spring.fillna(False).astype(np.int8),
            "is_october_dst_long": fall.fillna(False).astype(np.int8),
            "is_dst_window": (spring.fillna(False) | fall.fillna(False)).astype(np.int8),
        },
        index=index,
    )
    return out


def preprocess_from_zip(
    zip_path: Path,
    processed_parquet: Path,
    splits_json: Path,
    scaler_path: Path,
    meter: str | None = None,
) -> dict[str, Any]:
    """
    Full preprocessing. Writes hourly parquet (load_kw + DST flags), splits.json, scaler.joblib.
    Returns metadata dict for config.json.
    """
    zip_path = Path(zip_path)
    probe = _read_probe_from_zip(zip_path)
    meter = meter or auto_pick_meter(probe, "time")
    if meter not in probe.columns:
        raise ValueError(f"Meter column {meter!r} not in probe header.")

    df = _read_series_from_zip(zip_path, config.ZIP_MEMBER, meter)
    load = pd.to_numeric(df["load_kw"], errors="coerce").fillna(0.0)
    dfp = pd.DataFrame({"time": df["time"], "load_kw": load})
    dfp = dfp.set_index("time").sort_index()
    hourly = dfp.resample("h").mean()
    hourly = hourly.dropna(subset=["load_kw"])

    fi = _first_active_index(hourly["load_kw"], config.EPS_LOAD)
    if fi is None:
        raise ValueError("Hourly series is all zeros.")
    hourly = hourly.iloc[fi:].copy()

    dst = add_dst_flags(hourly.index)
    hourly = hourly.join(dst, how="left").fillna(0)

    n = len(hourly)
    if n < config.WINDOW + config.HORIZON + 1000:
        raise ValueError(f"Not enough hourly rows after trim: {n}")

    val_start = int(n * config.TRAIN_FRAC)
    test_start = int(n * (config.TRAIN_FRAC + config.VAL_FRAC))
    if val_start < config.WINDOW + config.HORIZON + 1:
        raise ValueError(f"val_start too small for window/horizon: val_start={val_start}, n={n}")
    if test_start <= val_start + config.HORIZON:
        raise ValueError(f"test_start must allow val windows: val_start={val_start}, test_start={test_start}")
    if test_start >= n - config.HORIZON:
        raise ValueError(f"test_start too late for horizon: test_start={test_start}, n={n}")

    train_load = hourly["load_kw"].iloc[:val_start].values.reshape(-1, 1)
    scaler = RobustScaler()
    scaler.fit(train_load)

    meta = {
        "meter": meter,
        "n_rows_hourly": n,
        "val_start": val_start,
        "test_start": test_start,
        "time_start": str(hourly.index.min()),
        "time_end": str(hourly.index.max()),
    }
    processed_parquet = Path(processed_parquet)
    splits_json = Path(splits_json)
    scaler_path = Path(scaler_path)
    processed_parquet.parent.mkdir(parents=True, exist_ok=True)
    hourly.reset_index().to_parquet(processed_parquet, index=False)
    write_json(
        splits_json,
        {
            **meta,
            "train_rows": [0, val_start],
            "val_rows": [val_start, test_start],
            "test_rows": [test_start, n],
        },
    )
    joblib.dump(scaler, scaler_path)
    return meta


def load_processed_parquet(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["time"] = pd.to_datetime(df["time"])
    return df.set_index("time").sort_index()
