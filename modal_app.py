"""
Modal entrypoints for electricity load forecasting (preprocess, train, evaluate).

Prereq: upload raw zip to the Volume once:
  modal volume put electricity-data ./data/electricityloaddiagrams20112014.zip /raw/

Remotes (**always use these `*_cli` entrypoints**) so Modal actually runs containers
(``modal run …::app.evaluate`` alone can sync the app without executing the remote):

  modal run modal_app.py::preprocess_cli
  modal run modal_app.py::train_cli
  modal run modal_app.py::evaluate_cli
  modal run modal_app.py::download_artifacts

Or: modal run modal_app.py::all_steps
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent
VOLUME_NAME = "electricity-data"
VOL_MOUNT = "/vol"

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _volume_get_models(dest: Path) -> None:
    """Download `/models` from the Volume with ``--force`` (safe to re-run)."""
    dest = Path(dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    cmd = ["modal", "volume", "get", "--force", VOLUME_NAME, "/models", str(dest)]
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "pandas==2.2.2",
        "numpy==1.26.4",
        "scikit-learn==1.4.2",
        "pyarrow==16.1.0",
        "matplotlib==3.8.4",
        "joblib==1.4.2",
        "tqdm==4.66.4",
        "torch==2.2.2+cu121",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    # Ship `src` as an importable package (rebuilds when Python under `src/` changes).
    .add_local_python_source("src", copy=True)
)

app = modal.App("electricity-load-forecast")


@app.function(
    image=image,
    volumes={VOL_MOUNT: volume},
    cpu=4.0,
    memory=16384,
    timeout=1800,
)
def preprocess(meter: str | None = None) -> dict:
    from pathlib import Path as P

    from src.data import preprocess_from_zip

    print("[preprocess] starting", flush=True)
    raw_zip = P(VOL_MOUNT) / "raw" / "electricityloaddiagrams20112014.zip"
    if not raw_zip.is_file():
        raise FileNotFoundError(
            f"Missing {raw_zip}. Run: modal volume put {VOLUME_NAME} "
            "./data/electricityloaddiagrams20112014.zip /raw/electricityloaddiagrams20112014.zip"
        )
    proc_dir = P(VOL_MOUNT) / "processed"
    proc_dir.mkdir(parents=True, exist_ok=True)
    meta = preprocess_from_zip(
        raw_zip,
        proc_dir / "electricity_hourly.parquet",
        proc_dir / "splits.json",
        proc_dir / "scaler.joblib",
        meter=meter,
    )
    volume.commit()
    print("[preprocess] done", meta, flush=True)
    return meta


@app.function(
    image=image,
    volumes={VOL_MOUNT: volume},
    gpu="A10G",
    timeout=32400,
    memory=49152,
)
def train() -> dict:
    from pathlib import Path as P

    from src.train import run_training

    base = P(VOL_MOUNT)
    proc = base / "processed"
    for name in ("electricity_hourly.parquet", "splits.json", "scaler.joblib"):
        if not (proc / name).is_file():
            raise FileNotFoundError(f"Missing {proc / name}. Run preprocess first.")
    models_dir = base / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    # Runs all seeds in src.config.ENSEMBLE_SEEDS (default: 3) + OneCycle LR.
    out = run_training(
        proc / "electricity_hourly.parquet",
        proc / "splits.json",
        proc / "scaler.joblib",
        models_dir,
        seed=None,
    )
    volume.commit()
    return out


@app.function(
    image=image,
    volumes={VOL_MOUNT: volume},
    gpu="A10G",
    timeout=10800,
    memory=49152,
)
def train_quick() -> dict:
    """Single-seed trainer for debugging (still uses improved recipe from config minus extra seeds)."""
    from pathlib import Path as P

    from src import config as cf
    from src.train import run_training

    base = P(VOL_MOUNT)
    proc = base / "processed"
    for name in ("electricity_hourly.parquet", "splits.json", "scaler.joblib"):
        if not (proc / name).is_file():
            raise FileNotFoundError(f"Missing {proc / name}. Run preprocess first.")
    models_dir = base / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    out = run_training(
        proc / "electricity_hourly.parquet",
        proc / "splits.json",
        proc / "scaler.joblib",
        models_dir,
        seed=cf.ENSEMBLE_SEEDS[0],
    )
    volume.commit()
    return out


@app.function(
    image=image,
    volumes={VOL_MOUNT: volume},
    gpu="T4",
    timeout=1800,
    memory=16384,
)
def evaluate() -> dict:
    from pathlib import Path as P

    from src.eval import run_eval

    base = P(VOL_MOUNT)
    proc = base / "processed"
    models_dir = base / "models"
    has_ckpt = (models_dir / "lstm_load.pt").is_file() or (
        models_dir / "ensemble_manifest.json"
    ).is_file() or bool(list(models_dir.glob("ensemble_member_seed*.pt")))
    for path in (proc / "electricity_hourly.parquet", proc / "splits.json", proc / "scaler.joblib"):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}. Run train first.")
    if not has_ckpt:
        raise FileNotFoundError(f"Missing checkpoints under {models_dir}. Run train first.")
    metrics = run_eval(
        proc / "electricity_hourly.parquet",
        proc / "splits.json",
        proc / "scaler.joblib",
        models_dir,
    )
    volume.commit()
    return metrics


@app.local_entrypoint()
def preprocess_cli(meter: str | None = None) -> None:
    """Run remote preprocess from a proper local entrypoint (`.remote()`)."""
    print(preprocess.remote(meter), flush=True)


@app.local_entrypoint()
def train_cli() -> None:
    print(train.remote(), flush=True)


@app.local_entrypoint()
def evaluate_cli() -> None:
    print(evaluate.remote(), flush=True)


@app.local_entrypoint()
def train_quick_cli() -> None:
    print(train_quick.remote(), flush=True)


@app.local_entrypoint()
def download_artifacts(dest: str = "./artifacts") -> None:
    """Pull /models from the Volume to a local folder (CLI). Uses ``--force`` so re-downloads overwrite."""
    dest_path = Path(dest).resolve()
    _volume_get_models(dest_path)
    print(f"Artifacts in {dest_path}")
    for pt in (
        dest_path / "lstm_load.pt",
        dest_path / "models" / "lstm_load.pt",
    ):
        if pt.is_file():
            print(f"Trained model: {pt}")
            break
    for mf in (dest_path / "ensemble_manifest.json", dest_path / "models" / "ensemble_manifest.json"):
        if mf.is_file():
            print(f"Ensemble manifest: {mf}")
            break


@app.local_entrypoint()
def all_steps() -> None:
    print(preprocess.remote(), flush=True)
    print(train.remote(), flush=True)
    print(evaluate.remote(), flush=True)
    dest_path = (REPO_ROOT / "artifacts").resolve()
    _volume_get_models(dest_path)
    print(f"Artifacts in {dest_path}")
    for pt in (
        dest_path / "lstm_load.pt",
        dest_path / "models" / "lstm_load.pt",
    ):
        if pt.is_file():
            print(f"Trained model: {pt}")
            break
