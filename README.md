# Electricity load forecasting (SE project)

Pipeline spec: [pipeline.md](pipeline.md). Training runs on [Modal](https://modal.com/) with GPU; data and artifacts live on a Modal **Volume** named `electricity-data`.

## Prerequisites

- Python 3.11+ locally for `modal` CLI
- `pip install modal` then `modal setup`
- Raw zip at `data/electricityloaddiagrams20112014.zip` (UCI Electricity Load Diagrams 2011–2014)

## One-time: Volume + upload data

Create the volume once (required before `volume put`):

```bash
modal volume create electricity-data
```

Upload the zip (path inside the volume must match `modal_app.py`):

```bash
cd /Users/haseeb/Desktop/SE_Project
modal volume put electricity-data ./data/electricityloaddiagrams20112014.zip /raw/electricityloaddiagrams20112014.zip
```

## Run on Modal

Use the **`*_cli` local entrypoints** so Modal actually calls `.remote()` and runs GPUs/CPUs (`modal run …::app.evaluate` can sync the app without running evaluate).

```bash
cd /Users/haseeb/Desktop/SE_Project
modal run modal_app.py::preprocess_cli
modal run modal_app.py::train_cli
modal run modal_app.py::evaluate_cli
```

You should see container logs (tqdm epochs for train; eval writes metrics + figures on the Volume). Quick single seed: `modal run modal_app.py::train_quick_cli`.

Optional meter: `modal run modal_app.py::preprocess_cli -- --meter MT_124`.

Preprocess can take several minutes.

Confirm processed files exist:

```bash
modal volume ls electricity-data /processed
```

Training defaults to **3 seeds**, **mean-pooled LSTM (~288 hid)**, **OneCycle LR**, horizon-weighted **L1**, and writes `ensemble_member_seed*.pt` + `ensemble_manifest.json`. **Eval averages** all members on the test set (~3× train time vs one seed). Old checkpoints may be incompatible—re-run **train → evaluate** after pulling updates.

`download_artifacts` and `all_steps` use **`modal volume get --force`** so re-downloading into `./artifacts/` overwrites cleanly (no “already exists” error).

Your **previous Modal download was moved** from `./artifacts/` to **`./artifacts_kept/`** (same layout as before). New runs continue to populate **`./artifacts/`** so nothing is overwritten automatically.

Or end-to-end (preprocess + train + evaluate + download `models` into `./artifacts`):

```bash
cd /Users/haseeb/Desktop/SE_Project
modal run modal_app.py::all_steps
```

## Download outputs only (checkpoint + metrics + figures)

```bash
modal run modal_app.py::download_artifacts
```

Defaults to `./artifacts/` (use e.g. `modal run modal_app.py::download_artifacts -- --dest ./artifacts_v2` to pull into another folder manually).

You should see:

- `lstm_load.pt` — `state_dict` + training config (load with `torch.load`)
- `config.json`, `metrics.json`, `splits.json`
- `scaler.joblib` (copied into `/models` during `evaluate` so it downloads with `volume get /models`)
- `figures/forecast_week.png`, `figures/residuals_by_hour.png`

## Local development (optional)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$(pwd)"
# preprocess_from_zip etc. can be called from Python if you have the zip path
```

## Automated tests (`testing/`)

Pytest suite for `src/windows.split_indices`, `backend/app/inference` (naive baseline, sMAPE), and FastAPI routes **without** real checkpoints on disk (uses a synthetic `LoadedState`). Aligns with project report **TC-01–TC-08**; **TC-09** remains manual (browser + Next.js).

```bash
pip install -r requirements.txt   # includes pytest + httpx
pytest testing -v
```

See [testing/README.md](testing/README.md).

## Web app (FastAPI + Next.js)

A small dashboard to pick a sample from the test split and compare the LSTM forecast against the seasonal-naive baseline lives in `backend/` and `frontend/`. Both run locally and use the artifacts under `artifacts_best/` (single best, seed 42) and the full ensemble under `artifacts/models/`.

One-time: pull the processed hourly parquet from the Modal Volume into `artifacts_best/processed/` so the backend has it locally:

```bash
modal volume get electricity-data /processed/electricity_hourly.parquet \
  ./artifacts_best/processed/electricity_hourly.parquet --force
```

Run backend and frontend in two terminals:

```bash
# terminal 1 - FastAPI on :8000
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
chmod +x run.sh && ./run.sh

# terminal 2 - Next.js on :3000
cd frontend
cp .env.local.example .env.local
npm install
npm run dev
# open http://localhost:3000
```

The UI shows overall test metrics, lets you switch between **Single best** (`lstm_load.pt`) and **Ensemble mean** (averages all `ensemble_member_seed*.pt`), pick any test window (first/last/random or by date prefix), and visualizes 72 h of history followed by 24 h of forecast vs actual vs naive, plus per-hour error bars. See [backend/README.md](backend/README.md) and [frontend/README.md](frontend/README.md).

## Modal docs in this repo

See [modal/SKILL.md](modal/SKILL.md) and [modal/references/](modal/references/).

## Citation

Trindade, A. (2015). *ElectricityLoadDiagrams20112014* [Dataset]. UCI Machine Learning Repository. <https://doi.org/10.24432/C58C86>
