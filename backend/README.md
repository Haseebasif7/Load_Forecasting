# Backend (FastAPI)

Serves the trained LSTM load forecaster:

- **Single best**: `artifacts_best/models/lstm_load.pt` (seed 42).
- **Ensemble mean**: averages predictions of all `artifacts/models/ensemble_member_seed*.pt` checkpoints (falls back to single if those are missing).

## One-time prerequisite

Download the processed hourly parquet from the Modal Volume into `artifacts_best/processed/`:

```bash
cd ..
modal volume get electricity-data /processed/electricity_hourly.parquet \
  ./artifacts_best/processed/electricity_hourly.parquet --force
```

The parquet is the source of all test windows the API serves.

The server looks for it in order: **`ELECTRICITY_PROCESSED_PARQUET`** (absolute path env var if set), then `artifacts_best/processed/`, `artifacts/processed/`, then `data/processed/`.

## Run

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
chmod +x run.sh
./run.sh
```

API at http://localhost:8000. Interactive docs at http://localhost:8000/docs.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Liveness + which models loaded + test-window count |
| GET | `/api/meta` | Meter, time range, hyperparameters, overall test metrics, ensemble manifest |
| GET | `/api/test-windows?limit=&offset=&q=&random=` | Paginated test windows (filter by ISO date prefix; or pick one at random) |
| POST | `/api/predict` | Run inference for one test window: `{ "t": int, "model": "single" \| "ensemble" }` |

## Notes

- CORS allows `http://localhost:3000` (override with `EXTRA_CORS_ORIGINS=https://foo,https://bar`).
- Inference runs on CPU (torch); each window is a single batch — fast enough for a UI.
- Reuses `src/models.py`, `src/windows.py`, `src/data.py`, `src/eval.py` from the repo root via `PYTHONPATH`.
