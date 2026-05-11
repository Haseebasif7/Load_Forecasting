# Testing

Automated checks for modules referenced in the project: `src/windows.py`, `backend/app/inference.py`, and HTTP routes in `backend/app/main.py`.

## Requirements

From the repository root (same venv as training/backend):

```bash
pip install pytest httpx
```

(`httpx` is used by Starlette’s `TestClient`.)

## Run

```bash
cd /path/to/SE_Project
pytest testing -q
```

Verbose:

```bash
pytest testing -v
```

## Layout

| File | What it exercises |
|------|-------------------|
| `fixtures_minimal_state.py` | Builds an in-memory `LoadedState` (synthetic hourly series, real `split_indices` / `build_feature_arrays` / small `LSTMForecaster`) so the API never reads `artifacts/`. |
| `test_split_indices.py` | Chronological window origins `t` for train/val/test (`split_indices`). |
| `test_naive_and_smape.py` | `seasonal_naive_forecast`, `smape` in `backend/app/inference.py`. |
| `test_api_contracts.py` | `GET /api/health`, `/api/meta`, `/api/test-windows`, `POST /api/predict` — aligned with project report **TC-01–TC-08**. |

**TC-09** (Next.js dashboard) remains a manual / browser test; it is not run here.

## Configuration

`pytest.ini` at the repo root sets `pythonpath` so imports resolve (`src`, `backend/app`).
