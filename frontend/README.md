# Frontend (Next.js + Recharts)

Dashboard for the FastAPI load-forecasting backend.

## Run

```bash
cd frontend
cp .env.local.example .env.local   # NEXT_PUBLIC_API_BASE=http://localhost:8000
npm install
npm run dev
# open http://localhost:3000
```

The page expects the backend (see [../backend/README.md](../backend/README.md)) to be running.

## What it shows

- **Header** — meter id, hourly window/horizon, full test span, and total windows from `/api/meta`.
- **Overall metrics** — full test-set MAE / RMSE / sMAPE / skill / NMAE / ensemble size from `metrics.json`.
- **Model toggle** — switch between **Single best** (`lstm_load.pt`) and **Ensemble mean** (averages all `ensemble_member_seed*.pt`).
- **Sample picker** — paginated list of test windows with First/Last/Random shortcuts and a date-prefix filter (e.g. `2014-08`).
- **Forecast chart** — last 72 hours of history, then the next 24 hours of actual / LSTM forecast / seasonal-naive baseline, with a "now" reference line.
- **Per-hour error bars** — absolute LSTM vs naive error at each of the 24 horizon steps.
- **Summary cards** — MAE / RMSE / sMAPE / skill score for the picked window.

## Tech

- Next.js 14 (App Router) + TypeScript
- Tailwind CSS for layout
- Recharts for line and bar charts
- SWR for cached `/api/meta` fetch
