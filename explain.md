# Electricity Load Forecasting - Simple Project Explanation

## 1) What problem are we solving?

Electricity usage changes every hour.  
If we can predict the next 24 hours, it helps with planning and better decisions.

This project builds a system that:
- learns from past electricity data,
- predicts the next 24 hours of load (in kW),
- compares model output with a simple baseline,
- shows everything in a clear dashboard.

---

## 2) Data in simple words

We use **UCI Electricity Load Diagrams 2011-2014**.

For this project:
- one meter is selected (example: `MT_124`),
- data is converted to **hourly** values,
- cleaned and prepared before training.

So finally, we have a clean hourly sequence like:
`hour 1 load, hour 2 load, hour 3 load, ...`

---

## 3) Important concepts (easy definitions)

## Time Series

A **time series** is data in time order.  
Here: electricity load values hour by hour.

## Window

A **window** means "how much past data we show to the model".

In this project:
- `Window = 336 hours`
- 336 hours = 14 days of history

So model sees last 14 days before predicting.

## Horizon

**Horizon** means "how far in the future we predict".

In this project:
- `Horizon = 24 hours`
- means next 24 hourly points are predicted at once.

## Train / Validation / Test split

Data is split by time (chronological):
- **Train**: first 80% (model learns here)
- **Validation**: next 10% (used to choose best model)
- **Test**: last 10% (final unseen evaluation)

This avoids data leakage from future into past.

---

## 4) LSTM in short (before using it)

**LSTM (Long Short-Term Memory)** is a neural network made for sequences/time-series.

Why LSTM is used:
- it remembers useful past patterns,
- handles longer dependencies better than simple RNN,
- works well when past values influence future values.

In this project, LSTM takes the last 336 hours and predicts next 24 hours.

---

## 5) What features go into the model?

At each hour, model input includes:
- scaled load value,
- hour-of-day cyclic features (sin/cos),
- day-of-week cyclic features (sin/cos),
- DST indicator flags.

These help model understand both:
- load behavior, and
- calendar/time effects.

---

## 6) Baseline model (for fair comparison)

We also use a **seasonal naive baseline**.

Simple idea:
- for a future hour, use similar past hour pattern (weekly lag, fallback daily lag).

Why baseline is important:
- if LSTM does not beat this simple method, model is not very useful.

So dashboard always compares:
- **LSTM forecast**
- **Seasonal naive**
- **Actual values**

---

## 7) Training and serving flow (end-to-end)

1. **Preprocess data**  
   Raw dataset is cleaned, converted to hourly, and saved with splits/scaler.

2. **Train model**  
   LSTM is trained (single best and optional ensemble seeds).

3. **Evaluate**  
   Metrics like MAE, RMSE, sMAPE, skill are saved in `metrics.json`.

4. **Serve API (FastAPI)**  
   API endpoints provide metadata, test windows, and prediction results.

5. **Frontend (Next.js)**  
   UI calls API and shows cards + charts for selected test window.

---

## 8) API endpoints used by UI

- `GET /api/health` -> service status and available model modes  
- `GET /api/meta` -> project metadata + overall test metrics  
- `GET /api/test-windows` -> list/filter/random valid test windows  
- `POST /api/predict` -> forecast and error stats for selected window `t`

---

## 9) UI explanation (what each value shows)

## Header (top)

- **Meter**: which meter/series is currently used (example `MT_124`)
- **Window**: input history size (`336h`)
- **Horizon**: forecast length (`24h`)
- **Test span**: date range of test windows and total window count

## Overall metric cards (full test set)

These are computed across all test windows:

- **Test MAE (LSTM)**  
  Average absolute error of LSTM (kW). Lower is better.  
  Small subtext also shows naive MAE for comparison.

- **Test RMSE (LSTM)**  
  Root mean squared error of LSTM (kW). Penalizes large mistakes more.

- **Test sMAPE (LSTM)**  
  Percentage error metric. Lower is better.

- **Skill vs naive**  
  How much LSTM improves over naive baseline.  
  Positive = better than naive, negative = worse.

- **NMAE / mean load**  
  MAE normalized by average load (percentage). Helps compare scale.

- **Ensemble members**  
  Number of models in ensemble (`n`).  
  Also shows best seed information.

## Left panel controls

- **Model**
  - **Single best**: one best checkpoint model
  - **Ensemble mean**: average prediction of multiple models

- **Pick a test window**
  - **First**: first window in current list
  - **Last**: last window in current list
  - **Random**: random valid test window

- **Filter by date prefix**
  - type `YYYY-MM` or `YYYY-MM-DD`
  - click **Apply** to filter windows

- **Prev / Next**
  - move through paginated test window list

- **Window list item**
  - main line: first prediction timestamp
  - sub line: internal index `t` and window end time

- **Re-run prediction**
  - runs prediction again for currently selected window

## Window metric cards (selected window only)

These update after you pick a window:

- **Window MAE (LSTM)** with naive MAE
- **Window RMSE (LSTM)** with naive RMSE
- **Window sMAPE**
- **Skill vs naive** for that exact window

So this section is local/per-window, not whole test set.

## Forecast vs Actual chart

This line chart shows:
- **History** (last 72h before forecast start)
- **Actual** future values (ground truth)
- **LSTM forecast**
- **Seasonal naive**
- vertical **now** line = start of forecast

Purpose: visually compare how close model is to actual and baseline.

## Per-hour absolute error chart

Bar chart with 24 steps (`+1h` to `+24h`):
- blue bar = LSTM absolute error
- orange bar = naive absolute error

Lower bars are better.

This helps you see if model is good at near-term steps vs later steps.

---

## 10) Quick interpretation tips

- Look at **Skill vs naive** first:
  - positive -> LSTM is useful
  - negative -> naive is better for that case

- Check both:
  - **overall test cards** (global performance),
  - **window cards/charts** (local behavior).

- If one window looks bad, it does not always mean whole model is bad.

---

## 11) Final one-line summary

This project is a full pipeline that takes historical hourly electricity data, trains an LSTM to forecast the next 24 hours, compares it against a seasonal naive baseline, and explains results through a clear API-driven dashboard.
