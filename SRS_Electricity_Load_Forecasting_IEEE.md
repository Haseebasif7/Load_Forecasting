<!--
  Software Requirements Specification — IEEE-style structured report
  (section layout aligned with sample: AI_Disaster_Relief_SRS.docx).

  For submission: open in Word/Google Docs, apply your course IEEE template
  (two-column optional), set heading styles for 1., 1.1, FR-1 tables, etc.

  Prepared for: Spring 2026 SE project (10 marks).
-->

**National University of Computer & Emerging Sciences, Karachi**

# Software Requirements Specification (SRS)

## Deep Learning Electricity Load Forecasting System

**Course:** Software Engineering  
**Semester:** Spring 2026  
**Technical Focus Area:** AI / Deep Learning — Time-Series Load Forecasting  
**Dataset Basis:** UCI *Electricity Load Diagrams 2011–2014*

**Prepared By**  
Abdul Wasay — 23L-0658  
Haseeb Asif — 23K-0539  

---

### Abstract

This Software Requirements Specification defines the requirements for a deep-learning–based hourly electricity load forecasting system. The system assists analysts and reviewers by predicting the next twenty-four consecutive hours of mean active power (kW) for a single Portuguese client series from historical consumption, using an LSTM sequence-to-horizon model trained offline and optionally averaged over multiple random seeds for stability. Users interact primarily through a web dashboard that retrieves meta-information, selects valid test-split windows from the held-out chronological period, and visualizes forecasts against ground truth and a seasonal naive baseline.

The document specifies scope, user classes, functional and non-functional requirements, external interfaces, data assumptions, architectural constraints consistent with layered design, verification and validation against test metrics including MAE, RMSE, sMAPE, and skill versus baseline, plus future enhancements. The implementation aligns with project pipeline documentation and reproducible Modal-based training artifacts.

### Index Terms

Load forecasting; LSTM; time series forecasting; electricity demand; seasonal naive baseline; REST API; software requirements specification.

---

### Table of Contents

1. Introduction  
2. Overall Description  
3. System Features and Functional Requirements  
4. External Interface Requirements  
5. Non-Functional Requirements  
6. System Data Requirements  
7. Analysis Models and Design Constraints  
8. Verification and Validation Requirements  
9. Future Enhancements  
10. Appendix  

---

## 1. Introduction

### 1.1 Purpose

The purpose of this document is to specify the software requirements for the **Deep Learning Electricity Load Forecasting System**. The SRS is intended for the project team, course instructor or evaluator, reviewers, and future maintainers. It defines what the delivered software must accomplish, boundaries of scope, measurable quality expectations, and traceability between requirements and planned modules—for an acceptable **10-mark** academic demonstration.

### 1.2 Scope

The proposed system is a **prototype forecasting application**—not a grid control or billing system. Scope includes:

- **Offline ML pipeline**: ingest UCI electricity data, preprocess to hourly resolution, chronological train/validation/test splits, robust scaling fitted on train only, PyTorch-based LSTM training (with optional multi-seed checkpoints), seasonal naive baseline, test evaluation producing metrics and static figures—executed remotely on Modal with persisted Volume storage (`electricity-data`) for reproducibility.

- **Runtime demo stack**: FastAPI REST service exposes health, dataset/model metadata, paginated enumeration of legal **test-split** forecasting windows, and per-window inference using either the **single best** checkpoint (`lstm_load.pt`, validation-best seed, typically 42) or an **ensemble mean** over separately trained checkpoints when available alongside the single model.

- **Web UI**: Next.js dashboard summarizing aggregated test-set metrics read from persisted `metrics.json`, allowing model mode selection, filtering or paging test windows by time, invoking prediction, and visualizing contextual history plus 24-step forecast curves and per-horizon error bars versus the naive comparator.

Excluded from baseline scope (unless explicitly adopted later):

- Streaming ingestion from utility SCADA/MQTT; regulatory compliance or production SLAs; multi-tenant auth; per-customer commercial deployment; training inside the web request path; automatic retraining triggers.

### 1.3 Definitions, Acronyms, and Abbreviations

| Term | Meaning |
|------|---------|
| AI | Artificial Intelligence |
| API | Application Programming Interface |
| CPU / GPU | Central / Graphics Processing Unit |
| DL | Deep Learning |
| DST | Daylight Saving Time |
| FR / NFR | Functional / Non-Functional Requirement |
| HTTP / REST | Hypertext Transfer Protocol; Representational State Transfer |
| JSON | JavaScript Object Notation |
| kW | Kilowatt (active power) |
| LSTM | Long Short-Term Memory (recurrent neural network) |
| MAE | Mean Absolute Error |
| MLP | Multi-Layer Perceptron |
| RMSE | Root Mean Square Error |
| SRS | Software Requirements Specification |
| sMAPE | Symmetric Mean Absolute Percentage Error |
| UI | User Interface |
| UCI | University of California, Irvine (ML Repository) |

**Skill score (vs seasonal naive):** \(1 - \mathrm{MAE}_{\mathrm{LSTM}} / \mathrm{MAE}_{\mathrm{naive}}\) on the same test windows (higher is better when LSTM beats naive).

### 1.4 References

1. IEEE Std 830-1998 (historical) / ISO/IEC/IEEE 29148–related practice for Software Requirements Specifications (structure inspiration).  
2. Course Software Engineering project guidelines, Spring 2026.  
3. Trindade, A. (2015). *ElectricityLoadDiagrams20112014* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C58C86  
4. Internal project documents: `pipeline.md`, `README.md`, `PROBLEM_AND_APPROACH.md`, `backend/README.md`, `frontend/README.md`.

### 1.5 Document Overview

Sections 2–6 describe the product context, features, interfaces, NFRs, and data. Sections 7–8 cover architectural / design constraints and V&V. Section 9 lists enhancements; Section 10 contains supplementary tables.

---

## 2. Overall Description

### 2.1 Product Perspective

The system comprises four logical subsystems:

1. **Data & preprocessing** (`src/data.py`): zip extraction, hourly resampling, leading-zero trimming, DST binary flags aligned to Portuguese wall-clock rules, parquet persistence.  
2. **Modeling core** (`src/windows.py`, `src/models.py`, `src/train.py`, `src/eval.py`): windowed tensors, LSTM encoder + pooled representation + dense head producing 24 outputs, baseline naive forecast for comparison, ensemble evaluation averaging members in scaled space before inverse scaling.  
3. **Serving layer** (`backend/`): artifact resolution, deterministic index validation, synchronous CPU inference batches of size one for UI latency.  
4. **Presentation layer** (`frontend/`): SPA-style dashboard invoking documented REST endpoints.

Temporal integrity is preserved: scaler transform parameters derive **only from training-period load** statistics; chronological splits forbid future leakage during supervised window construction.

### 2.2 Product Functions

- Preprocess historical load into hourly series with reproducible DST features.  
- Train / evaluate LSTM and capture checkpoints, scaler, splits manifest, aggregated metrics JSON, diagnostic plots.  
- Offer interactive exploration of canonical **held-out test** forecasting origins `t`.  
- Generate per-window forecasts (single or averaged ensemble members) versus actuals and naive reference.  
- Visualize condensed history preceding each forecast horizon plus error diagnostics.  

### 2.3 User Classes and Characteristics

| User class | Role | Technical skill |
|------------|------|-----------------|
| Project evaluator / Instructor | Judges correctness, reproducibility, metrics, usability during demo | General technical familiarity |
| Data / ML reviewer | Validates methodology, leakage controls, fairness of baseline comparison | Intermediate ML literacy |
| Domain curious analyst | Interprets kW timelines and residuals | Basic statistics |
| Future maintainer | Refactors modules, swaps models | Intermediate Python / TypeScript |

No separate production operator role exists in baseline scope.

### 2.4 Operating Environment

- **Development / demo hosts:** macOS / Linux / Windows with Python ≥3.10/3.11, Node.js compatible with Next.js 14, modern Chromium-class browser.  
- **Training / batch evaluation:** Modal cloud sandbox with CUDA GPU capability and named Volume for `/raw`, `/processed`, `/models`.  
- **Inference default:** FastAPI worker on localhost (or equivalent LAN host) reachable from frontend via configurable `NEXT_PUBLIC_API_BASE`.

### 2.5 Design and Implementation Constraints

- Academic prototype constrained to **one-meter** univariate path for clarity; generalized multi-meter reuse is deferred.  
- Training compute budget limited by course timeline; exhaustive hyper-parameter search optional.  
- Forecast outputs are research / educational—they **must not** be represented as authoritative grid dispatch directives.  
- Modal Volume availability assumed for repeatable cloud runs; offline demo relies on fetched parquet + checkpoints.  
- Direct multi-step formulation fixed at **24** outputs to avoid iterative error compounding mandated by naive autoregressive rollouts unless scope expands.

### 2.6 Assumptions and Dependencies

- Historical series integrity: duplicate timestamps negligible or dropped; DST localization choices documented consistently.  
- Users supply / system ships consistent triple of **checkpoint(s)**, **`scaler.joblib`**, **`splits.json`** plus compatible processed parquet aligned on row counts implied by persisted split metadata—or explicit custom path (`ELECTRICITY_PROCESSED_PARQUET`) points to validated file.  
- Ensemble files may be absent partially; degraded mode falls back cleanly to fewer members or solely single best checkpoint.  
- Frontend depends on CORS allowances for whichever host serves API.

---

## 3. System Features and Functional Requirements

### 3.1 Feature F1 — Data Preprocessing Pipeline

| ID | Requirement |
|----|-------------|
| FR-1 | The system shall ingest `LD2011_2014.txt` (semicolon SEP, comma decimal) from the published UCI ZIP source. |
| FR-2 | The system shall resample sub-hourly readings to **mean hourly** kW and persist a parquet representation including derived DST indicator columns (`is_march_dst_short`, `is_october_dst_long`, `is_dst_window`). |
| FR-3 | The system shall trim leading stationary near-zero inactive prefix before modeling per preprocessing rules documented in pipeline materials. |
| FR-4 | The system shall fit **RobustScaler** exclusively on training-segment loads and reuse those parameters during validation/test window construction & inference. |
| FR-5 | The system shall materialize chronological split boundaries (train earliest 80 %, contiguous validation 10 %, contiguous test remainder 10 %) surfaced through `splits.json`. |

### 3.2 Feature F2 — Model Training & Baseline Comparison

| ID | Requirement |
|----|-------------|
| FR-6 | The system shall expose an LSTM architecture accepting sliding windows shaped **(336 timesteps × 8 features)** where features concatenated per timestep comprise scaled lagged load plus sinusoidal calendar encodings plus DST binaries. |
| FR-7 | The system shall implement a seasonal naive predictor using primary lag **168 h** per horizon step falling back where necessary to lag **24 h** as parity with evaluation module. |
| FR-8 | The system shall support optional training across multiple RNG seeds exporting distinct member checkpoints and a manifest documenting validation MAE in kW and best_seed selection. |

### 3.3 Feature F3 — Aggregation & Persisted Metrics

| ID | Requirement |
|----|-------------|
| FR-9 | The evaluation routine shall optionally **average ensemble member scaled-space outputs** prior to inverse scaling when multiple compatible checkpoints exist—mirroring full-dataset aggregation logic. |
| FR-10 | The system shall write `metrics.json` containing at minimum test MAE/RMSE for LSTM vs naive **in kW**, sMAPE, skill score versus naive, count of enumerated test windows, and ensemble membership metadata. |

### 3.4 Feature F4 — Inference API

| ID | Requirement |
|----|-------------|
| FR-11 | The REST service shall expose `GET /api/health` returning availability boolean, served model modality labels, total valid **test-origin** indices. |
| FR-12 | `GET /api/meta` shall return fused metadata: meter id, chronological coverage, window (`W`) & horizon (`H`), first/last test prediction timestamps, raw `metrics`, optional ensemble manifest parity with disk JSON, plus serialized training-config snapshot fragments. |
| FR-13 | `GET /api/test-windows` shall paginate enumerated legal indices with ISO timestamps bounding each window optionally filterable (`q=` date prefix substring) plus `random=` convenience flag. |
| FR-14 | `POST /api/predict` shall accept `{t, model∈{"single","ensemble"}}`, reject illegal `t` not in enumerated test set membership, assemble context history length **W**, produce **H** sequential predicted kW vectors, naive baseline vector, residual magnitudes bundle, aggregated window MAE/RMSE/sMAPE & skill against naive. |
| FR-15 | Service startup shall fail fast with actionable guidance if required artifact files are missing, enumerating search order for processed parquet path resolution. |

### 3.5 Feature F5 — Web Dashboard

| ID | Requirement |
|----|-------------|
| FR-16 | Dashboard shall render global cards summarizing stored aggregate test metrics (MAE, RMSE, sMAPE, skill, NMAE%, ensemble size). |
| FR-17 | User shall toggle inference between **Single best** and **Ensemble mean** when backend advertises availability. |
| FR-18 | User shall select or filter at least one valid test window and trigger prediction display without manual tensor construction. |
| FR-19 | Visualization shall plot trailing **72 h** historical actual load contiguously joined with **24 h** overlay of model forecast, ground truth, and naive curve including a vertical transition marker at first predicted hour. |
| FR-20 | Secondary chart shall compare absolute per-horizon-step errors for LSTM vs naive (24 grouped bars). |
| FR-21 | Localized per-window summary statistics (MAE/RMSE/sMAPE/skill) shall update after each successful prediction request. |

---

## 4. External Interface Requirements

### 4.1 User Interface (Web)

- Dark-themed responsive layout with primary navigation implicit (single page).  
- Left control stack: model mode, filter / pagination / random pick affordances, explicit re-run action.  
- Right analytics stack: metric cards, combined line chart, error bar chart.  
- Empty and error states shall explain missing backend connectivity or absent data.

### 4.2 Software Interfaces

- FastAPI ASGI app on port **8000** default with OpenAPI (`/docs`) optional but recommended for grading reviews.  
- Next.js dev server on **3000** default via `npm run dev`.  
- Environment variable `NEXT_PUBLIC_API_BASE` directs client-side `fetch` calls.  
- Optional `ELECTRICITY_PROCESSED_PARQUET` environment variable on server for advanced path override.  
- Optional `EXTRA_CORS_ORIGINS` comma list extends allowed browser origins.

### 4.3 Hardware Interfaces

None beyond standard keyboard / pointer on general-purpose computers.

### 4.4 Communication Interfaces

HTTP/JSON only in prototype; TLS termination optional for local demo; no mandatory third-party OAuth or paid API keys in baseline.

---

## 5. Non-Functional Requirements

| ID | Category | Requirement |
|----|----------|-------------|
| NFR-1 | Usability | A first-time reviewer shall complete a valid forecast visualization path (select window → view charts) in **≤ 5 minutes** following README instructions. |
| NFR-2 | Performance | Single-window prediction round-trip (local CPU) shall typically complete **< 2 s** excluding network latency; cold model load occurs at process start not per request. |
| NFR-3 | Reliability | Invalid `t` or absent artifacts shall yield structured HTTP 4xx/5xx with descriptive JSON/text body—no silent partial tensors. |
| NFR-4 | Maintainability | Separation of concerns: `src` training core vs `backend` serving vs `frontend` UI; configuration constants centralized (`src/config.py`). |
| NFR-5 | Portability | Stack runs on macOS/Linux/Windows with documented dependency installs (`requirements.txt`, `npm`). |
| NFR-6 | Reproducibility | Training seeds enumerated; manifest records multiple members; evaluation label distinguishes ensemble vs single in metrics file. |
| NFR-7 | Observability | Health endpoint supports minimal operational ping; logging of startup failures includes missing file enumeration. |
| NFR-8 | Explainability | UI differentiates model vs naive vs actual; summary skill metric contextualizes uplift over naive. |

---

## 6. System Data Requirements

### 6.1 Dataset Description

Source: UCI Electricity Load Diagrams 2011–2014. One selected `MT_*` series (e.g., `MT_124` in current configuration) after automated stable-activity heuristics. Hourly table columns minimally: timestamp index, `load_kw`, three DST binary flags.

### 6.2 Model Input Construction (Per Supervised Sample)

For window origin `t`:

- Input tensor rows `t−W … t−1` each: `[scaled_load, sin_h, cos_h, sin_dow, cos_dow, dst_flags×3]`.  
- Target during training: vector `load_scaled[t … t+H−1]`.  
- At inference UI demonstration: ground truth read from raw `load_kw[t …]`.

### 6.3 Target / Output Semantics

Direct vector of **H = 24** future hourly mean kW predictions (post inverse scaler). No intermediate classification head.

### 6.4 Data Quality & Integrity Rules

- Strict monotonic time index after cleaning.  
- Train-only scaler fit prevents distribution peeking.  
- Test windows only originate from **post–test_start** positions ensuring model never trained on those target hours.  
- Parquet path resolution order documented to avoid silent wrong-file binding.

---

## 7. Analysis Models and Design Constraints

### 7.1 Architectural Style

Layered: **data → feature/window builder → model → evaluation → API adapter → UI**. Training remote; inference local CPU acceptable.

### 7.2 Design Patterns (Conceptual Application)

| Pattern | Use |
|---------|-----|
| Strategy | Swapping single vs ensemble forward path without duplicating window assembly |
| Facade | FastAPI routes abstracting `LoadedState` orchestration |
| Dependency Injection (light) | Path resolution / environment overrides |
| Immutable config snapshots | Embedded in checkpoint JSON payload for faithful model rebuild |

### 7.3 Core Use Cases (Summary)

1. Engineer runs preprocess → train → evaluate on Modal Volume.  
2. Reviewer downloads artifacts & processed parquet.  
3. Reviewer launches API + UI, inspects meta.  
4. Reviewer explores random / filtered test window.  
5. System returns forecast visualization + error analytics.  

### 7.4 Risks

| Risk | Mitigation |
|------|------------|
| Non-stationary regime shift reduces LSTM edge | Document skill vs naive; show residual-by-hour plot in batch eval |
| Single-series scope limits generalization claim | Scope statement + future work |
| User misplaces mismatched scaler vs checkpoint | Bundled `artifacts_best` folder guidance |
| Cloud dependency for retrain | Local CPU re-run possible if user reprovisions environment |

---

## 8. Verification and Validation Requirements

### 8.1 Functional Testing

- Valid `t` returns 200 with consistent length arrays (`W` history, `H` forecast).  
- Illegal `t` returns 400.  
- Meta endpoint fields parseable and consistent with on-disk JSON.  
- Toggle single vs ensemble changes `n_members` field & (when members differ) potentially forecast values.  
- Random test-window endpoint returns item present in full index universe.

### 8.2 Model Validation

- Report test **MAE/RMSE/sMAPE** in **kW / %** from `metrics.json`.  
- Skill score vs seasonal naive must be computable reproducibly from exported predictions or logged arrays.  
- Baseline mandated for interpretability—not optional.

### 8.3 Usability / Acceptance

Minimal peer walkthrough verifies interpretability absent developer narration.

### 8.4 Acceptance Criteria (Course Demo Ready)

✅ Held-out chronological test windows browsable via UI  
✅ Visual forecast + naive comparison renders without runtime training  
✅ Documented SRS trace from FR → implemented modules (`src`, `backend`, `frontend`)  
✅ Artifacts reproducibly fetchable (`modal volume get` commands recorded in README linkage)

---

## 9. Future Enhancements

- Parallel multi-meter selection & comparative dashboards.  
- Quantile regression or MC dropout for predictive intervals.  
- TCN / Transformer parity experiments integrated into selectable API mode.  
- Automated scheduled retraining + drift watchdog.  
- OAuth-protected collaborative deployment behind HTTPS reverse proxy & container images.  

---

## 10. Appendix

### 10.A Primary Implementation ↔ Requirement Trace (Abbreviated)

| Area | reqs |
|------|------|
| `src/data.py` | FR-1–FR-5 |
| `src/windows.py`, `src/models.py`, `src/train.py`, `src/eval.py` | FR-6–FR-10 |
| `backend/app/*` | FR-11–FR-15 |
| `frontend/*` | FR-16–FR-21 |

### 10.B Key Hyperparameters Snapshot (Configurable)

Window **336**, horizon **24**, hidden size **288** (current strong config), dropout **0.25**, ensemble seeds tuple `(42,123,777)` unless edited in `src/config.py`.

### 10.C Example REST Payload Fragment (Abbreviated Shape)

```
POST /api/predict
{ "t": 31894, "model": "ensemble" }
→ { t, first_pred_time, history{times[], load_kw[]},
    forecast{times[], pred_kw[], actual_kw[], naive_kw[]}, errors{}, summary{}, model_used, n_members }
```

---

**Document status:** Draft aligned with implemented Spring 2026 prototype codebase. Revision history should annotate metric refresh dates after major retraining.

---

### *End of SRS*
