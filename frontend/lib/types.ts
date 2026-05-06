export type ModelChoice = "single" | "ensemble";

export interface HealthResponse {
  ok: boolean;
  models: string[];
  n_test_windows: number;
}

export interface TestWindowItem {
  id: number;
  t: number;
  start_time: string;
  first_pred_time: string;
  last_pred_time: string;
}

export interface TestWindowsResponse {
  total: number;
  offset: number;
  limit: number;
  items: TestWindowItem[];
}

export interface MetaResponse {
  meter: string;
  time_start: string;
  time_end: string;
  window: number;
  horizon: number;
  n_rows_hourly: number;
  n_test_windows: number;
  test_first_pred_time: string;
  test_last_pred_time: string;
  available_models: string[];
  metrics: Record<string, number | string>;
  manifest: null | {
    ensemble_seeds?: number[];
    member_files?: string[];
    member_val_mae_kw?: number[];
    best_seed?: number;
  };
  config: Record<string, unknown>;
}

export interface PredictResponse {
  t: number;
  first_pred_time: string;
  history: { times: string[]; load_kw: number[] };
  forecast: {
    times: string[];
    pred_kw: number[];
    actual_kw: number[];
    naive_kw: number[];
  };
  errors: { lstm_abs: number[]; naive_abs: number[] };
  summary: {
    mae_kw: number;
    rmse_kw: number;
    smape_pct: number;
    naive_mae_kw: number;
    naive_rmse_kw: number;
    skill_score: number;
  };
  model_used: ModelChoice;
  n_members: number;
}
