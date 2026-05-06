import type { MetaResponse } from "@/lib/types";
import { fmtDate } from "@/lib/format";

export function Header({ meta }: { meta: MetaResponse | undefined }) {
  return (
    <header className="border-b border-border">
      <div className="max-w-7xl mx-auto px-6 py-5 flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold text-white">
            Electricity Load Forecast
            <span className="text-muted text-base font-normal ml-3">
              UCI Electricity Load Diagrams 2011–2014
            </span>
          </h1>
          <p className="text-sm text-muted mt-1">
            24-hour LSTM forecast vs seasonal-naive baseline. Pick any window from the held-out test set.
          </p>
        </div>
        {meta && (
          <div className="text-xs text-muted text-right space-y-0.5">
            <div>
              Meter: <span className="text-white font-mono">{meta.meter}</span>
            </div>
            <div>
              Window: <span className="text-white">{meta.window}h</span> · Horizon:{" "}
              <span className="text-white">{meta.horizon}h</span>
            </div>
            <div>
              Test span:{" "}
              <span className="text-white">
                {fmtDate(meta.test_first_pred_time)} → {fmtDate(meta.test_last_pred_time)}
              </span>{" "}
              ({meta.n_test_windows.toLocaleString()} windows)
            </div>
          </div>
        )}
      </div>
    </header>
  );
}
