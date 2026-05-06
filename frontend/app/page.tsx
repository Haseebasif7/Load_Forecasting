"use client";

import { useEffect, useState } from "react";
import useSWR from "swr";
import { fetcher, predict } from "@/lib/api";
import type {
  MetaResponse,
  ModelChoice,
  PredictResponse,
  TestWindowItem,
} from "@/lib/types";
import { Header } from "@/components/Header";
import { OverallMetrics } from "@/components/OverallMetrics";
import { ModelToggle } from "@/components/ModelToggle";
import { SamplePicker } from "@/components/SamplePicker";
import { SummaryCards } from "@/components/SummaryCards";
import { ForecastChart } from "@/components/ForecastChart";
import { ErrorBars } from "@/components/ErrorBars";

export default function Page() {
  const { data: meta, error: metaErr } = useSWR<MetaResponse>("/api/meta", fetcher);

  const [model, setModel] = useState<ModelChoice>("ensemble");
  const [picked, setPicked] = useState<TestWindowItem | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [predicting, setPredicting] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    if (meta && !meta.available_models.includes("ensemble") && model === "ensemble") {
      setModel("single");
    }
  }, [meta, model]);

  async function runPredict(target: TestWindowItem | null = picked, choice = model) {
    if (!target) return;
    setPredicting(true);
    setErr(null);
    try {
      const res = await predict(target.t, choice);
      setResult(res);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setPredicting(false);
    }
  }

  function onPick(item: TestWindowItem) {
    setPicked(item);
    runPredict(item, model);
  }

  function onModelChange(next: ModelChoice) {
    setModel(next);
    if (picked) runPredict(picked, next);
  }

  return (
    <main className="min-h-screen">
      <Header meta={meta} />

      <div className="max-w-7xl mx-auto px-6 py-6 space-y-6">
        {metaErr && (
          <div className="card border-bad/60">
            <div className="text-bad font-medium">Could not reach the API</div>
            <div className="text-sm text-muted mt-1">
              Make sure the backend is running on{" "}
              <span className="font-mono">{process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000"}</span>.
              Detail: {String((metaErr as Error).message ?? metaErr)}
            </div>
          </div>
        )}

        {meta && <OverallMetrics meta={meta} />}

        <div className="grid grid-cols-1 lg:grid-cols-[320px_minmax(0,1fr)] gap-6">
          <aside className="card space-y-5 self-start">
            <ModelToggle
              value={model}
              onChange={onModelChange}
              available={meta?.available_models ?? ["single"]}
              bestSeed={meta?.manifest?.best_seed}
              ensembleN={
                typeof meta?.metrics?.ensemble_n_members === "number"
                  ? (meta.metrics.ensemble_n_members as number)
                  : meta?.manifest?.ensemble_seeds?.length
              }
            />
            <SamplePicker onPick={onPick} current={picked} />
            <button
              className="btn btn-primary w-full"
              onClick={() => runPredict()}
              disabled={!picked || predicting}
            >
              {predicting ? "Predicting…" : picked ? "Re-run prediction" : "Pick a window first"}
            </button>
            {err && <div className="text-sm text-bad">{err}</div>}
          </aside>

          <section className="space-y-5 min-w-0">
            {!result && (
              <div className="card text-sm text-muted">
                Pick a test window from the left to see the LSTM forecast, the seasonal-naive
                baseline, and per-hour error charts.
              </div>
            )}
            {result && (
              <>
                <SummaryCards res={result} />
                <ForecastChart res={result} contextHours={72} />
                <ErrorBars res={result} />
              </>
            )}
          </section>
        </div>

        <footer className="text-xs text-muted text-center pt-6 pb-4">
          Trained on UCI Electricity Load Diagrams 2011–2014 · LSTM (W=336, H=24) ·
          {" "}see{" "}
          <a className="underline hover:text-white" href="https://doi.org/10.24432/C58C86" target="_blank" rel="noreferrer">
            Trindade (2015)
          </a>
        </footer>
      </div>
    </main>
  );
}
