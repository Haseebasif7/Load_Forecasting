import type { MetaResponse } from "@/lib/types";
import { fmtKw, fmtNum, fmtPct } from "@/lib/format";

function num(meta: MetaResponse, key: string): number | undefined {
  const v = meta.metrics?.[key];
  return typeof v === "number" ? v : undefined;
}

export function OverallMetrics({ meta }: { meta: MetaResponse }) {
  const lstmMae = num(meta, "test_mae_kw_lstm");
  const lstmRmse = num(meta, "test_rmse_kw_lstm");
  const lstmSmape = num(meta, "test_smape_pct_lstm");
  const naiveMae = num(meta, "test_mae_kw_naive");
  const skill = num(meta, "skill_score_vs_naive");
  const nmae = num(meta, "test_nmae_pct_of_mean_load_lstm");
  const nMembers = num(meta, "ensemble_n_members");

  const cards: Array<{ label: string; value: string; sub?: string }> = [
    { label: "Test MAE (LSTM)", value: fmtKw(lstmMae), sub: `naive: ${fmtKw(naiveMae)}` },
    { label: "Test RMSE (LSTM)", value: fmtKw(lstmRmse) },
    { label: "Test sMAPE (LSTM)", value: fmtPct(lstmSmape) },
    {
      label: "Skill vs naive",
      value: skill != null ? fmtNum(skill, 3) : "—",
      sub: skill != null ? `${(skill * 100).toFixed(1)}% better` : undefined,
    },
    { label: "NMAE / mean load", value: fmtPct(nmae) },
    {
      label: "Ensemble members",
      value: nMembers != null ? String(Math.round(nMembers)) : "—",
      sub: meta.manifest?.best_seed != null ? `best seed: ${meta.manifest.best_seed}` : undefined,
    },
  ];

  return (
    <section className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
      {cards.map((c) => (
        <div key={c.label} className="card">
          <div className="card-title">{c.label}</div>
          <div className="metric-value mt-1">{c.value}</div>
          {c.sub && <div className="text-xs text-muted mt-1">{c.sub}</div>}
        </div>
      ))}
    </section>
  );
}
