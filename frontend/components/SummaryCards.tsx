import type { PredictResponse } from "@/lib/types";
import { fmtKw, fmtNum, fmtPct } from "@/lib/format";

export function SummaryCards({ res }: { res: PredictResponse }) {
  const s = res.summary;
  const skillPct = (s.skill_score * 100).toFixed(1);
  const cards = [
    { label: "Window MAE (LSTM)", value: fmtKw(s.mae_kw), sub: `naive: ${fmtKw(s.naive_mae_kw)}` },
    { label: "Window RMSE (LSTM)", value: fmtKw(s.rmse_kw), sub: `naive: ${fmtKw(s.naive_rmse_kw)}` },
    { label: "Window sMAPE", value: fmtPct(s.smape_pct) },
    {
      label: "Skill vs naive",
      value: fmtNum(s.skill_score, 3),
      sub: `${skillPct}% better than naive`,
      good: s.skill_score > 0,
      bad: s.skill_score < 0,
    },
  ];
  return (
    <section className="grid grid-cols-2 lg:grid-cols-4 gap-3">
      {cards.map((c) => (
        <div key={c.label} className="card">
          <div className="card-title">{c.label}</div>
          <div className="metric-value mt-1">{c.value}</div>
          {c.sub && (
            <div
              className={`text-xs mt-1 ${
                c.good ? "text-good" : c.bad ? "text-bad" : "text-muted"
              }`}
            >
              {c.sub}
            </div>
          )}
        </div>
      ))}
    </section>
  );
}
