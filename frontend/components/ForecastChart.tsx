"use client";

import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { PredictResponse } from "@/lib/types";
import { fmtDate, fmtKw } from "@/lib/format";

interface Props {
  res: PredictResponse;
  contextHours?: number;
}

interface Row {
  time: string;
  label: string;
  history?: number;
  pred?: number;
  actual?: number;
  naive?: number;
}

function buildRows(res: PredictResponse, contextHours: number): { rows: Row[]; cutoffLabel: string } {
  const ctx = Math.min(contextHours, res.history.times.length);
  const histStart = res.history.times.length - ctx;

  const rows: Row[] = [];
  for (let i = histStart; i < res.history.times.length; i++) {
    const time = res.history.times[i];
    rows.push({
      time,
      label: time.slice(5, 16).replace("T", " "),
      history: res.history.load_kw[i],
    });
  }
  const cutoffLabel = res.first_pred_time.slice(5, 16).replace("T", " ");
  for (let i = 0; i < res.forecast.times.length; i++) {
    const time = res.forecast.times[i];
    rows.push({
      time,
      label: time.slice(5, 16).replace("T", " "),
      pred: res.forecast.pred_kw[i],
      actual: res.forecast.actual_kw[i],
      naive: res.forecast.naive_kw[i],
    });
  }
  return { rows, cutoffLabel };
}

export function ForecastChart({ res, contextHours = 72 }: Props) {
  const { rows, cutoffLabel } = buildRows(res, contextHours);

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-3">
        <div>
          <div className="card-title">Forecast vs actual</div>
          <div className="text-xs text-muted mt-1">
            History: last {contextHours}h · Forecast horizon: {res.forecast.times.length}h ·
            First predicted: {fmtDate(res.first_pred_time)}
          </div>
        </div>
        <div className="text-xs text-muted">
          model: <span className="text-white">{res.model_used}</span>
          {res.model_used === "ensemble" && (
            <span className="ml-1 opacity-70">(n={res.n_members})</span>
          )}
        </div>
      </div>
      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={rows} margin={{ top: 10, right: 16, left: 0, bottom: 0 }}>
            <CartesianGrid stroke="#1f2a44" strokeDasharray="3 3" />
            <XAxis
              dataKey="label"
              stroke="#94a3b8"
              tick={{ fontSize: 11 }}
              minTickGap={28}
            />
            <YAxis
              stroke="#94a3b8"
              tick={{ fontSize: 11 }}
              label={{
                value: "kW",
                angle: -90,
                position: "insideLeft",
                fill: "#94a3b8",
                fontSize: 11,
              }}
            />
            <Tooltip
              contentStyle={{
                background: "#101a2e",
                border: "1px solid #1f2a44",
                borderRadius: 8,
                fontSize: 12,
              }}
              formatter={(value: number, name) => [fmtKw(value), name]}
            />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            <ReferenceLine
              x={cutoffLabel}
              stroke="#a78bfa"
              strokeDasharray="4 3"
              label={{ value: "now", fill: "#a78bfa", fontSize: 11, position: "top" }}
            />
            <Line
              type="monotone"
              dataKey="history"
              name="History"
              stroke="#94a3b8"
              dot={false}
              strokeWidth={1.5}
              isAnimationActive={false}
              connectNulls={false}
            />
            <Line
              type="monotone"
              dataKey="actual"
              name="Actual"
              stroke="#34d399"
              dot={false}
              strokeWidth={2}
              isAnimationActive={false}
              connectNulls={false}
            />
            <Line
              type="monotone"
              dataKey="pred"
              name="LSTM forecast"
              stroke="#60a5fa"
              dot={false}
              strokeWidth={2.5}
              isAnimationActive={false}
              connectNulls={false}
            />
            <Line
              type="monotone"
              dataKey="naive"
              name="Seasonal naive"
              stroke="#f59e0b"
              dot={false}
              strokeWidth={1.5}
              strokeDasharray="5 4"
              isAnimationActive={false}
              connectNulls={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
