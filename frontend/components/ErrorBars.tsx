"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { PredictResponse } from "@/lib/types";
import { fmtKw } from "@/lib/format";

export function ErrorBars({ res }: { res: PredictResponse }) {
  const rows = res.errors.lstm_abs.map((v, i) => ({
    step: `+${i + 1}h`,
    LSTM: v,
    Naive: res.errors.naive_abs[i],
  }));

  return (
    <div className="card">
      <div className="card-title mb-1">Per-hour absolute error</div>
      <div className="text-xs text-muted mb-3">
        |prediction − actual| at each horizon step (hours from forecast start). Lower is better.
      </div>
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={rows} margin={{ top: 10, right: 16, left: 0, bottom: 0 }}>
            <CartesianGrid stroke="#1f2a44" strokeDasharray="3 3" />
            <XAxis dataKey="step" stroke="#94a3b8" tick={{ fontSize: 11 }} />
            <YAxis
              stroke="#94a3b8"
              tick={{ fontSize: 11 }}
              label={{
                value: "|err| kW",
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
            <Bar dataKey="LSTM" fill="#60a5fa" radius={[3, 3, 0, 0]} />
            <Bar dataKey="Naive" fill="#f59e0b" radius={[3, 3, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
