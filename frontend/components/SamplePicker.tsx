"use client";

import { useEffect, useState } from "react";
import { listTestWindows } from "@/lib/api";
import type { TestWindowItem } from "@/lib/types";
import { fmtDate } from "@/lib/format";

interface Props {
  onPick: (item: TestWindowItem) => void;
  current?: TestWindowItem | null;
}

const PAGE = 100;

export function SamplePicker({ onPick, current }: Props) {
  const [items, setItems] = useState<TestWindowItem[]>([]);
  const [total, setTotal] = useState(0);
  const [offset, setOffset] = useState(0);
  const [q, setQ] = useState("");
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  async function load(nextOffset = offset, nextQ = q) {
    setLoading(true);
    setErr(null);
    try {
      const res = await listTestWindows({
        limit: PAGE,
        offset: nextOffset,
        q: nextQ || undefined,
      });
      setItems(res.items);
      setTotal(res.total);
      setOffset(res.offset);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load(0, "");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function pickRandom() {
    setLoading(true);
    setErr(null);
    try {
      const res = await listTestWindows({ random: true });
      if (res.items[0]) onPick(res.items[0]);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  function pickFirst() {
    if (items[0]) onPick(items[0]);
  }

  function pickLast() {
    const last = items[items.length - 1];
    if (last) onPick(last);
  }

  return (
    <div className="space-y-3">
      <div>
        <div className="label mb-2">Pick a test window</div>
        <div className="flex flex-wrap gap-2">
          <button className="btn" onClick={pickFirst} disabled={loading || !items.length}>
            First
          </button>
          <button className="btn" onClick={pickLast} disabled={loading || !items.length}>
            Last
          </button>
          <button className="btn" onClick={pickRandom} disabled={loading}>
            Random
          </button>
        </div>
      </div>

      <div>
        <label className="label">Filter by date prefix (e.g. 2014-08)</label>
        <div className="flex gap-2 mt-1">
          <input
            className="input"
            placeholder="YYYY-MM or YYYY-MM-DD"
            value={q}
            onChange={(e) => setQ(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") load(0, q);
            }}
          />
          <button className="btn" onClick={() => load(0, q)} disabled={loading}>
            Apply
          </button>
        </div>
      </div>

      <div>
        <div className="flex items-center justify-between mb-2">
          <div className="label">
            {total.toLocaleString()} window{total === 1 ? "" : "s"}
          </div>
          <div className="flex gap-2">
            <button
              className="btn"
              disabled={loading || offset <= 0}
              onClick={() => load(Math.max(0, offset - PAGE), q)}
            >
              Prev
            </button>
            <button
              className="btn"
              disabled={loading || offset + PAGE >= total}
              onClick={() => load(offset + PAGE, q)}
            >
              Next
            </button>
          </div>
        </div>
        <div className="border border-border rounded-lg max-h-72 overflow-auto bg-panel2">
          {loading && <div className="p-3 text-sm text-muted">Loading…</div>}
          {!loading && items.length === 0 && (
            <div className="p-3 text-sm text-muted">No windows match.</div>
          )}
          {!loading &&
            items.map((it) => {
              const active = current?.t === it.t;
              return (
                <button
                  key={it.t}
                  onClick={() => onPick(it)}
                  className={`w-full text-left px-3 py-2 text-sm border-b border-border/60 last:border-0 hover:bg-border/40 ${
                    active ? "bg-border/60 text-white" : "text-slate-200"
                  }`}
                >
                  <div className="font-mono">{fmtDate(it.first_pred_time)}</div>
                  <div className="text-xs text-muted">
                    t={it.t} · ends {fmtDate(it.last_pred_time)}
                  </div>
                </button>
              );
            })}
        </div>
      </div>

      {err && <div className="text-sm text-bad">{err}</div>}
    </div>
  );
}
