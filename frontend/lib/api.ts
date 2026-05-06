import type {
  HealthResponse,
  MetaResponse,
  ModelChoice,
  PredictResponse,
  TestWindowsResponse,
} from "./types";

export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE?.replace(/\/$/, "") || "http://localhost:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers || {}),
    },
    cache: "no-store",
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`API ${res.status} ${res.statusText}: ${text || path}`);
  }
  return (await res.json()) as T;
}

export const fetcher = <T,>(path: string) => request<T>(path);

export function getHealth() {
  return request<HealthResponse>("/api/health");
}

export function getMeta() {
  return request<MetaResponse>("/api/meta");
}

export function listTestWindows(opts: {
  limit?: number;
  offset?: number;
  q?: string;
  random?: boolean;
}) {
  const p = new URLSearchParams();
  if (opts.limit != null) p.set("limit", String(opts.limit));
  if (opts.offset != null) p.set("offset", String(opts.offset));
  if (opts.q) p.set("q", opts.q);
  if (opts.random) p.set("random", "true");
  const qs = p.toString();
  return request<TestWindowsResponse>(`/api/test-windows${qs ? `?${qs}` : ""}`);
}

export function predict(t: number, model: ModelChoice) {
  return request<PredictResponse>("/api/predict", {
    method: "POST",
    body: JSON.stringify({ t, model }),
  });
}
