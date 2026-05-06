export function fmtKw(v: number | undefined | null, digits = 1): string {
  if (v == null || Number.isNaN(v)) return "—";
  return `${v.toFixed(digits)} kW`;
}

export function fmtPct(v: number | undefined | null, digits = 2): string {
  if (v == null || Number.isNaN(v)) return "—";
  return `${v.toFixed(digits)} %`;
}

export function fmtNum(v: number | undefined | null, digits = 3): string {
  if (v == null || Number.isNaN(v)) return "—";
  return v.toFixed(digits);
}

export function fmtDate(iso: string | undefined | null): string {
  if (!iso) return "—";
  return iso.replace("T", " ").slice(0, 16);
}

export function fmtHour(iso: string | undefined | null): string {
  if (!iso) return "—";
  return iso.slice(11, 16);
}
