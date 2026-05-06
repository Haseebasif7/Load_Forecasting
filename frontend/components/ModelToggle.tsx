import clsx from "clsx";
import type { ModelChoice } from "@/lib/types";

interface Props {
  value: ModelChoice;
  onChange: (v: ModelChoice) => void;
  available: string[];
  bestSeed?: number;
  ensembleN?: number;
}

export function ModelToggle({ value, onChange, available, bestSeed, ensembleN }: Props) {
  const ensembleOk = available.includes("ensemble");
  return (
    <div>
      <div className="label mb-2">Model</div>
      <div className="grid grid-cols-2 gap-2">
        <button
          type="button"
          onClick={() => onChange("single")}
          className={clsx(
            "btn",
            value === "single" && "btn-primary",
          )}
        >
          Single best
          {bestSeed != null && (
            <span className="ml-2 text-xs opacity-70">seed {bestSeed}</span>
          )}
        </button>
        <button
          type="button"
          onClick={() => ensembleOk && onChange("ensemble")}
          disabled={!ensembleOk}
          className={clsx(
            "btn",
            value === "ensemble" && "btn-primary",
          )}
          title={ensembleOk ? "Average of all ensemble members" : "Ensemble checkpoints not found"}
        >
          Ensemble mean
          {ensembleN != null && ensembleOk && (
            <span className="ml-2 text-xs opacity-70">n={ensembleN}</span>
          )}
        </button>
      </div>
    </div>
  );
}
