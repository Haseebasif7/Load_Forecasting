#!/usr/bin/env python3
"""Draw layered system architecture (matplotlib) for SRS §7.4."""

from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out = root / "docs" / "figures" / "system_architecture.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9.5, 11), dpi=160)
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.35, 11.2)
    ax.axis("off")
    fig.patch.set_facecolor("#f7fafc")
    ax.set_facecolor("#f7fafc")

    colors = {
        "client": "#1a365d",
        "api": "#2c5282",
        "infer": "#2b6cb0",
        "ml": "#4a5568",
        "data": "#22543d",
        "side": "#744210",
    }

    def rounded_box(
        cx: float,
        cy: float,
        w: float,
        h: float,
        face: str,
        lines: list[str],
        *,
        fs: float = 8.6,
        text_color: str = "white",
    ) -> None:
        x, y = cx - w / 2, cy - h / 2
        patch = mpatches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.04,rounding_size=0.18",
            facecolor=face,
            edgecolor="#e2e8f0",
            linewidth=1.4,
        )
        ax.add_patch(patch)
        ax.text(
            cx,
            cy,
            "\n".join(lines),
            ha="center",
            va="center",
            color=text_color,
            fontsize=fs,
            fontweight="semibold",
            linespacing=1.38,
        )

    def arrow_down(x: float, y_tip: float, y_tail: float) -> None:
        """Arrow from y_tail (higher on page) to y_tip (lower)."""
        ax.annotate(
            "",
            xy=(x, y_tip),
            xytext=(x, y_tail),
            arrowprops=dict(arrowstyle="-|>", color="#2d3748", lw=2.0, mutation_scale=14),
        )

    ax.text(
        5,
        10.55,
        "Electricity Load Forecasting — System Architecture",
        ha="center",
        va="center",
        fontsize=14.5,
        fontweight="bold",
        color="#1a202c",
    )
    ax.text(
        5,
        10.05,
        "Layered deployment (course demo): browser → API → inference → ML core → artifacts",
        ha="center",
        va="center",
        fontsize=10,
        color="#4a5568",
    )

    # --- Presentation ---
    cy = 8.85
    h1 = 1.15
    rounded_box(
        5,
        cy,
        8.4,
        h1,
        colors["client"],
        [
            "Presentation layer — Next.js 14",
            "Dashboard :3000  •  metrics, test-window picker, forecast & error charts",
        ],
        fs=8.8,
    )
    y_edge_bottom = cy - h1 / 2
    arrow_down(5, y_edge_bottom - 0.4, y_edge_bottom - 0.02)
    ax.text(
        5,
        y_edge_bottom - 0.45,
        "HTTP/JSON  •  NEXT_PUBLIC_API_BASE",
        ha="center",
        va="top",
        fontsize=8.5,
        color="#2d3748",
        style="italic",
    )

    # --- FastAPI ---
    cy = 6.55
    h2 = 1.35
    rounded_box(
        5,
        cy,
        8.4,
        h2,
        colors["api"],
        [
            "Application layer — FastAPI (ASGI) :8000",
            "GET /api/health, /api/meta, /api/test-windows",
            "POST /api/predict  •  CORS + EXTRA_CORS_ORIGINS",
        ],
        fs=8.5,
    )
    y2_top = cy + h2 / 2
    arrow_down(5, cy - h2 / 2 - 0.02, y2_top + 0.02)

    # --- Inference ---
    cy = 4.35
    h3 = 1.55
    rounded_box(
        5,
        cy,
        8.4,
        h3,
        colors["infer"],
        [
            "Inference — backend/app/inference.py (LoadedState)",
            "Artifact load: splits.json, scaler.joblib, metrics.json, lstm_load.pt, ensemble members",
            "Validate t ∈ test_indices  •  window W=336  •  PyTorch eval  •  naive baseline + inverse scaler",
        ],
        fs=8.2,
    )
    y3_top = cy + h3 / 2
    arrow_down(5, cy - h3 / 2 - 0.02, y3_top + 0.02)

    # --- ML core ---
    cy = 2.35
    h4 = 1.15
    rounded_box(
        5,
        cy,
        8.4,
        h4,
        colors["ml"],
        [
            "ML core (src/) — not on inference request path",
            "data.py  •  windows.py  •  models.py  •  train.py  •  eval.py",
        ],
        fs=8.4,
    )
    y4_top = cy + h4 / 2
    arrow_down(5, cy - h4 / 2 - 0.02, y4_top + 0.02)

    # --- Persistence ---
    cy = 0.75
    h5 = 1.05
    rounded_box(
        5,
        cy,
        8.4,
        h5,
        colors["data"],
        [
            "Persistence — processed parquet, Modal Volume electricity-data (optional)",
            "*.pt  •  ensemble_manifest.json  •  metrics.json  •  figures/",
        ],
        fs=8.0,
    )
    y5_top = cy + h5 / 2
    arrow_down(5, cy - h5 / 2 - 0.02, y5_top + 0.02)

    # --- Offline pipeline (side note) ---
    rounded_box(
        8.55,
        4.35,
        2.75,
        1.25,
        colors["side"],
        [
            "Offline pipeline",
            "Modal GPU:",
            "preprocess → train → eval",
        ],
        fs=7.6,
    )
    ax.annotate(
        "",
        xy=(6.45, 1.22),
        xytext=(8.0, 3.72),
        arrowprops=dict(
            arrowstyle="-|>",
            color="#718096",
            lw=1.6,
            linestyle=(0, (4, 3)),
            mutation_scale=11,
            connectionstyle="arc3,rad=0.08",
        ),
    )
    ax.text(
        5,
        0.08,
        "Dashed arrow: Modal preprocess / train / eval writes checkpoints and metrics into the persistence layer (not invoked per HTTP request).",
        ha="center",
        va="bottom",
        fontsize=7.9,
        color="#4a5568",
    )

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor=fig.patch.get_facecolor(), pad_inches=0.2)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
