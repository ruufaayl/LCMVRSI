from __future__ import annotations

from pathlib import Path
from typing import Any


def plot_recall_vs_pairs(summary: dict[str, Any], out_path: str | Path) -> Path:
    """Render held-out recall vs. ``num_pairs``, one line per model, to ``out_path``.

    matplotlib is imported lazily (it lives in the optional ``viz`` extra) and forced onto the
    non-interactive Agg backend so this works headless (CI, servers). Returns the figure path.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = summary["points"]
    models: list[str] = []
    for p in points:
        if p["model"] not in models:
            models.append(p["model"])

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for model in models:
        pts = sorted(
            (p for p in points if p["model"] == model), key=lambda p: p["num_pairs"]
        )
        xs = [p["num_pairs"] for p in pts]
        ys = [p["recall_accuracy"] for p in pts]
        ax.plot(xs, ys, marker="o", label=model)

    ax.set_xlabel("number of key-value pairs (associative-recall difficulty)")
    ax.set_ylabel("held-out recall accuracy")
    ax.set_title("MQAR: recall vs. recall difficulty")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
