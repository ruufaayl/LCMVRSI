from __future__ import annotations

from pathlib import Path
from typing import Any


def plot_frontier(summary: dict[str, Any], out_path: str | Path) -> Path:
    """Plot the recall-memory and recall-throughput frontiers (one labeled point per model).

    Left: held-out accuracy vs. fixed recurrent-state bytes (0 = a growing cache / no fixed
    state). Right: accuracy vs. throughput. matplotlib is imported lazily (viz extra) on the Agg
    backend so this runs headless. Returns the figure path.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = summary["points"]
    bench = summary.get("setup", {}).get("benchmark", "task")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.5))
    for p in points:
        ax1.scatter(p["state_size_bytes"], p["accuracy"], s=60)
        ax1.annotate(
            p["model"],
            (p["state_size_bytes"], p["accuracy"]),
            fontsize=8,
            xytext=(5, 4),
            textcoords="offset points",
        )
        ax2.scatter(p["tokens_per_sec"], p["accuracy"], s=60)
        ax2.annotate(
            p["model"],
            (p["tokens_per_sec"], p["accuracy"]),
            fontsize=8,
            xytext=(5, 4),
            textcoords="offset points",
        )

    ax1.set_xlabel("fixed recurrent state (bytes); 0 = growing cache / no fixed state")
    ax1.set_ylabel("held-out accuracy")
    ax1.set_title(f"recall-memory frontier ({bench})")
    ax1.set_ylim(0.0, 1.02)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("throughput (tokens/sec, train)")
    ax2.set_ylabel("held-out accuracy")
    ax2.set_title(f"recall-throughput frontier ({bench})")
    ax2.set_ylim(0.0, 1.02)
    ax2.grid(True, alpha=0.3)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
