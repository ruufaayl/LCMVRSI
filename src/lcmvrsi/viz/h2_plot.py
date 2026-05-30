from __future__ import annotations

from pathlib import Path
from typing import Any


def _effective_state(p: dict[str, Any]) -> int:
    """Realized store bytes for gated models; worst-case/fixed state otherwise."""
    realized = p.get("realized_store_bytes")
    return realized if realized is not None else p["state_size_bytes_worstcase"]


def plot_h2_comparison(summary: dict[str, Any], out_path: str | Path) -> Path:
    """Plot the H2 achievability comparison on structured recall.

    Left: held-out recall vs. *effective* state (realized store for SGSM, fixed/worst-case
    otherwise) -- the question is whether SGSM reaches high recall at small realized state.
    Right: the surprise-gate write fraction per model (the ablation should write ~everything).
    matplotlib is imported lazily (viz extra), Agg backend. Returns the figure path.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = summary["points"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.5))

    for p in points:
        ax1.scatter(_effective_state(p), p["accuracy"], s=60)
        ax1.annotate(
            p["label"],
            (_effective_state(p), p["accuracy"]),
            fontsize=8,
            xytext=(5, 4),
            textcoords="offset points",
        )
    ax1.set_xlabel("effective state (bytes): realized store for SGSM, fixed/worst-case otherwise")
    ax1.set_ylabel("held-out recall accuracy")
    ax1.set_title("structured recall: accuracy vs. effective state")
    ax1.set_ylim(0.0, 1.02)
    ax1.grid(True, alpha=0.3)

    labels = [p["label"] for p in points]
    write_frac = [(p.get("write_fraction") or 0.0) for p in points]
    ax2.bar(range(len(labels)), write_frac)
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax2.set_ylabel("surprise-gate write fraction")
    ax2.set_title("write fraction (gate sparsity)")
    ax2.set_ylim(0.0, 1.02)
    ax2.grid(True, alpha=0.3, axis="y")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
