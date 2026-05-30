#!/usr/bin/env python
"""Generate the supporting 'state spectrum' figure and stage all figures for paper + LinkedIn.

Reproducible: derives the state-size bar chart from the live registered models (so it cannot
drift), then copies the committed result figures into ``paper/figures/`` (for the LaTeX build)
and ``marketing/linkedin/images/`` (for easy extraction into posts).

Usage::

    uv run --extra viz python experiments/make_figures.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

from lcmvrsi.analysis import complexity_table

RESULTS_FIG = Path("results/figures")
PAPER_FIG = Path("paper/figures")
LINKEDIN_FIG = Path("marketing/linkedin/images")

# The three figures produced by the tested experiment code (recall sweep / frontier / H2).
_RESULT_FIGURES = ["recall_vs_pairs.png", "frontier_mqar.png", "h2_structured_recall.png"]


def make_state_spectrum(out_path: Path) -> Path:
    """Horizontal bar chart of fixed recurrent-state bytes per architecture (d_model=64)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = sorted(complexity_table(seq_len=128), key=lambda r: r["state_size_bytes"])
    names = [r["model"] for r in rows]
    bytes_ = [r["state_size_bytes"] for r in rows]

    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.barh(range(len(names)), bytes_)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("fixed recurrent state (bytes) at d_model=64; bigger is more to compress into")
    ax.set_title("State per architecture: the axis the recall-memory bound constrains")
    for i, r in enumerate(rows):
        ax.annotate(f"  {r['state_kind']}", (r["state_size_bytes"], i), va="center", fontsize=8)
    ax.margins(x=0.25)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    spectrum = make_state_spectrum(RESULTS_FIG / "state_spectrum.png")
    print(f"generated -> {spectrum}")

    PAPER_FIG.mkdir(parents=True, exist_ok=True)
    LINKEDIN_FIG.mkdir(parents=True, exist_ok=True)
    for name in _RESULT_FIGURES:
        src = RESULTS_FIG / name
        if not src.exists():
            print(f"  WARNING: {src} missing (run its experiment first); skipping")
            continue
        shutil.copy2(src, PAPER_FIG / name)
        shutil.copy2(src, LINKEDIN_FIG / name)
    # state_spectrum goes to LinkedIn assets too
    shutil.copy2(spectrum, LINKEDIN_FIG / spectrum.name)
    print(f"staged figures -> {PAPER_FIG}/ and {LINKEDIN_FIG}/")


if __name__ == "__main__":
    main()
