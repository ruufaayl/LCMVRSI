#!/usr/bin/env python
"""Sweep MQAR recall vs. num_pairs for transformer and linear_attention.

Trains every (model, num_pairs) cell at matched width/depth/budget, writes a committed
summary JSON plus per-run JSONs (gitignored), and renders the recall-vs-difficulty figure.

Usage::

    uv run --extra viz python experiments/sweep_recall.py
    uv run --extra viz python experiments/sweep_recall.py --num-pairs 1 4 8 16 32 --steps 3000

The figure step needs the ``viz`` extra (matplotlib); without it the sweep still runs and the
summary is written -- re-run with ``--extra viz`` to render the figure from the summary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lcmvrsi.train.sweep import run_sweep


def main(argv: list[str] | None = None) -> Path:
    p = argparse.ArgumentParser(description="MQAR recall-vs-difficulty sweep.")
    p.add_argument("--models", nargs="+", default=["transformer", "linear_attention"])
    p.add_argument("--num-pairs", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    p.add_argument("--vocab-size", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=48)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-frac", type=float, default=0.1)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--eval-n", type=int, default=256)
    p.add_argument("-o", "--out", default="results")
    p.add_argument("--no-plot", action="store_true", help="Skip rendering the figure.")
    args = p.parse_args(argv)

    summary = run_sweep(
        args.models,
        args.num_pairs,
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_frac=args.warmup_frac,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        device=args.device,
        eval_n=args.eval_n,
        out_dir=args.out,
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "recall_sweep_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"summary -> {summary_path}")
    for pt in summary["points"]:
        print(
            f"  {pt['model']:>16s}  pairs={pt['num_pairs']:>3d}  "
            f"recall={pt['recall_accuracy']:.3f}  state_bytes={pt['state_size_bytes']}"
        )

    if not args.no_plot:
        try:
            from lcmvrsi.viz.recall_plot import plot_recall_vs_pairs

            fig_path = plot_recall_vs_pairs(summary, out / "figures" / "recall_vs_pairs.png")
            print(f"figure  -> {fig_path}")
        except ImportError:
            print("matplotlib not installed; re-run with `uv run --extra viz ...` for the figure.")

    return summary_path


if __name__ == "__main__":
    main()
