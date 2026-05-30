#!/usr/bin/env python
"""Map the empirical recall-memory / recall-throughput frontier across all models.

Trains every model on one benchmark at a fixed difficulty (matched width/depth/budget) and
plots held-out accuracy against each model's fixed recurrent-state size and throughput.

Usage::

    uv run --extra viz python experiments/frontier.py
    uv run --extra viz python experiments/frontier.py --benchmark induction --seq-len 64
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lcmvrsi.train.frontier import run_frontier

_ALL_MODELS = ["transformer", "hyena", "linear_attention", "ssm", "rwkv"]


def _benchmark_params(args: argparse.Namespace) -> dict[str, object]:
    params: dict[str, object] = {"vocab_size": args.vocab_size, "seq_len": args.seq_len}
    if args.benchmark == "mqar":
        params["num_pairs"] = args.num_pairs
    elif args.benchmark == "copying":
        params["copy_len"] = args.copy_len
    return params


def main(argv: list[str] | None = None) -> Path:
    p = argparse.ArgumentParser(description="Recall-memory frontier across models.")
    p.add_argument("--models", nargs="+", default=_ALL_MODELS)
    p.add_argument(
        "--benchmark", default="mqar", choices=["mqar", "copying", "induction", "needle"]
    )
    p.add_argument("--vocab-size", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=24)
    p.add_argument("--num-pairs", type=int, default=8, help="mqar difficulty")
    p.add_argument("--copy-len", type=int, default=8, help="copying difficulty")
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--steps", type=int, default=2500)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-frac", type=float, default=0.1)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--eval-n", type=int, default=256)
    p.add_argument("-o", "--out", default="results")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args(argv)

    summary = run_frontier(
        args.models,
        args.benchmark,
        benchmark_params=_benchmark_params(args),
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
    summary_path = out / f"frontier_{args.benchmark}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"summary -> {summary_path}")
    for pt in sorted(summary["points"], key=lambda p: p["state_size_bytes"]):
        print(
            f"  {pt['model']:>16s}  acc={pt['accuracy']:.3f}  "
            f"state_bytes={pt['state_size_bytes']:>6d}  tok/s={pt['tokens_per_sec']:.0f}"
        )

    if not args.no_plot:
        try:
            from lcmvrsi.viz.frontier_plot import plot_frontier

            fig_path = plot_frontier(summary, out / "figures" / f"frontier_{args.benchmark}.png")
            print(f"figure  -> {fig_path}")
        except ImportError:
            print("matplotlib not installed; re-run with `uv run --extra viz ...` for the figure.")

    return summary_path


if __name__ == "__main__":
    main()
