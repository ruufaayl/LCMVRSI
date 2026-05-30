#!/usr/bin/env python
"""M4.4 --- H2 achievability test on the structured-recall task.

Trains SGSM, baselines, and a gate-ablation at matched width/depth/budget on `structured_recall`
(predictable cyclic filler + sparse novel bindings), and reports held-out recall against each
model's state. For SGSM the *realized* store size (mean writes) is logged, so we can ask whether
recall is attained at state far below the worst case --- the achievability question Proposition 1
leaves open. Honest by construction: a null/negative outcome is a valid result.

Usage::

    uv run --extra viz python experiments/h2_compare.py
    uv run --extra viz python experiments/h2_compare.py --steps 4000 --seq-len 96 --num-pairs 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lcmvrsi.train.runner import run_experiment, save_result
from lcmvrsi.utils.config import (
    BenchmarkConfig,
    ExperimentConfig,
    ModelConfig,
    TrainConfig,
)

# (label, registered model name, extra model params)
_RUNS = [
    ("transformer", "transformer", {}),
    ("linear_attention", "linear_attention", {}),
    ("sgsm", "sgsm", {}),
    ("sgsm_no_gate", "sgsm", {"init_threshold": -20.0}),  # ablation: gate always open
]


def main(argv: list[str] | None = None) -> Path:
    p = argparse.ArgumentParser(description="H2 achievability on structured recall.")
    p.add_argument("--vocab-size", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=96)
    p.add_argument("--num-pairs", type=int, default=8)
    p.add_argument("--cycle-len", type=int, default=8)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--steps", type=int, default=3000)
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

    bench_params = {
        "vocab_size": args.vocab_size,
        "num_pairs": args.num_pairs,
        "cycle_len": args.cycle_len,
        "seq_len": args.seq_len,
    }

    points = []
    for label, model_name, extra in _RUNS:
        cfg = ExperimentConfig(
            seed=args.seed,
            model=ModelConfig(
                name=model_name,
                params={"d_model": args.d_model, "n_layers": args.n_layers,
                        "n_heads": args.n_heads, **extra},
            ),
            benchmark=BenchmarkConfig(name="structured_recall", params=dict(bench_params)),
            train=TrainConfig(
                steps=args.steps, batch_size=args.batch_size, lr=args.lr,
                weight_decay=args.weight_decay, warmup_frac=args.warmup_frac,
                max_grad_norm=args.max_grad_norm, device=args.device,
            ),
        )
        result = run_experiment(cfg, eval_n=args.eval_n)
        save_result(result, Path(args.out) / "runs", name=f"h2_{label}_seed{args.seed}")
        mem = result["memory"]
        writes = mem.get("realized_writes_per_seq")
        realized_store_bytes = (
            int(writes * 2 * args.d_model * 4) if writes is not None else None
        )
        points.append({
            "label": label,
            "model": model_name,
            "accuracy": result["eval"]["accuracy"],
            "state_size_bytes_worstcase": mem["state_size_bytes"],
            "realized_writes_per_seq": writes,
            "write_fraction": mem.get("write_fraction"),
            "realized_store_bytes": realized_store_bytes,
            "tokens_per_sec": result["train"]["tokens_per_sec"],
            "param_count": mem["param_count"],
        })

    summary = {"setup": {"benchmark": "structured_recall", **bench_params,
                         "d_model": args.d_model, "n_layers": args.n_layers,
                         "n_heads": args.n_heads, "steps": args.steps, "lr": args.lr,
                         "seed": args.seed, "eval_n": args.eval_n}, "points": points}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "h2_structured_recall.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"summary -> {summary_path}")
    for pt in points:
        wf = pt["write_fraction"]
        wf_s = f"{wf:.2f}" if wf is not None else "  - "
        print(
            f"  {pt['label']:>16s}  acc={pt['accuracy']:.3f}  write_frac={wf_s}  "
            f"realized_store={pt['realized_store_bytes']}  worst={pt['state_size_bytes_worstcase']}"
        )

    if not args.no_plot:
        try:
            from lcmvrsi.viz.h2_plot import plot_h2_comparison

            fig = plot_h2_comparison(summary, out / "figures" / "h2_structured_recall.png")
            print(f"figure  -> {fig}")
        except ImportError:
            print("matplotlib not installed; re-run with `uv run --extra viz ...` for the figure.")

    return summary_path


if __name__ == "__main__":
    main()
