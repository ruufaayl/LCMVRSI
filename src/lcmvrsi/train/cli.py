from __future__ import annotations

import argparse
from pathlib import Path

from lcmvrsi.train.runner import run_experiment, save_result
from lcmvrsi.utils.config import ExperimentConfig, load_config


def build_parser() -> argparse.ArgumentParser:
    """Build the experiment-runner argument parser."""
    parser = argparse.ArgumentParser(
        prog="lcmvrsi-run",
        description="Train a model on a benchmark from a YAML config and log a result JSON.",
    )
    parser.add_argument("-c", "--config", required=True, help="Path to the experiment YAML config.")
    parser.add_argument("-o", "--out", default="results", help="Directory for the result JSON.")
    parser.add_argument("--name", default=None, help="Result filename stem (no extension).")
    parser.add_argument("--steps", type=int, default=None, help="Override train.steps.")
    parser.add_argument("--device", default=None, help="Override train.device (e.g. cpu, cuda).")
    parser.add_argument("--seed", type=int, default=None, help="Override the experiment seed.")
    parser.add_argument("--eval-n", type=int, default=256, help="Held-out sequences for eval.")
    return parser


def apply_overrides(
    cfg: ExperimentConfig,
    *,
    steps: int | None = None,
    device: str | None = None,
    seed: int | None = None,
) -> ExperimentConfig:
    """Return a copy of ``cfg`` with the given CLI overrides applied (None = leave unchanged)."""
    data = cfg.model_dump()
    if steps is not None:
        data["train"]["steps"] = steps
    if device is not None:
        data["train"]["device"] = device
    if seed is not None:
        data["seed"] = seed
    return ExperimentConfig.model_validate(data)


def main(argv: list[str] | None = None) -> Path:
    """Parse args, run the configured experiment, save the result JSON, and return its path."""
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, steps=args.steps, device=args.device, seed=args.seed)

    result = run_experiment(cfg, eval_n=args.eval_n)
    path = save_result(result, args.out, name=args.name)

    print(
        f"[{cfg.model.name} / {cfg.benchmark.name}] "
        f"recall_accuracy={result['eval']['recall_accuracy']:.4f} "
        f"final_loss={result['train']['final_loss']:.4f} "
        f"state_bytes={result['model']['state_size_bytes']} "
        f"tokens/s={result['train']['tokens_per_sec']:.0f} "
        f"-> {path}"
    )
    return path


if __name__ == "__main__":
    main()
