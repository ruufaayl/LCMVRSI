from __future__ import annotations

from pathlib import Path
from typing import Any

from lcmvrsi.train.runner import run_experiment, save_result
from lcmvrsi.utils.config import (
    BenchmarkConfig,
    ExperimentConfig,
    ModelConfig,
    TrainConfig,
)


def run_frontier(
    model_names: list[str],
    benchmark_name: str,
    *,
    benchmark_params: dict[str, Any],
    d_model: int = 64,
    n_layers: int = 2,
    n_heads: int = 4,
    steps: int = 2500,
    batch_size: int = 32,
    lr: float = 2e-3,
    weight_decay: float = 0.01,
    warmup_frac: float = 0.1,
    max_grad_norm: float = 1.0,
    seed: int = 0,
    device: str = "cpu",
    eval_n: int = 256,
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Train every model on one benchmark at a fixed difficulty; collect the frontier points.

    All models share width/depth/budget, so each point's ``(state_size_bytes, accuracy)`` and
    ``(tokens_per_sec, accuracy)`` place it on the empirical recall-memory and recall-throughput
    frontiers. ``n_heads`` is carried in the shared config and silently dropped by models that
    do not use it (see ``build_model``), so heterogeneous architectures compare on one schema.
    """
    points: list[dict[str, Any]] = []
    for model_name in model_names:
        cfg = ExperimentConfig(
            seed=seed,
            model=ModelConfig(
                name=model_name,
                params={"d_model": d_model, "n_layers": n_layers, "n_heads": n_heads},
            ),
            benchmark=BenchmarkConfig(name=benchmark_name, params=dict(benchmark_params)),
            train=TrainConfig(
                steps=steps,
                batch_size=batch_size,
                lr=lr,
                weight_decay=weight_decay,
                warmup_frac=warmup_frac,
                max_grad_norm=max_grad_norm,
                device=device,
            ),
        )
        result = run_experiment(cfg, eval_n=eval_n)
        if out_dir is not None:
            save_result(
                result, Path(out_dir) / "runs", name=f"{benchmark_name}_{model_name}_seed{seed}"
            )
        points.append(
            {
                "model": model_name,
                "accuracy": result["eval"]["accuracy"],
                "state_size_bytes": result["model"]["state_size_bytes"],
                "param_count": result["memory"]["param_count"],
                "tokens_per_sec": result["train"]["tokens_per_sec"],
                "final_loss": result["train"]["final_loss"],
            }
        )

    summary: dict[str, Any] = {
        "setup": {
            "benchmark": benchmark_name,
            "benchmark_params": dict(benchmark_params),
            "model_names": model_names,
            "d_model": d_model,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "steps": steps,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "warmup_frac": warmup_frac,
            "max_grad_norm": max_grad_norm,
            "seed": seed,
            "device": device,
            "eval_n": eval_n,
        },
        "points": points,
    }
    return summary
