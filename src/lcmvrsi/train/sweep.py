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


def _make_config(
    model_name: str,
    num_pairs: int,
    *,
    vocab_size: int,
    seq_len: int,
    d_model: int,
    n_layers: int,
    n_heads: int,
    steps: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    warmup_frac: float,
    max_grad_norm: float,
    seed: int,
    device: str,
) -> ExperimentConfig:
    return ExperimentConfig(
        seed=seed,
        model=ModelConfig(
            name=model_name,
            params={"d_model": d_model, "n_layers": n_layers, "n_heads": n_heads},
        ),
        benchmark=BenchmarkConfig(
            name="mqar",
            params={"vocab_size": vocab_size, "num_pairs": num_pairs, "seq_len": seq_len},
        ),
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


def run_sweep(
    model_names: list[str],
    num_pairs_list: list[int],
    *,
    vocab_size: int = 64,
    seq_len: int = 64,
    d_model: int = 64,
    n_layers: int = 2,
    n_heads: int = 2,
    steps: int = 2000,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    warmup_frac: float = 0.0,
    max_grad_norm: float = 0.0,
    seed: int = 0,
    device: str = "cpu",
    eval_n: int = 512,
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Train each model at each ``num_pairs`` and collect held-out recall vs. difficulty.

    All models share identical width/depth and training budget, so the only thing that
    varies within a model's curve is the recall difficulty (``num_pairs``); the only thing
    that varies across curves is the architecture. That isolation is what lets the resulting
    figure speak to the recall-memory tradeoff rather than to a confound.

    When ``out_dir`` is given, every individual run's full result JSON is written under
    ``out_dir/runs/`` (gitignored, regenerable) for auditability. The returned summary holds
    one compact point per run.
    """
    points: list[dict[str, Any]] = []
    for model_name in model_names:
        for num_pairs in num_pairs_list:
            cfg = _make_config(
                model_name,
                num_pairs,
                vocab_size=vocab_size,
                seq_len=seq_len,
                d_model=d_model,
                n_layers=n_layers,
                n_heads=n_heads,
                steps=steps,
                batch_size=batch_size,
                lr=lr,
                weight_decay=weight_decay,
                warmup_frac=warmup_frac,
                max_grad_norm=max_grad_norm,
                seed=seed,
                device=device,
            )
            result = run_experiment(cfg, eval_n=eval_n)
            if out_dir is not None:
                save_result(
                    result,
                    Path(out_dir) / "runs",
                    name=f"{model_name}_pairs{num_pairs}_seed{seed}",
                )
            points.append(
                {
                    "model": model_name,
                    "num_pairs": num_pairs,
                    "recall_accuracy": result["eval"]["recall_accuracy"],
                    "num_queries": result["eval"]["num_queries"],
                    "state_size_bytes": result["model"]["state_size_bytes"],
                    "final_loss": result["train"]["final_loss"],
                    "param_count": result["memory"]["param_count"],
                }
            )

    summary: dict[str, Any] = {
        "setup": {
            "model_names": model_names,
            "num_pairs_list": num_pairs_list,
            "vocab_size": vocab_size,
            "seq_len": seq_len,
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
