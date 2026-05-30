from __future__ import annotations

import inspect
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from lcmvrsi.benchmarks.base import IGNORE_INDEX, Benchmark
from lcmvrsi.benchmarks.registry import get_benchmark
from lcmvrsi.models.base import SequenceModel
from lcmvrsi.models.registry import get_model
from lcmvrsi.train.metrics import environment, peak_memory_bytes, reset_peak_memory
from lcmvrsi.utils.config import BenchmarkConfig, ExperimentConfig, ModelConfig
from lcmvrsi.utils.profiling import count_parameters
from lcmvrsi.utils.seed import set_seed

# Eval bindings are drawn from a seed far outside the training range so the reported
# recall accuracy is measured on key->value bindings the model never trained on.
_EVAL_SEED_OFFSET = 10_000_000


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer, total_steps: int, warmup_frac: float
) -> torch.optim.lr_scheduler.LambdaLR | None:
    """Linear warmup for ``warmup_frac`` of steps, then cosine decay to ~0.

    Returns None when ``warmup_frac <= 0`` (constant LR). Decaying the LR to zero is what lets
    the model settle into the exact-recall solution on harder MQAR instances, where a constant
    LR tends to leave the loss oscillating well above zero.
    """
    if warmup_frac <= 0:
        return None
    warmup = max(1, int(total_steps * warmup_frac))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_benchmark(cfg: BenchmarkConfig) -> Benchmark:
    """Instantiate a registered benchmark from its config."""
    return get_benchmark(cfg.name)(**cfg.params)


def build_model(cfg: ModelConfig, vocab_size: int, seq_len: int | None = None) -> SequenceModel:
    """Instantiate a registered model, injecting the benchmark-derived vocab/length.

    ``vocab_size`` and ``seq_len`` live under the benchmark config (the data defines them),
    so the runner injects them into model construction. Explicit ``model.params`` win, so a
    config can still override ``max_seq_len`` or ``vocab_size`` deliberately.
    """
    cls = get_model(cfg.name)
    params: dict[str, Any] = {"vocab_size": vocab_size}
    if seq_len is not None:
        params["max_seq_len"] = seq_len
    params.update(cfg.params)
    # Configs carry a common superset of knobs (e.g. n_heads); drop those a given model's
    # __init__ does not accept so heterogeneous models share one declarative config schema.
    sig = inspect.signature(cls.__init__)
    if not any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
        allowed = set(sig.parameters) - {"self"}
        params = {k: v for k, v in params.items() if k in allowed}
    return cls(**params)


def run_experiment(cfg: ExperimentConfig, eval_n: int = 256) -> dict[str, Any]:
    """Train a model on a benchmark per ``cfg`` and return a structured result dict.

    The result is JSON-serializable and self-describing: it records the config, the model's
    memory/complexity self-report, the training curve and throughput, held-out recall, and
    the software environment -- everything needed to reproduce and compare the run.
    """
    set_seed(cfg.seed)
    device = torch.device(cfg.train.device)

    benchmark = build_benchmark(cfg.benchmark)
    vocab_size = int(cfg.benchmark.params["vocab_size"])
    seq_len = int(cfg.benchmark.params["seq_len"])
    model = build_model(cfg.model, vocab_size=vocab_size, seq_len=seq_len).to(device)

    optim = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    scheduler = build_lr_scheduler(optim, cfg.train.steps, cfg.train.warmup_frac)
    reset_peak_memory(device)
    model.train()

    losses: list[float] = []
    start = time.perf_counter()
    for step in range(cfg.train.steps):
        x, y = benchmark.generate(cfg.train.batch_size, seq_len, seed=cfg.seed + 1 + step)
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size), y.reshape(-1), ignore_index=IGNORE_INDEX
        )
        optim.zero_grad()
        loss.backward()
        if cfg.train.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.max_grad_norm)
        optim.step()
        if scheduler is not None:
            scheduler.step()
        losses.append(float(loss.item()))
    wall = time.perf_counter() - start

    eval_metrics = benchmark.evaluate(model, eval_n, seq_len, seed=cfg.seed + _EVAL_SEED_OFFSET)

    n_tokens = cfg.train.steps * cfg.train.batch_size * seq_len
    param_count = count_parameters(model, trainable_only=False)
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    return {
        "config": cfg.model_dump(),
        "model": {
            "name": cfg.model.name,
            "state_size_bytes": int(model.state_size),
            "complexity": model.complexity(seq_len),
        },
        "benchmark": {"name": cfg.benchmark.name, "params": dict(cfg.benchmark.params)},
        "train": {
            "steps": cfg.train.steps,
            "batch_size": cfg.train.batch_size,
            "lr": cfg.train.lr,
            "weight_decay": cfg.train.weight_decay,
            "warmup_frac": cfg.train.warmup_frac,
            "max_grad_norm": cfg.train.max_grad_norm,
            "losses": losses,
            "final_loss": losses[-1] if losses else float("nan"),
            "wall_time_sec": wall,
            "tokens_per_sec": (n_tokens / wall) if wall > 0 else 0.0,
            "steps_per_sec": (cfg.train.steps / wall) if wall > 0 else 0.0,
        },
        "eval": eval_metrics,
        "memory": {
            "state_size_bytes": int(model.state_size),
            "param_count": int(param_count),
            "param_bytes": int(param_bytes),
            "peak_cuda_bytes": peak_memory_bytes(device),
        },
        "env": {**environment(), "device": str(device)},
    }


def save_result(result: dict[str, Any], out_dir: str | Path, name: str | None = None) -> Path:
    """Write a result dict to ``out_dir`` as indented JSON; return the file path.

    A ``name`` (without extension) can be supplied for deterministic filenames (e.g. sweeps);
    otherwise a model/benchmark/seed/timestamp stem is generated.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if name is None:
        m = result.get("model", {}).get("name", "model")
        b = result.get("benchmark", {}).get("name", "bench")
        s = result.get("config", {}).get("seed", 0)
        name = f"{m}_{b}_seed{s}_{int(time.time() * 1000)}"
    path = out / f"{name}.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return path
