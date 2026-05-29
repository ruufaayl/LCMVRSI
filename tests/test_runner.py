import json

import torch

from lcmvrsi.train.metrics import environment
from lcmvrsi.train.runner import (
    build_benchmark,
    build_lr_scheduler,
    build_model,
    run_experiment,
    save_result,
)
from lcmvrsi.utils.config import (
    BenchmarkConfig,
    ExperimentConfig,
    ModelConfig,
    TrainConfig,
)


def _tiny_config(model_name: str = "transformer", steps: int = 3) -> ExperimentConfig:
    return ExperimentConfig(
        seed=0,
        model=ModelConfig(name=model_name, params={"d_model": 16, "n_layers": 1, "n_heads": 2}),
        benchmark=BenchmarkConfig(
            name="mqar", params={"vocab_size": 32, "num_pairs": 3, "seq_len": 16}
        ),
        train=TrainConfig(steps=steps, batch_size=8, lr=1e-3, device="cpu"),
    )


def test_build_benchmark_from_config():
    from lcmvrsi.benchmarks.mqar import MQAR

    bench = build_benchmark(
        BenchmarkConfig(name="mqar", params={"vocab_size": 32, "num_pairs": 4, "seq_len": 24})
    )
    assert isinstance(bench, MQAR)
    assert bench.vocab_size == 32
    assert bench.num_pairs == 4


def test_build_model_injects_vocab_size_and_max_seq_len():
    # vocab_size and seq_len live under benchmark config; the runner must inject them.
    model = build_model(
        ModelConfig(name="transformer", params={"d_model": 16, "n_layers": 1, "n_heads": 2}),
        vocab_size=32,
        seq_len=40,
    )
    logits = model(torch.zeros(2, 40, dtype=torch.long))
    assert logits.shape == (2, 40, 32)


def test_run_experiment_returns_structured_result():
    result = run_experiment(_tiny_config(steps=3), eval_n=16)
    for key in ("config", "model", "benchmark", "train", "eval", "memory", "env"):
        assert key in result
    assert result["model"]["name"] == "transformer"
    assert result["benchmark"]["name"] == "mqar"
    acc = result["eval"]["recall_accuracy"]
    assert 0.0 <= acc <= 1.0
    assert isinstance(result["train"]["final_loss"], float)
    assert len(result["train"]["losses"]) == 3
    assert result["train"]["tokens_per_sec"] > 0
    assert result["model"]["state_size_bytes"] == 0  # transformer keeps no fixed state
    assert result["memory"]["param_count"] > 0


def test_run_experiment_linear_attention_reports_fixed_state():
    result = run_experiment(_tiny_config(model_name="linear_attention", steps=2), eval_n=16)
    # the whole point of the comparison: linear attention has a real fixed-size bottleneck
    assert result["model"]["state_size_bytes"] > 0


def test_transformer_learns_mqar_loss_decreases():
    result = run_experiment(_tiny_config(steps=150), eval_n=64)
    losses = result["train"]["losses"]
    assert result["train"]["final_loss"] < losses[0]


def test_run_experiment_is_reproducible():
    r1 = run_experiment(_tiny_config(steps=5), eval_n=16)
    r2 = run_experiment(_tiny_config(steps=5), eval_n=16)
    assert r1["train"]["losses"] == r2["train"]["losses"]
    assert r1["eval"]["recall_accuracy"] == r2["eval"]["recall_accuracy"]


def test_save_result_writes_roundtrippable_json(tmp_path):
    result = run_experiment(_tiny_config(steps=2), eval_n=16)
    path = save_result(result, tmp_path)
    assert path.exists()
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["model"]["name"] == "transformer"
    assert loaded["config"]["seed"] == 0


def test_environment_reports_versions():
    env = environment()
    assert "python" in env
    assert env["torch"]


def test_build_lr_scheduler_none_when_warmup_zero():
    opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    assert build_lr_scheduler(opt, total_steps=100, warmup_frac=0.0) is None


def test_lr_scheduler_warms_up_then_cosine_decays():
    p = torch.nn.Parameter(torch.zeros(1))
    p.grad = torch.zeros_like(p)
    opt = torch.optim.AdamW([p], lr=1.0)
    sched = build_lr_scheduler(opt, total_steps=100, warmup_frac=0.1)
    assert sched is not None

    lrs = []
    for _ in range(100):
        opt.step()
        sched.step()
        lrs.append(opt.param_groups[0]["lr"])

    assert max(lrs) > lrs[0]  # warmup raises the LR above its starting value
    assert lrs[-1] < lrs[len(lrs) // 2]  # cosine decay brings it back down
    assert lrs[-1] < 0.05  # decays toward ~0 so the solution can settle
