import json

from lcmvrsi.train.cli import apply_overrides, build_parser, main
from lcmvrsi.utils.config import load_config

BASE_CONFIG = "configs/base.yaml"


def test_build_parser_parses_core_args():
    parser = build_parser()
    args = parser.parse_args(["-c", BASE_CONFIG, "--steps", "5", "--device", "cpu", "--seed", "3"])
    assert args.config == BASE_CONFIG
    assert args.steps == 5
    assert args.device == "cpu"
    assert args.seed == 3


def test_apply_overrides_changes_nested_fields():
    cfg = load_config(BASE_CONFIG)
    out = apply_overrides(cfg, steps=7, device="cpu", seed=11)
    assert out.train.steps == 7
    assert out.train.device == "cpu"
    assert out.seed == 11
    # untouched fields preserved
    assert out.model.name == cfg.model.name
    assert out.benchmark.params == cfg.benchmark.params


def test_apply_overrides_none_is_noop():
    cfg = load_config(BASE_CONFIG)
    out = apply_overrides(cfg, steps=None, device=None, seed=None)
    assert out.train.steps == cfg.train.steps
    assert out.seed == cfg.seed


def test_main_runs_end_to_end_and_writes_json(tmp_path):
    path = main(
        [
            "-c",
            BASE_CONFIG,
            "--steps",
            "2",
            "--eval-n",
            "16",
            "-o",
            str(tmp_path),
            "--name",
            "smoke",
        ]
    )
    assert path.exists()
    assert path.name == "smoke.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    assert 0.0 <= result["eval"]["recall_accuracy"] <= 1.0
    assert result["train"]["steps"] == 2
    assert result["config"]["benchmark"]["params"]["vocab_size"] == 64
