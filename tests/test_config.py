from pathlib import Path

import pydantic
import pytest

from lcmvrsi.utils.config import ExperimentConfig, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_load_committed_base_config():
    cfg = load_config(REPO_ROOT / "configs" / "base.yaml")
    assert isinstance(cfg, ExperimentConfig)
    assert cfg.model.name
    assert cfg.benchmark.name
    assert cfg.train.device in {"cpu", "cuda"}


def test_missing_required_fields_raise():
    with pytest.raises(pydantic.ValidationError):
        ExperimentConfig.model_validate({"seed": 0})  # no model/benchmark


def test_load_from_dict_roundtrip(tmp_path):
    import yaml

    data = {
        "seed": 7,
        "model": {"name": "transformer", "params": {"d_model": 32}},
        "benchmark": {"name": "mqar", "params": {"seq_len": 64}},
        "train": {"steps": 10, "batch_size": 4, "lr": 0.01, "device": "cpu"},
    }
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    cfg = load_config(p)
    assert cfg.seed == 7
    assert cfg.model.params["d_model"] == 32
    assert cfg.train.steps == 10
