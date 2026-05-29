from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class BenchmarkConfig(BaseModel):
    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class TrainConfig(BaseModel):
    steps: int = 1000
    batch_size: int = 32
    lr: float = 1e-3
    device: str = "cpu"


class ExperimentConfig(BaseModel):
    seed: int = 0
    model: ModelConfig
    benchmark: BenchmarkConfig
    train: TrainConfig = Field(default_factory=TrainConfig)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment config from a YAML file."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return ExperimentConfig.model_validate(data)
