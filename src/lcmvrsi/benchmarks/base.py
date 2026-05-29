from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from lcmvrsi.models.base import SequenceModel


class Benchmark(ABC):
    """Common interface for synthetic sequence benchmarks."""

    @abstractmethod
    def generate(self, n: int, seq_len: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (inputs (n, T) long, targets (n, T) long) for a split."""
        raise NotImplementedError

    @abstractmethod
    def evaluate(
        self, model: SequenceModel, n: int, seq_len: int, seed: int
    ) -> dict[str, float]:
        """Return a metrics dict, e.g. {'accuracy': 0.9}."""
        raise NotImplementedError
