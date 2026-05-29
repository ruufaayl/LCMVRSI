from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from lcmvrsi.models.base import SequenceModel

# Label ignored by the loss (matches torch's cross_entropy default). Benchmarks set
# non-supervised positions to this value; trainers pass it as `ignore_index` so that only
# the positions a benchmark actually scores contribute to the loss.
IGNORE_INDEX = -100


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
