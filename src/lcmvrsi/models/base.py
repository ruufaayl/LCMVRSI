from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class SequenceModel(nn.Module, ABC):
    """Common interface for sequence models compared in LCMVRSI.

    Subclasses must implement `forward` and self-report their memory/compute story via
    `state_size` and `complexity`, so models are comparable on the recall-memory frontier.
    """

    @abstractmethod
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Map token ids (B, T) to logits (B, T, vocab_size)."""
        raise NotImplementedError

    @property
    @abstractmethod
    def state_size(self) -> int:
        """Bytes of recurrent state per sequence at inference (0 if O(T) cache / not applicable)."""
        raise NotImplementedError

    @abstractmethod
    def complexity(self, seq_len: int) -> dict[str, str]:
        """Big-O annotations, e.g. {'time': 'O(T^2)', 'memory': 'O(T)'}."""
        raise NotImplementedError
