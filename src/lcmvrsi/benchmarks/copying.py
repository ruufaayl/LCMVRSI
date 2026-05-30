from __future__ import annotations

import torch

from lcmvrsi.benchmarks.base import IGNORE_INDEX, Benchmark
from lcmvrsi.benchmarks.registry import register_benchmark
from lcmvrsi.models.base import SequenceModel

PAD_TOKEN = 0
COPY_TOKEN = 1


@register_benchmark("copying")
class Copying(Benchmark):
    """Sequence copying (Jelassi et al.-style): reproduce a random string from memory.

    Layout per sequence (``L = copy_len``)::

        [ c_0 .. c_{L-1} ,  COPY ,  PAD .. PAD  (L slots) ,  PAD .. ]
          content (≥2)      =1     reproduce region          filler

    Targets are ``IGNORE_INDEX`` everywhere except the reproduce region, where target ``j`` is
    ``c_j`` -- the model must emit the content in order, indexing by position into whatever it
    stored. Content is resampled per sequence (random, no internal structure), so the answer
    must be retrieved from context, and a fixed-size recurrent state must hold all ``L`` tokens.
    """

    def __init__(self, vocab_size: int = 64, copy_len: int = 16, seq_len: int = 128) -> None:
        if vocab_size < 4:
            raise ValueError("vocab_size must be >= 4 (need PAD, COPY, and data tokens)")
        if copy_len < 1:
            raise ValueError("copy_len must be >= 1")
        self.vocab_size = vocab_size
        self.copy_len = copy_len
        self.seq_len = seq_len  # default only; generate() honours its own seq_len arg
        self.data_lo = 2
        self.n_data = vocab_size - 2

    def generate(self, n: int, seq_len: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        length = self.copy_len
        if 2 * length + 1 > seq_len:
            raise ValueError(
                f"seq_len={seq_len} too short for copy_len={length} (need >= 2*copy_len + 1)"
            )
        g = torch.Generator().manual_seed(int(seed))
        inputs = torch.full((n, seq_len), PAD_TOKEN, dtype=torch.long)
        targets = torch.full((n, seq_len), IGNORE_INDEX, dtype=torch.long)
        content = torch.randint(0, self.n_data, (n, length), generator=g) + self.data_lo
        inputs[:, 0:length] = content
        inputs[:, length] = COPY_TOKEN
        targets[:, length + 1 : 2 * length + 1] = content
        return inputs, targets

    def evaluate(self, model: SequenceModel, n: int, seq_len: int, seed: int) -> dict[str, float]:
        x, y = self.generate(n, seq_len, seed)
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        was_training = model.training
        model.eval()
        with torch.no_grad():
            logits = model(x.to(device))
        if was_training:
            model.train()
        preds = logits.argmax(dim=-1).cpu()
        mask = y != IGNORE_INDEX
        accuracy = (preds[mask] == y[mask]).float().mean().item()
        return {
            "accuracy": float(accuracy),
            "copy_accuracy": float(accuracy),
            "num_targets": int(mask.sum().item()),
        }
