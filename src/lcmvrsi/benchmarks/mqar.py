from __future__ import annotations

import torch

from lcmvrsi.benchmarks.base import IGNORE_INDEX, Benchmark
from lcmvrsi.benchmarks.registry import register_benchmark
from lcmvrsi.models.base import SequenceModel


@register_benchmark("mqar")
class MQAR(Benchmark):
    """Multi-Query Associative Recall (Zoology-style synthetic task).

    Vocabulary layout (token 0 is reserved as filler / PAD)::

        keys   in [1, 1 + n_keys)
        values in [1 + n_keys, vocab_size)

    Each sequence writes ``num_pairs`` distinct key->value bindings at the front as
    ``[k1, v1, k2, v2, ...]``, fills the middle with PAD, and ends with the same keys
    in shuffled order as queries. Targets are ``IGNORE_INDEX`` everywhere except the
    trailing query positions, where the target is the value bound to that key.

    Bindings are resampled per sequence, so the answer cannot live in the weights --
    it must be retrieved from context. This is what makes the task measure *recall*.
    """

    def __init__(self, vocab_size: int = 64, num_pairs: int = 8, seq_len: int = 128) -> None:
        if vocab_size < 4:
            raise ValueError("vocab_size must be >= 4 (need room for PAD, keys, values)")
        self.vocab_size = vocab_size
        self.num_pairs = num_pairs
        self.seq_len = seq_len  # default only; generate() honours its own seq_len arg
        self.n_keys = (vocab_size - 1) // 2
        self.key_lo, self.key_hi = 1, 1 + self.n_keys
        self.val_lo, self.val_hi = 1 + self.n_keys, vocab_size
        if num_pairs > self.n_keys:
            raise ValueError(f"num_pairs={num_pairs} exceeds available keys={self.n_keys}")

    def generate(self, n: int, seq_len: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        d = self.num_pairs
        if 3 * d > seq_len:
            raise ValueError(f"seq_len={seq_len} too short for num_pairs={d} (need >= 3*num_pairs)")
        g = torch.Generator().manual_seed(int(seed))
        n_vals = self.val_hi - self.val_lo
        inputs = torch.zeros((n, seq_len), dtype=torch.long)
        targets = torch.full((n, seq_len), IGNORE_INDEX, dtype=torch.long)
        qpos = seq_len - d
        for row in range(n):
            keys = torch.randperm(self.n_keys, generator=g)[:d] + self.key_lo
            vals = torch.randint(0, n_vals, (d,), generator=g) + self.val_lo
            inputs[row, 0 : 2 * d : 2] = keys
            inputs[row, 1 : 2 * d : 2] = vals
            order = torch.randperm(d, generator=g)
            inputs[row, qpos:] = keys[order]
            targets[row, qpos:] = vals[order]
        return inputs, targets

    def evaluate(
        self, model: SequenceModel, n: int, seq_len: int, seed: int
    ) -> dict[str, float]:
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
        return {"recall_accuracy": float(accuracy), "num_queries": int(mask.sum().item())}
