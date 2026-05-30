from __future__ import annotations

import torch

from lcmvrsi.benchmarks.base import IGNORE_INDEX, Benchmark
from lcmvrsi.benchmarks.registry import register_benchmark
from lcmvrsi.models.base import SequenceModel


@register_benchmark("structured_recall")
class StructuredRecall(Benchmark):
    """Associative recall hidden in a *predictable* stream (the H2 achievability testbed).

    The sequence is a perfectly predictable repeating filler cycle of period ``cycle_len``, into
    which ``num_pairs`` novel key->value bindings are scattered (``[key, value]`` overwriting two
    consecutive filler slots). The trailing ``num_pairs`` positions re-present the keys (shuffled)
    as queries; targets are their values, IGNORE_INDEX elsewhere.

    Unlike MQAR's degenerate constant-PAD filler, the cyclic filler is non-trivially predictable,
    so after training only ``~3*num_pairs`` tokens are genuinely surprising. This is the regime in
    which a surprise-gated store can stay small (state ``~ Theta(H) << n``): the realized novelty
    is decoupled from sequence length. Fixed-rank recurrent states remain capacity-limited by the
    number of distinct bindings.
    """

    def __init__(
        self, vocab_size: int = 64, num_pairs: int = 8, cycle_len: int = 8, seq_len: int = 128
    ) -> None:
        if cycle_len < 2:
            raise ValueError("cycle_len must be >= 2")
        self.vocab_size = vocab_size
        self.num_pairs = num_pairs
        self.cycle_len = cycle_len
        self.seq_len = seq_len  # default only; generate() honours its own seq_len arg
        self.filler_lo = 1
        self.key_lo = 1 + cycle_len
        remaining = vocab_size - self.key_lo
        if remaining < 2 * num_pairs:
            raise ValueError("vocab_size too small for cycle_len + key/value ranges")
        self.n_keys = remaining // 2
        self.val_lo = self.key_lo + self.n_keys
        self.n_vals = vocab_size - self.val_lo
        if num_pairs > self.n_keys:
            raise ValueError(f"num_pairs={num_pairs} exceeds available keys={self.n_keys}")

    def generate(self, n: int, seq_len: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        d = self.num_pairs
        if 3 * d >= seq_len:
            raise ValueError(f"seq_len={seq_len} too short for num_pairs={d} (need > 3*num_pairs)")
        g = torch.Generator().manual_seed(int(seed))
        cycle = self.filler_lo + (torch.arange(seq_len) % self.cycle_len)
        inputs = cycle.unsqueeze(0).repeat(n, 1).contiguous()
        targets = torch.full((n, seq_len), IGNORE_INDEX, dtype=torch.long)
        qstart = seq_len - d
        even_slots = torch.arange(0, qstart - 1, 2)  # key at even slot, value at slot+1
        for row in range(n):
            keys = torch.randperm(self.n_keys, generator=g)[:d] + self.key_lo
            vals = torch.randint(0, self.n_vals, (d,), generator=g) + self.val_lo
            pos = even_slots[torch.randperm(len(even_slots), generator=g)[:d]]
            inputs[row, pos] = keys
            inputs[row, pos + 1] = vals
            order = torch.randperm(d, generator=g)
            inputs[row, qstart:] = keys[order]
            targets[row, qstart:] = vals[order]
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
            "recall_accuracy": float(accuracy),
            "num_targets": int(mask.sum().item()),
        }
