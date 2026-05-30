from __future__ import annotations

import torch

from lcmvrsi.benchmarks.base import IGNORE_INDEX, Benchmark
from lcmvrsi.benchmarks.registry import register_benchmark
from lcmvrsi.models.base import SequenceModel

PAD_TOKEN = 0


@register_benchmark("needle")
class Needle(Benchmark):
    """Needle-in-a-haystack long-context retrieval.

    The sequence is filled with distractor *value* tokens (the haystack). A single key->value
    "needle" is planted at a random depth ``p`` (``s[p]=key``, ``s[p+1]=value``); the query key
    is repeated at the last position, where the target is the planted value. Keys and values use
    disjoint id ranges and filler is drawn only from values, so the planted key is unique and
    retrieval is unambiguous -- the model must locate the needle at arbitrary depth and return
    the token that followed it.
    """

    def __init__(self, vocab_size: int = 64, seq_len: int = 128) -> None:
        if vocab_size < 4:
            raise ValueError("vocab_size must be >= 4 (need PAD, keys, values)")
        self.vocab_size = vocab_size
        self.seq_len = seq_len  # default only; generate() honours its own seq_len arg
        self.n_keys = (vocab_size - 1) // 2
        self.key_lo = 1
        self.val_lo = 1 + self.n_keys
        self.n_vals = vocab_size - self.val_lo

    def generate(self, n: int, seq_len: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len < 3:
            raise ValueError(f"seq_len={seq_len} too short (need >= 3)")
        g = torch.Generator().manual_seed(int(seed))
        # haystack: distractor value tokens only (disjoint from the key range)
        inputs = torch.randint(0, self.n_vals, (n, seq_len), generator=g) + self.val_lo
        targets = torch.full((n, seq_len), IGNORE_INDEX, dtype=torch.long)
        for row in range(n):
            key = int(torch.randint(0, self.n_keys, (1,), generator=g).item()) + self.key_lo
            value = int(torch.randint(0, self.n_vals, (1,), generator=g).item()) + self.val_lo
            p = int(torch.randint(0, seq_len - 2, (1,), generator=g).item())  # depth in [0, T-3]
            inputs[row, p] = key
            inputs[row, p + 1] = value
            inputs[row, seq_len - 1] = key  # query
            targets[row, seq_len - 1] = value
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
            "needle_accuracy": float(accuracy),
            "num_targets": int(mask.sum().item()),
        }
