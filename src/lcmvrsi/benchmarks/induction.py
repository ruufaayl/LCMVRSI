from __future__ import annotations

import torch

from lcmvrsi.benchmarks.base import IGNORE_INDEX, Benchmark
from lcmvrsi.benchmarks.registry import register_benchmark
from lcmvrsi.models.base import SequenceModel


@register_benchmark("induction")
class Induction(Benchmark):
    """Induction-head probe (Olsson et al.): complete ``... A B ... A -> B``.

    Each sequence is random data tokens; the final (query) token is forced to repeat an earlier
    token so a previous occurrence always exists. The single supervised target, at the last
    position, is the successor of the query token's **most recent** previous occurrence -- the
    classic induction rule. Solving it requires matching the query back to context and emitting
    the following token; everything else is IGNORE_INDEX.
    """

    def __init__(self, vocab_size: int = 64, seq_len: int = 64) -> None:
        if vocab_size < 3:
            raise ValueError("vocab_size must be >= 3")
        self.vocab_size = vocab_size
        self.seq_len = seq_len  # default only; generate() honours its own seq_len arg
        self.data_lo = 1
        self.n_data = vocab_size - 1

    def generate(self, n: int, seq_len: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len < 3:
            raise ValueError(f"seq_len={seq_len} too short (need >= 3)")
        g = torch.Generator().manual_seed(int(seed))
        inputs = torch.randint(0, self.n_data, (n, seq_len), generator=g) + self.data_lo
        targets = torch.full((n, seq_len), IGNORE_INDEX, dtype=torch.long)
        for row in range(n):
            # force the query token to repeat an earlier token so induction is well-defined
            k = int(torch.randint(0, seq_len - 1, (1,), generator=g).item())
            inputs[row, seq_len - 1] = inputs[row, k]
            query = inputs[row, seq_len - 1]
            prev = int((inputs[row, : seq_len - 1] == query).nonzero().max().item())
            targets[row, seq_len - 1] = inputs[row, prev + 1]
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
            "induction_accuracy": float(accuracy),
            "num_targets": int(mask.sum().item()),
        }
