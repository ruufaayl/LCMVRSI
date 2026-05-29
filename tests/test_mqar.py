import torch
import torch.nn as nn

from lcmvrsi.benchmarks.mqar import IGNORE_INDEX, MQAR
from lcmvrsi.benchmarks.registry import get_benchmark, list_benchmarks
from lcmvrsi.models.base import SequenceModel


def test_mqar_registered():
    assert "mqar" in list_benchmarks()
    assert get_benchmark("mqar") is MQAR


def test_shapes_and_dtype():
    bench = MQAR(vocab_size=64, num_pairs=8)
    x, y = bench.generate(n=5, seq_len=128, seed=0)
    assert x.shape == (5, 128)
    assert y.shape == (5, 128)
    assert x.dtype == torch.long
    assert y.dtype == torch.long


def test_only_query_positions_supervised():
    d = 8
    bench = MQAR(vocab_size=64, num_pairs=d)
    x, y = bench.generate(n=4, seq_len=128, seed=1)
    mask = y != IGNORE_INDEX
    assert torch.all(mask.sum(dim=1) == d)  # exactly d supervised positions per row
    assert torch.all(mask[:, -d:])  # which are the trailing d positions
    assert not mask[:, :-d].any()


def test_targets_reconstruct_bindings():
    # The heart of recall: the target at a query position must equal the value
    # bound to that key earlier in the SAME sequence.
    d, vocab, t = 6, 64, 96
    bench = MQAR(vocab_size=vocab, num_pairs=d)
    x, y = bench.generate(n=10, seq_len=t, seed=2)
    for row in range(x.shape[0]):
        binding = {x[row, 2 * i].item(): x[row, 2 * i + 1].item() for i in range(d)}
        for p in range(t - d, t):
            key = x[row, p].item()
            assert key in binding
            assert y[row, p].item() == binding[key]


def test_keys_and_values_in_disjoint_ranges():
    d, vocab = 8, 64
    bench = MQAR(vocab_size=vocab, num_pairs=d)
    x, y = bench.generate(n=4, seq_len=128, seed=3)
    n_keys = (vocab - 1) // 2
    q = x[:, -d:]
    assert torch.all((q >= 1) & (q < 1 + n_keys))  # query inputs are keys
    vals = y[y != IGNORE_INDEX]
    assert torch.all((vals >= 1 + n_keys) & (vals < vocab))  # targets are values


def test_determinism_and_resampling():
    bench = MQAR(vocab_size=64, num_pairs=8)
    x0, y0 = bench.generate(n=4, seq_len=128, seed=7)
    x1, y1 = bench.generate(n=4, seq_len=128, seed=7)
    x2, _ = bench.generate(n=4, seq_len=128, seed=8)
    assert torch.equal(x0, x1) and torch.equal(y0, y1)  # same seed -> identical
    assert not torch.equal(x0, x2)  # bindings resampled across seeds


def test_too_short_sequence_raises():
    bench = MQAR(vocab_size=64, num_pairs=8)
    try:
        bench.generate(n=1, seq_len=20, seed=0)  # 3*8 = 24 > 20
    except ValueError:
        return
    raise AssertionError("expected ValueError for seq_len < 3*num_pairs")


def test_blind_model_stuck_at_chance():
    # A context-free model that always predicts one fixed value cannot beat chance,
    # proving the benchmark's targets genuinely require recall from context.
    d, vocab = 8, 64

    class Blind(SequenceModel):
        def __init__(self, vocab_size: int, const_token: int) -> None:
            super().__init__()
            self.vocab_size = vocab_size
            self.const = const_token
            self._p = nn.Parameter(torch.zeros(1))  # keep .parameters() non-empty

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            b, t = input_ids.shape
            logits = torch.full((b, t, self.vocab_size), -1e9)
            logits[..., self.const] = 1e9
            return logits

        @property
        def state_size(self) -> int:
            return 0

        def complexity(self, seq_len: int) -> dict[str, str]:
            return {"time": "O(T)", "memory": "O(1)"}

    bench = MQAR(vocab_size=vocab, num_pairs=d)
    model = Blind(vocab, const_token=1 + (vocab - 1) // 2)
    metrics = bench.evaluate(model, n=200, seq_len=96, seed=11)
    assert metrics["num_queries"] == 200 * d
    assert metrics["recall_accuracy"] < 0.2  # ~1/32 chance, generous slack
