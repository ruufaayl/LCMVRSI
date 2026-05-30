import torch

from lcmvrsi.benchmarks.base import IGNORE_INDEX
from lcmvrsi.benchmarks.registry import get_benchmark, list_benchmarks
from lcmvrsi.benchmarks.structured_recall import StructuredRecall
from lcmvrsi.models.base import SequenceModel


class _ConstantModel(SequenceModel):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        b, t = input_ids.shape
        logits = torch.zeros(b, t, self.vocab_size)
        logits[..., self.vocab_size - 1] = 10.0
        return logits

    @property
    def state_size(self) -> int:
        return 0

    def complexity(self, seq_len: int) -> dict[str, str]:
        return {"time": "O(1)", "memory": "O(1)"}


def test_structured_recall_registered():
    assert "structured_recall" in list_benchmarks()
    assert get_benchmark("structured_recall") is StructuredRecall


def test_shapes_and_dtype():
    bench = StructuredRecall(vocab_size=64, num_pairs=4, cycle_len=8, seq_len=64)
    x, y = bench.generate(n=5, seq_len=64, seed=0)
    assert x.shape == (5, 64)
    assert y.shape == (5, 64)
    assert x.dtype == torch.long and y.dtype == torch.long


def test_only_query_positions_supervised():
    bench = StructuredRecall(vocab_size=64, num_pairs=4, cycle_len=8, seq_len=64)
    _, y = bench.generate(n=7, seq_len=64, seed=1)
    assert (y != IGNORE_INDEX).sum(dim=1).unique().tolist() == [4]


def test_filler_is_a_predictable_cycle():
    bench = StructuredRecall(vocab_size=64, num_pairs=4, cycle_len=8, seq_len=64)
    x, _ = bench.generate(n=8, seq_len=64, seed=2)
    cycle = 1 + (torch.arange(64) % 8)  # filler_lo=1, period 8
    matches = (x == cycle[None, :]).float().mean().item()
    # only ~3*num_pairs of 64 positions are overwritten (bindings + queries); the rest are cycle
    assert matches > (64 - 3 * 4 - 1) / 64


def test_targets_reconstruct_bindings():
    bench = StructuredRecall(vocab_size=64, num_pairs=4, cycle_len=8, seq_len=64)
    x, y = bench.generate(n=6, seq_len=64, seed=3)
    t = x.shape[1]
    qstart = t - 4
    for row in range(x.shape[0]):
        for q in range(qstart, t):
            key = x[row, q].item()
            # the binding was planted earlier as [key, value] at consecutive positions
            hit = [(x[row, p + 1].item()) for p in range(qstart - 1) if x[row, p].item() == key]
            assert y[row, q].item() in hit


def test_determinism_and_resampling():
    bench = StructuredRecall(vocab_size=64, num_pairs=4, cycle_len=8, seq_len=64)
    a, _ = bench.generate(3, 64, seed=5)
    b, _ = bench.generate(3, 64, seed=5)
    c, _ = bench.generate(3, 64, seed=6)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_blind_model_near_chance():
    bench = StructuredRecall(vocab_size=64, num_pairs=4, cycle_len=8, seq_len=64)
    metrics = bench.evaluate(_ConstantModel(64), n=64, seq_len=64, seed=3)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["accuracy"] < 0.2
    assert metrics["num_targets"] == 64 * 4
