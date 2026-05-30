import torch

from lcmvrsi.benchmarks.base import IGNORE_INDEX
from lcmvrsi.benchmarks.induction import Induction
from lcmvrsi.benchmarks.registry import get_benchmark, list_benchmarks
from lcmvrsi.models.base import SequenceModel


class _ConstantModel(SequenceModel):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        b, t = input_ids.shape
        logits = torch.zeros(b, t, self.vocab_size)
        logits[..., 1] = 10.0
        return logits

    @property
    def state_size(self) -> int:
        return 0

    def complexity(self, seq_len: int) -> dict[str, str]:
        return {"time": "O(1)", "memory": "O(1)"}


def test_induction_registered():
    assert "induction" in list_benchmarks()
    assert get_benchmark("induction") is Induction


def test_shapes_and_dtype():
    bench = Induction(vocab_size=32, seq_len=24)
    x, y = bench.generate(n=5, seq_len=24, seed=0)
    assert x.shape == (5, 24)
    assert y.shape == (5, 24)
    assert x.dtype == torch.long and y.dtype == torch.long


def test_exactly_one_supervised_query_per_row():
    bench = Induction(vocab_size=32, seq_len=24)
    _, y = bench.generate(n=9, seq_len=24, seed=1)
    assert (y != IGNORE_INDEX).sum(dim=1).unique().tolist() == [1]


def test_target_is_successor_of_previous_occurrence():
    bench = Induction(vocab_size=32, seq_len=24)
    x, y = bench.generate(n=8, seq_len=24, seed=2)
    t = x.shape[1]
    for row in range(x.shape[0]):
        query = x[row, t - 1].item()
        earlier = (x[row, : t - 1] == query).nonzero().flatten()
        assert earlier.numel() >= 1  # the query token always has a prior occurrence
        prev = earlier.max().item()
        assert y[row, t - 1].item() == x[row, prev + 1].item()


def test_determinism_and_resampling():
    bench = Induction(vocab_size=32, seq_len=24)
    a, _ = bench.generate(4, 24, seed=7)
    b, _ = bench.generate(4, 24, seed=7)
    c, _ = bench.generate(4, 24, seed=8)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_blind_model_near_chance():
    bench = Induction(vocab_size=32, seq_len=24)
    metrics = bench.evaluate(_ConstantModel(32), n=128, seq_len=24, seed=3)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["accuracy"] < 0.2
    assert metrics["num_targets"] == 128
