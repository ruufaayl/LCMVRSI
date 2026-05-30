import torch

from lcmvrsi.benchmarks.base import IGNORE_INDEX
from lcmvrsi.benchmarks.needle import Needle
from lcmvrsi.benchmarks.registry import get_benchmark, list_benchmarks
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


def test_needle_registered():
    assert "needle" in list_benchmarks()
    assert get_benchmark("needle") is Needle


def test_shapes_and_dtype():
    bench = Needle(vocab_size=64, seq_len=48)
    x, y = bench.generate(n=5, seq_len=48, seed=0)
    assert x.shape == (5, 48)
    assert y.shape == (5, 48)
    assert x.dtype == torch.long and y.dtype == torch.long


def test_exactly_one_supervised_target_per_row():
    bench = Needle(vocab_size=64, seq_len=48)
    _, y = bench.generate(n=9, seq_len=48, seed=1)
    assert (y != IGNORE_INDEX).sum(dim=1).unique().tolist() == [1]


def test_target_is_the_value_following_the_planted_key():
    bench = Needle(vocab_size=64, seq_len=48)
    x, y = bench.generate(n=8, seq_len=48, seed=2)
    t = x.shape[1]
    for row in range(x.shape[0]):
        key = x[row, t - 1].item()
        first = (x[row, : t - 1] == key).nonzero().flatten()
        assert first.numel() == 1  # the key is unique in the haystack (planted once)
        p = first.item()
        assert y[row, t - 1].item() == x[row, p + 1].item()


def test_needle_position_varies_across_rows():
    bench = Needle(vocab_size=64, seq_len=48)
    x, _ = bench.generate(n=16, seq_len=48, seed=4)
    t = x.shape[1]
    positions = {(x[row, : t - 1] == x[row, t - 1]).nonzero().item() for row in range(16)}
    assert len(positions) > 1  # the needle is hidden at varying depths


def test_blind_model_near_chance():
    bench = Needle(vocab_size=64, seq_len=48)
    metrics = bench.evaluate(_ConstantModel(64), n=128, seq_len=48, seed=3)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["accuracy"] < 0.2
    assert metrics["num_targets"] == 128
