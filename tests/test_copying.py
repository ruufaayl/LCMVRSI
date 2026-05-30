import torch

from lcmvrsi.benchmarks.base import IGNORE_INDEX
from lcmvrsi.benchmarks.copying import COPY_TOKEN, Copying
from lcmvrsi.benchmarks.registry import get_benchmark, list_benchmarks
from lcmvrsi.models.base import SequenceModel


class _ConstantModel(SequenceModel):
    """Emits the same logits everywhere — cannot copy, so it sits near chance."""

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        b, t = input_ids.shape
        logits = torch.zeros(b, t, self.vocab_size)
        logits[..., 3] = 10.0  # always predict one fixed data token
        return logits

    @property
    def state_size(self) -> int:
        return 0

    def complexity(self, seq_len: int) -> dict[str, str]:
        return {"time": "O(1)", "memory": "O(1)"}


def test_copying_registered():
    assert "copying" in list_benchmarks()
    assert get_benchmark("copying") is Copying


def test_shapes_and_dtype():
    bench = Copying(vocab_size=32, copy_len=4, seq_len=20)
    x, y = bench.generate(n=5, seq_len=20, seed=0)
    assert x.shape == (5, 20)
    assert y.shape == (5, 20)
    assert x.dtype == torch.long
    assert y.dtype == torch.long


def test_only_reproduce_positions_are_supervised():
    bench = Copying(vocab_size=32, copy_len=4, seq_len=20)
    _, y = bench.generate(n=7, seq_len=20, seed=1)
    # exactly copy_len supervised targets per row, the rest IGNORE_INDEX
    assert (y != IGNORE_INDEX).sum(dim=1).unique().tolist() == [4]


def test_targets_reconstruct_the_copied_content():
    bench = Copying(vocab_size=32, copy_len=5, seq_len=20)
    x, y = bench.generate(n=6, seq_len=20, seed=2)
    L = 5
    # content sits at [0, L); the COPY delimiter at position L; targets at [L+1, 2L+1)
    assert (x[:, L] == COPY_TOKEN).all()
    assert torch.equal(y[:, L + 1 : 2 * L + 1], x[:, 0:L])


def test_too_short_raises():
    bench = Copying(vocab_size=32, copy_len=10, seq_len=15)  # needs 2*10+1=21 > 15
    try:
        bench.generate(n=1, seq_len=15, seed=0)
    except ValueError:
        return
    raise AssertionError("expected ValueError when seq_len < 2*copy_len + 1")


def test_determinism_and_resampling():
    bench = Copying(vocab_size=32, copy_len=4, seq_len=20)
    a, _ = bench.generate(3, 20, seed=5)
    b, _ = bench.generate(3, 20, seed=5)
    c, _ = bench.generate(3, 20, seed=6)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_blind_model_near_chance():
    bench = Copying(vocab_size=32, copy_len=4, seq_len=20)
    metrics = bench.evaluate(_ConstantModel(32), n=64, seq_len=20, seed=3)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["accuracy"] < 0.2  # ~1/30 by chance
    assert metrics["num_targets"] == 64 * 4
