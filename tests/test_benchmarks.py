import torch

from lcmvrsi.benchmarks.base import Benchmark
from lcmvrsi.benchmarks.registry import get_benchmark, list_benchmarks, register_benchmark
from lcmvrsi.models.base import SequenceModel


def test_abstract_cannot_instantiate():
    import pytest

    with pytest.raises(TypeError):
        Benchmark()


def test_concrete_benchmark_registers_generates_evaluates():
    @register_benchmark("echo")
    class Echo(Benchmark):
        def generate(self, n: int, seq_len: int, seed: int):
            g = torch.Generator().manual_seed(seed)
            x = torch.randint(0, 5, (n, seq_len), generator=g)
            return x, x.clone()

        def evaluate(self, model: SequenceModel, n: int, seq_len: int, seed: int):
            x, y = self.generate(n, seq_len, seed)
            logits = model(x)
            acc = (logits.argmax(-1) == y).float().mean().item()
            return {"accuracy": acc}

    assert "echo" in list_benchmarks()
    bench = get_benchmark("echo")()
    x, y = bench.generate(n=3, seq_len=4, seed=0)
    assert x.shape == (3, 4)
    assert torch.equal(x, y)
