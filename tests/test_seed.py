import torch

from lcmvrsi.utils.seed import set_seed


def test_same_seed_reproduces_torch_draw():
    set_seed(123)
    a = torch.rand(8)
    set_seed(123)
    b = torch.rand(8)
    assert torch.equal(a, b)


def test_different_seeds_differ():
    set_seed(1)
    a = torch.rand(8)
    set_seed(2)
    b = torch.rand(8)
    assert not torch.equal(a, b)
