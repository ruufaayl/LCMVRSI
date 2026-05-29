import torch.nn as nn

from lcmvrsi.utils.profiling import count_parameters


def test_counts_linear_params():
    # Linear(10, 10): weight 10*10 + bias 10 = 110
    assert count_parameters(nn.Linear(10, 10)) == 110


def test_trainable_only_excludes_frozen():
    layer = nn.Linear(10, 10)
    for p in layer.parameters():
        p.requires_grad_(False)
    assert count_parameters(layer, trainable_only=True) == 0
    assert count_parameters(layer, trainable_only=False) == 110
