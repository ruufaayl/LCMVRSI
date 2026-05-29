from __future__ import annotations

import torch.nn as nn


def count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    """Count (trainable) parameters in a module."""
    params = module.parameters()
    if trainable_only:
        return sum(p.numel() for p in params if p.requires_grad)
    return sum(p.numel() for p in params)
