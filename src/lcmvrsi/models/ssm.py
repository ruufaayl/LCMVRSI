from __future__ import annotations

import torch
import torch.nn as nn

from lcmvrsi.models.base import SequenceModel
from lcmvrsi.models.registry import register_model


class _DiagonalSSM(nn.Module):
    """Simplified diagonal state-space mixer (S4D-style), content-independent A/B/C.

    Per channel, a length-``N`` diagonal recurrence::

        h_t = a ⊙ h_{t-1} + B · u_t ,    y_t = <C, h_t> + D · u_t

    with ``a = exp(-exp(A_log)) ∈ (0,1)^N`` for a stable, multi-timescale decay. Computed by an
    explicit causal scan over ``t``: ``O(T·d·N)`` time and a fixed ``O(d·N)`` recurrent state,
    independent of ``T``. This is a legitimate *non-selective* (LTI) SSM baseline -- not the
    full input-dependent selective Mamba kernel, and named accordingly.
    """

    def __init__(self, d_model: int, state_dim: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim
        self.in_proj = nn.Linear(d_model, d_model)
        # A_log init so a = exp(-exp(A_log)) spans a range of timescales (S4D-style).
        a_init = torch.log(torch.arange(1, state_dim + 1, dtype=torch.float32))  # (N,)
        self.A_log = nn.Parameter(a_init.expand(d_model, state_dim).clone())  # (d, N)
        self.B = nn.Parameter(torch.ones(d_model, state_dim))
        self.C = nn.Parameter(torch.randn(d_model, state_dim) * state_dim**-0.5)
        self.D = nn.Parameter(torch.ones(d_model))
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        u = self.in_proj(x)  # (b, t, d)
        a = torch.exp(-torch.exp(self.A_log))  # (d, N) in (0, 1)
        h = x.new_zeros(b, self.d_model, self.state_dim)  # fixed recurrent state
        outputs = []
        for i in range(t):
            ui = u[:, i, :]  # (b, d)
            h = a * h + ui.unsqueeze(-1) * self.B  # (b, d, N)
            yi = (h * self.C).sum(dim=-1) + self.D * ui  # (b, d)
            outputs.append(yi)
        y = torch.stack(outputs, dim=1)  # (b, t, d)
        return self.out_proj(y)


class _Block(nn.Module):
    def __init__(self, d_model: int, state_dim: int, mlp_ratio: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.mixer = _DiagonalSSM(d_model, state_dim)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_ratio * d_model),
            nn.GELU(),
            nn.Linear(mlp_ratio * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mixer(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


@register_model("ssm")
class SSM(SequenceModel):
    """Tiny simplified diagonal SSM (S4D-style) baseline: fixed O(d*N) state, O(T d N) time."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_layers: int = 2,
        state_dim: int = 16,
        max_seq_len: int = 512,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.n_layers = n_layers
        self.d_model = d_model
        self.state_dim = state_dim
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList(
            [_Block(d_model, state_dim, mlp_ratio, dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        _, t = input_ids.shape
        if t > self.max_seq_len:
            raise ValueError(f"seq_len {t} exceeds max_seq_len {self.max_seq_len}")
        pos = torch.arange(t, device=input_ids.device)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))

    @property
    def state_size(self) -> int:
        # Per layer: a diagonal state h of shape (d_model, state_dim) floats, 4 bytes each.
        return self.n_layers * self.d_model * self.state_dim * 4

    def complexity(self, seq_len: int) -> dict[str, str]:
        return {"time": "O(T d N)", "memory": "O(d N)", "inference_state": "O(d N)"}
