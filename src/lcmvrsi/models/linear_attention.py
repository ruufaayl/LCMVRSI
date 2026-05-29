from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from lcmvrsi.models.base import SequenceModel
from lcmvrsi.models.registry import register_model

_EPS = 1e-6


def _phi(x: torch.Tensor) -> torch.Tensor:
    """Feature map ensuring positivity (Katharopoulos et al., 2020): elu(x) + 1."""
    return F.elu(x) + 1.0


class _CausalLinearAttention(nn.Module):
    """Causal linear attention with a fixed-size recurrent state per head.

    Maintains, implicitly via prefix sums, S_t = sum_{i<=t} phi(k_i) v_i^T and
    z_t = sum_{i<=t} phi(k_i); the output is phi(q_t)^T S_t / (phi(q_t)^T z_t). The
    state size is O(d_phi * d_v) per head, independent of sequence length -- the
    rank-limited bottleneck behind the recall-memory tradeoff.
    """

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.n_heads, self.d_head)
        q, k, v = (z.transpose(1, 2) for z in qkv.unbind(dim=2))  # each (b, h, t, d_head)
        qf, kf = _phi(q), _phi(k)
        # prefix sums give the causal recurrent state at every position
        kv = kf.unsqueeze(-1) * v.unsqueeze(-2)  # (b, h, t, p, e) outer products
        s = kv.cumsum(dim=2)  # S_t
        zsum = kf.cumsum(dim=2)  # z_t  (b, h, t, p)
        num = torch.einsum("bhtp,bhtpe->bhte", qf, s)
        den = torch.einsum("bhtp,bhtp->bht", qf, zsum).clamp_min(_EPS)
        out = num / den.unsqueeze(-1)
        out = out.transpose(1, 2).reshape(b, t, c)
        return self.proj(out)


class _Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = _CausalLinearAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_ratio * d_model),
            nn.GELU(),
            nn.Linear(mlp_ratio * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


@register_model("linear_attention")
class LinearAttention(SequenceModel):
    """Tiny linear-attention baseline: fixed O(d^2) recurrent state, O(T) time."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_layers: int = 2,
        n_heads: int = 2,
        max_seq_len: int = 512,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList(
            [_Block(d_model, n_heads, mlp_ratio, dropout) for _ in range(n_layers)]
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
        # Per head: S (d_head x d_head) + z (d_head) floats, 4 bytes each (float32).
        elems = self.n_layers * self.n_heads * (self.d_head * self.d_head + self.d_head)
        return elems * 4

    def complexity(self, seq_len: int) -> dict[str, str]:
        return {"time": "O(T d^2)", "memory": "O(d^2)", "inference_state": "O(d^2)"}
