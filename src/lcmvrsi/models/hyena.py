from __future__ import annotations

import torch
import torch.nn as nn

from lcmvrsi.models.base import SequenceModel
from lcmvrsi.models.registry import register_model


def _causal_conv_fft(v: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Causal depthwise long convolution via FFT.

    ``v``: (b, d, T) signal, ``h``: (d, T) per-channel filter indexed by lag (h[:, k] weights
    the token k steps in the past). Returns (b, d, T) with ``y[t] = Σ_{k=0}^{t} h[k] v[t-k]`` --
    strictly causal, computed as a linear convolution truncated to the first T samples in
    ``O(T log T)`` rather than the ``O(T^2)`` of a dense filter.
    """
    t = v.shape[-1]
    n = 2 * t  # zero-pad to linear- (not circular-) convolution length
    vf = torch.fft.rfft(v, n=n, dim=-1)
    hf = torch.fft.rfft(h, n=n, dim=-1).unsqueeze(0)  # (1, d, F)
    return torch.fft.irfft(vf * hf, n=n, dim=-1)[..., :t]


class _HyenaMixer(nn.Module):
    """Hyena-style gated long convolution (order-2) with an explicit decaying filter.

    Projects the input to a gate ``u`` and a signal ``v``; convolves ``v`` with a learned
    per-channel long filter (length up to ``max_seq_len``, damped by a fixed exponential decay
    window for stability) via FFT; then forms the multiplicative interaction ``u ⊙ conv(v)``.
    Subquadratic (``O(T log T)``) and content-aware through the gate. This uses an *explicit*
    learned filter rather than Hyena's implicit (FFN-of-positions) parametrization, and is named
    accordingly.
    """

    def __init__(self, d_model: int, max_seq_len: int, filter_decay: float = 0.02) -> None:
        super().__init__()
        self.in_proj = nn.Linear(d_model, 2 * d_model)
        self.filter = nn.Parameter(torch.randn(d_model, max_seq_len) * max_seq_len**-0.5)
        decay = torch.exp(-filter_decay * torch.arange(max_seq_len, dtype=torch.float32))
        self.register_buffer("decay", decay)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, t, _ = x.shape
        u, v = self.in_proj(x).chunk(2, dim=-1)  # each (b, t, d)
        h = self.filter[:, :t] * self.decay[:t]  # (d, t) damped long filter
        y = _causal_conv_fft(v.transpose(1, 2), h).transpose(1, 2)  # (b, t, d)
        return self.out_proj(u * y)  # multiplicative gating × long conv


class _Block(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int, mlp_ratio: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.mixer = _HyenaMixer(d_model, max_seq_len)
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


@register_model("hyena")
class Hyena(SequenceModel):
    """Tiny Hyena-style baseline: gated FFT long convolution, O(T log T) time, no fixed state."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_layers: int = 2,
        max_seq_len: int = 512,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.n_layers = n_layers
        self.d_model = d_model
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList(
            [_Block(d_model, max_seq_len, mlp_ratio, dropout) for _ in range(n_layers)]
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
        # A global convolution keeps no fixed recurrent state; autoregressive inference needs
        # the full O(T) history (like attention's cache), so there is no fixed-size bottleneck.
        return 0

    def complexity(self, seq_len: int) -> dict[str, str]:
        return {"time": "O(T log T d)", "memory": "O(T d)", "inference_cache": "O(T d)"}
