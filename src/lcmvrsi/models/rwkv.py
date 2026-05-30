from __future__ import annotations

import torch
import torch.nn as nn

from lcmvrsi.models.base import SequenceModel
from lcmvrsi.models.registry import register_model


class _WKV(nn.Module):
    """RWKV-style time-mixing: token-shift + a numerically-stable WKV recurrence.

    Per channel, with decay ``w = -exp(time_decay) < 0`` and current-token bonus ``u``::

        wkv_t = ( Σ_{i<t} e^{(t-1-i) w + k_i} v_i + e^{u + k_t} v_t )
                / ( Σ_{i<t} e^{(t-1-i) w + k_i}   + e^{u + k_t} )

    maintained by carrying running (numerator ``a``, denominator ``b``, max-exponent ``p``) with
    the standard max-shift for stability -- a fixed ``O(d)`` recurrent state (plus an ``O(d)``
    token-shift buffer), ``O(T d)`` time. Output is gated by ``sigmoid(receptance)``.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.time_decay = nn.Parameter(torch.zeros(d_model))  # w = -exp(time_decay)
        self.time_first = nn.Parameter(torch.zeros(d_model))  # u (current-token bonus)
        self.mix_k = nn.Parameter(torch.full((d_model,), 0.5))
        self.mix_v = nn.Parameter(torch.full((d_model,), 0.5))
        self.mix_r = nn.Parameter(torch.full((d_model,), 0.5))
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.receptance = nn.Linear(d_model, d_model, bias=False)
        self.output = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        x_prev = torch.cat([x.new_zeros(b, 1, d), x[:, :-1, :]], dim=1)  # token-shift
        k = self.key(x * self.mix_k + x_prev * (1 - self.mix_k))
        v = self.value(x * self.mix_v + x_prev * (1 - self.mix_v))
        r = torch.sigmoid(self.receptance(x * self.mix_r + x_prev * (1 - self.mix_r)))

        w = -torch.exp(self.time_decay)  # (d,), strictly negative log-decay
        u = self.time_first  # (d,)

        a = x.new_zeros(b, d)  # running numerator  Σ e^{..} v
        bb = x.new_zeros(b, d)  # running denominator Σ e^{..}
        p = x.new_full((b, d), -1e38)  # running max exponent (for stability)
        outputs = []
        for i in range(t):
            kt, vt = k[:, i, :], v[:, i, :]
            # output: fold in the current token with bonus u
            q = torch.maximum(p, u + kt)
            e1, e2 = torch.exp(p - q), torch.exp(u + kt - q)
            outputs.append((e1 * a + e2 * vt) / (e1 * bb + e2))
            # state update: decay the running sums, then add the current token
            q2 = torch.maximum(p + w, kt)
            e1, e2 = torch.exp(p + w - q2), torch.exp(kt - q2)
            a = e1 * a + e2 * vt
            bb = e1 * bb + e2
            p = q2
        wkv = torch.stack(outputs, dim=1)  # (b, t, d)
        return self.output(r * wkv)


class _Block(nn.Module):
    def __init__(self, d_model: int, mlp_ratio: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.mixer = _WKV(d_model)
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


@register_model("rwkv")
class RWKV(SequenceModel):
    """Tiny RWKV-style baseline: fixed O(d) WKV state per layer, O(T d) time."""

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
        self.blocks = nn.ModuleList([_Block(d_model, mlp_ratio, dropout) for _ in range(n_layers)])
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
        # Per layer the WKV recurrence carries (a, b, p) = 3 vectors of size d_model, float32.
        # (A separate O(d) token-shift buffer is also kept at inference but not counted here.)
        return self.n_layers * 3 * self.d_model * 4

    def complexity(self, seq_len: int) -> dict[str, str]:
        return {"time": "O(T d)", "memory": "O(d)", "inference_state": "O(d)"}
