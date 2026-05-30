from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from lcmvrsi.models.base import SequenceModel
from lcmvrsi.models.linear_attention import _Block as _LinearBlock
from lcmvrsi.models.registry import register_model

_EPS = 1e-6


class _SurpriseGatedMemory(nn.Module):
    """External key->value store with writes gated by predictive surprise (H2 mechanism).

    Given backbone states ``h`` and a per-position surprise ``s_t`` (= NLL of the observed
    token), the write gate is ``g_t = sigmoid(alpha * (s_t - tau))`` with learnable ``tau, alpha``.
    The read at position ``t`` is causal attention over entries ``j <= t`` with the write-gate
    folded into the weights, so low-gate (unsurprising) entries are effectively absent. Soft and
    differentiable for training; at inference a hard write is ``s_t > tau`` and the realized store
    size (number of writes) tracks the stream's novelty rather than its length.
    """

    def __init__(self, d_model: int, gate_sharpness: float = 5.0, init_threshold: float = 2.0):
        super().__init__()
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.scale = d_model**-0.5
        self.alpha = nn.Parameter(torch.tensor(float(gate_sharpness)))
        self.tau = nn.Parameter(torch.tensor(float(init_threshold)))

    def forward(self, h: torch.Tensor, surprise: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, t, _ = h.shape
        gate = torch.sigmoid(self.alpha * (surprise - self.tau))  # (b, t) in (0, 1)
        q, k, v = self.qkv(h).chunk(3, dim=-1)  # each (b, t, d)
        scores = torch.einsum("btd,bsd->bts", q, k) * self.scale  # (b, t, s)
        causal = torch.ones(t, t, dtype=torch.bool, device=h.device).tril()  # s <= t
        scores = scores.masked_fill(~causal, float("-inf"))
        # fold the write gate into the read weights: unwritten entries contribute ~0
        scores = scores + torch.log(gate + _EPS)[:, None, :]
        attn = torch.softmax(scores, dim=-1)
        read = torch.einsum("bts,bsd->btd", attn, v)
        return self.proj(read), gate


@register_model("sgsm")
class SGSM(SequenceModel):
    """Surprise-Gated Sparse Memory (H2): linear-attention backbone + surprise-gated KV store.

    The store is the model's growing state; on structured (low-entropy) inputs few tokens are
    surprising, so realized state stays small, while on uniform inputs the gate fires everywhere
    and the module degrades gracefully to attention. The achievability claim (recall at state
    ~ Theta(H)) is a [CONJECTURE] tested empirically -- see docs/mechanism.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_layers: int = 2,
        n_heads: int = 2,
        max_seq_len: int = 512,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        gate_sharpness: float = 5.0,
        init_threshold: float = 2.0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_head = d_model // n_heads
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList(
            [_LinearBlock(d_model, n_heads, mlp_ratio, dropout) for _ in range(n_layers)]
        )
        self.ln_h = nn.LayerNorm(d_model)
        self.memory = _SurpriseGatedMemory(d_model, gate_sharpness, init_threshold)
        self.ln_out = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.last_gate: torch.Tensor | None = None  # realized write gate from the last forward

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        b, t = input_ids.shape
        if t > self.max_seq_len:
            raise ValueError(f"seq_len {t} exceeds max_seq_len {self.max_seq_len}")
        pos = torch.arange(t, device=input_ids.device)
        h = self.tok_emb(input_ids) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            h = block(h)

        # predictive surprise of the observed token: s_t = -log p(x_t | x_{<t}), s_0 = 0
        logp = F.log_softmax(self.head(self.ln_h(h)), dim=-1)
        surprise = h.new_zeros(b, t)
        if t > 1:
            picked = logp[:, :-1, :].gather(-1, input_ids[:, 1:, None]).squeeze(-1)
            surprise = torch.cat([surprise[:, :1], -picked], dim=1)

        read, gate = self.memory(h, surprise)
        self.last_gate = gate.detach()
        out = h + read
        return self.head(self.ln_out(out))

    @property
    def state_size(self) -> int:
        # Backbone: fixed linear-attention state. Store: worst case all T tokens written, each a
        # (key, value) pair of d_model floats. Realized store is << this on structured inputs.
        backbone = self.n_layers * self.n_heads * (self.d_head * self.d_head + self.d_head) * 4
        store_worst_case = self.max_seq_len * 2 * self.d_model * 4
        return backbone + store_worst_case

    def complexity(self, seq_len: int) -> dict[str, str]:
        return {
            "time": "O(T^2 d) train / O(T m d) infer",
            "memory": "O(T^2 + d^2)",
            "inference_state": "O(m d), m = #writes <= T (grows with surprise)",
        }
