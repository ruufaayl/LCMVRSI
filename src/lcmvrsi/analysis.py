from __future__ import annotations

from typing import Any

from lcmvrsi.models.registry import list_models
from lcmvrsi.train.runner import build_model
from lcmvrsi.utils.config import ModelConfig
from lcmvrsi.utils.profiling import count_parameters

# Default instantiation knobs for the symbolic table. complexity() returns Big-O strings and
# state_size is independent of sequence length, so these concrete numbers only affect param_count.
_DEFAULT_PARAMS = {"d_model": 64, "n_layers": 2, "n_heads": 2}


def complexity_table(
    model_names: list[str] | None = None,
    *,
    seq_len: int = 512,
    vocab_size: int = 64,
    params: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build a complexity/memory row per registered model, read from the live objects.

    Generating the table from instantiated models (rather than hand-maintained text) means it
    cannot silently drift from the implementation: ``time``/``memory``/``inference`` come from
    each model's ``complexity()``, and ``state_size_bytes`` from its ``state_size``. Cross-check
    against the hand-derived table in ``docs/architecture-review`` (§5).
    """
    if model_names is None:
        model_names = list_models()
    params = params or dict(_DEFAULT_PARAMS)

    rows: list[dict[str, Any]] = []
    for name in model_names:
        cfg = ModelConfig(name=name, params=params)
        model = build_model(cfg, vocab_size=vocab_size, seq_len=seq_len)
        c = model.complexity(seq_len)
        state_bytes = int(model.state_size)
        rows.append(
            {
                "model": name,
                "time": c.get("time", "?"),
                "memory": c.get("memory", "?"),
                # transformers report a growing KV cache; recurrent models a fixed state
                "inference": c.get("inference_cache") or c.get("inference_state") or "?",
                "state_kind": "growing KV cache" if state_bytes == 0 else "fixed recurrent state",
                "state_size_bytes": state_bytes,
                "param_count": int(count_parameters(model, trainable_only=False)),
            }
        )
    return rows


def render_markdown_table(rows: list[dict[str, Any]]) -> str:
    """Render complexity rows as a GitHub-flavored Markdown table."""
    header = (
        "| Model | Time | Memory | Inference/step | State kind | Fixed state (bytes) | Params |"
    )
    sep = "|---|---|---|---|---|---|---|"
    lines = [header, sep]
    for r in rows:
        lines.append(
            f"| {r['model']} | `{r['time']}` | `{r['memory']}` | `{r['inference']}` "
            f"| {r['state_kind']} | {r['state_size_bytes']} | {r['param_count']} |"
        )
    return "\n".join(lines)
