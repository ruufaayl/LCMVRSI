from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Pure, dependency-light loaders for experiment results. No streamlit/pandas import here, so
# this module (and its tests) run in the core environment; the streamlit app imports these and
# adds the UI layer on top.


def load_summary(path: str | Path) -> dict[str, Any] | None:
    """Load a sweep summary JSON, or return None if the file does not exist."""
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def load_runs(runs_dir: str | Path) -> list[dict[str, Any]]:
    """Load every per-run result JSON in ``runs_dir`` (empty list if the dir is absent)."""
    d = Path(runs_dir)
    if not d.exists():
        return []
    runs: list[dict[str, Any]] = []
    for f in sorted(d.glob("*.json")):
        try:
            runs.append(json.loads(f.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return runs


def summary_to_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the flat per-run rows from a summary (one dict per (model, num_pairs) cell)."""
    return list(summary.get("points", []))


def pivot_recall(summary: dict[str, Any]) -> dict[str, dict[int, float]]:
    """Pivot summary points into ``{model: {num_pairs: recall_accuracy}}`` for plotting."""
    table: dict[str, dict[int, float]] = {}
    for pt in summary.get("points", []):
        table.setdefault(pt["model"], {})[pt["num_pairs"]] = pt["recall_accuracy"]
    return table
