"""Streamlit dashboard for LCMVRSI experiment results.

Run::

    uv run --extra viz streamlit run dashboard/app.py

Reads the committed sweep summary (``results/recall_sweep_summary.json``) and any raw per-run
JSONs (``results/runs/``). All data loading lives in ``lcmvrsi.viz.dashboard_data`` so the core
package and CI never depend on streamlit; this script is the only streamlit entry point.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from lcmvrsi.viz.dashboard_data import load_runs, load_summary, summary_to_rows

RESULTS = Path("results")

st.set_page_config(page_title="LCMVRSI — recall-memory frontier", layout="wide")
st.title("LCMVRSI — recall-memory frontier")
st.caption(
    "Synthetic MQAR associative-recall results. Transformers keep no fixed recurrent state "
    "(an O(T) KV cache); linear attention keeps a fixed O(d^2) state. This dashboard shows how "
    "held-out recall changes with recall difficulty for each architecture."
)

summary = load_summary(RESULTS / "recall_sweep_summary.json")

if summary is None:
    st.warning(
        "No sweep summary found at `results/recall_sweep_summary.json`.\n\n"
        "Generate it with:\n\n"
        "```\nuv run --extra viz python experiments/sweep_recall.py\n```"
    )
    st.stop()

rows = summary_to_rows(summary)
df = pd.DataFrame(rows)

left, right = st.columns([2, 1])

with left:
    st.subheader("Held-out recall vs. number of key-value pairs")
    if {"num_pairs", "model", "recall_accuracy"} <= set(df.columns):
        chart_df = df.pivot(index="num_pairs", columns="model", values="recall_accuracy")
        st.line_chart(chart_df)
    fig_path = RESULTS / "figures" / "recall_vs_pairs.png"
    if fig_path.exists():
        st.image(str(fig_path), caption="Committed figure (results/figures/recall_vs_pairs.png)")

with right:
    st.subheader("Fixed recurrent state (bytes)")
    if {"model", "state_size_bytes"} <= set(df.columns):
        state_df = (
            df[["model", "state_size_bytes"]].drop_duplicates().set_index("model")
        )
        st.bar_chart(state_df)
    st.subheader("Setup")
    st.json(summary.get("setup", {}))

st.subheader("Per-run results")
st.dataframe(df, use_container_width=True)

runs = load_runs(RESULTS / "runs")
if runs:
    st.caption(f"{len(runs)} raw per-run JSON file(s) found under results/runs/.")

st.header("Recall–memory frontier (across models)")
frontier_files = sorted(RESULTS.glob("frontier_*.json"))
if not frontier_files:
    st.info(
        "No frontier results yet. Generate with:\n\n"
        "```\nuv run --extra viz python experiments/frontier.py --benchmark mqar\n```"
    )
for fp in frontier_files:
    fsummary = load_summary(fp)
    if fsummary is None:
        continue
    task = fsummary.get("setup", {}).get("benchmark", fp.stem)
    st.subheader(f"Task: {task}")
    fig = RESULTS / "figures" / f"frontier_{task}.png"
    if fig.exists():
        st.image(str(fig), caption=f"results/figures/frontier_{task}.png")
    st.dataframe(pd.DataFrame(fsummary.get("points", [])), use_container_width=True)
