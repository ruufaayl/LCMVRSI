# LCMVRSI

**Long-Context Memory, Verifiable Reasoning, and Scalable Intelligence** — a research
scaffold for studying the **recall–memory tradeoff** in subquadratic sequence models.

> **Honesty discipline.** Every non-trivial claim in this repo and the paper is tagged
> **[PROVEN]**, **[EMPIRICAL]**, or **[CONJECTURE]**. No fabricated theorems, citations,
> or results. References enter `paper/refs.bib` only after verification.

## What this is
A runnable, tested comparison of sequence-model architectures (transformer, linear
attention, SSM/Mamba, RWKV, …) on synthetic associative-recall tasks (MQAR), plus an
honest attempt at a novel result: a lower bound linking recall capacity to recurrent
state size, and an entropy-gated sparse-memory mechanism. See
[`docs/superpowers/specs/2026-05-29-lcmvrsi-design.md`](docs/superpowers/specs/2026-05-29-lcmvrsi-design.md).

## Quickstart
```bash
# install uv: https://docs.astral.sh/uv/
uv sync --group dev      # create .venv and install deps + dev tools
uv run pytest            # run the test suite
uv run ruff check .      # lint
```
GPU note: `uv sync` installs the CPU build of PyTorch by default. For your CUDA GPU,
install the matching CUDA wheel per https://pytorch.org/get-started/locally/.

## Experiments
Tiny by default — every command below runs on CPU in minutes.
```bash
# Train one model on MQAR from a YAML config; writes a self-describing result JSON to results/
uv run python experiments/run.py -c configs/base.yaml

# Recall–memory sweep: held-out recall vs. number of key–value pairs, per architecture.
# Writes results/recall_sweep_summary.json (committed), per-run JSONs (results/runs/, gitignored),
# and the figure results/figures/recall_vs_pairs.png (committed).
uv run --extra viz python experiments/sweep_recall.py

# Complexity/memory table, generated from the live models (cross-checks docs/architecture-review):
uv run python experiments/complexity_table.py -o results/complexity_table.md

# Interactive dashboard over results/ (needs the viz extra):
uv run --extra viz streamlit run dashboard/app.py
```
Each result JSON records the config, the model's `complexity()`/`state_size`, the training
curve and throughput, held-out recall, and the software environment (incl. git commit) — so
every number is reproducible and traceable.

## Layout
- `src/lcmvrsi/` — package: `models/`, `benchmarks/`, `train/` (runner, sweep, LR schedule),
  `viz/`, `utils/`, `analysis.py` (complexity table)
- `configs/` — YAML experiment configs (tiny-by-default; scale-up provided later)
- `experiments/` — CLIs: `run.py` (single run), `sweep_recall.py` (recall sweep), `complexity_table.py`
- `dashboard/` — Streamlit app over `results/`
- `results/` — outputs: committed summary JSON + `figures/`; raw per-run JSONs in `runs/` (gitignored)
- `paper/` — LaTeX sources; PDF built by CI
- `docs/` — knowledge map, architecture review, problem formalization

## Status
Milestone **M2 (vertical slice)** — runnable model/benchmark/runner/sweep/dashboard, all tested.
Roadmap M0→M5 in the design spec.
