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

## Layout
- `src/lcmvrsi/` — package: `models/`, `benchmarks/`, `train/`, `utils/`
- `configs/` — YAML experiment configs (tiny-by-default; scale-up provided later)
- `experiments/`, `dashboard/` — runner CLI and Streamlit dashboard (later milestones)
- `paper/` — LaTeX sources; PDF built by CI
- `docs/` — knowledge map, architecture review, problem formalization

## Status
Milestone **M0 (foundation)**. Roadmap M0→M5 in the design spec.
