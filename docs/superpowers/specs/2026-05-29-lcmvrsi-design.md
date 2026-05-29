# LCMVRSI — Design Spec

**Project:** Long-Context Memory, Verifiable Reasoning, and Scalable Intelligence (LCMVRSI)
**Author:** Rufayl Waseem
**Date:** 2026-05-29
**Status:** Draft — awaiting user review
**Remote:** https://github.com/ruufaayl/LCMVRSI.git

---

## 1. Goal and honest scope

The umbrella ambition (from `claude.md`) is frontier AGI research across long-context
memory, verifiable reasoning, and scalable intelligence. That umbrella spans many open
problems and **cannot be "solved" wholesale** — nor can a breakthrough be promised before
the work exists. This project therefore commits to the *method* of real research:

1. Attack **one precise, tractable open problem** (the beachhead, §3).
2. Build deep prerequisite knowledge and a **citation-verified** literature review.
3. Attempt a **genuine novel result** (a theorem / lower bound and a mechanism).
4. Report honestly — **[PROVEN] / [EMPIRICAL] / [CONJECTURE]** — including negative results.

Whether the result is "groundbreaking" is *discovered through the work, not asserted before
it*. Per directive §8: no fabricated theorems, no fabricated citations, no false certainty.

### Non-goals (YAGNI)
- Not "solving" hallucination / AGI / reasoning in general this milestone.
- No fabricated experimental numbers, theorems, or references — ever.
- No large-scale training (hardware is a 4 GB laptop GPU; experiments are small/synthetic
  by design, which is ideal for isolating the mechanism).

---

## 2. Deliverable form
- **Experiments + code:** showcased on GitHub (runnable, tested, CI-checked).
- **Theory + paper:** LaTeX in `paper/`, built to **PDF via GitHub Actions** (`tectonic`),
  published as a CI artifact / release so no local LaTeX install is required.
- **Dashboards:** a local **Streamlit** app reading experiment logs, plus committed
  matplotlib figures for the README/paper.

---

## 3. Beachhead problem: recall vs. memory (MQAR)

**Phenomenon.** Subquadratic models (linear attention, SSMs/Mamba, RWKV) trade away exact
associative recall. To answer many independent key→value queries over a long context, a
fixed-size recurrent state must grow with the number of distinct pairs; softmax attention
instead pays O(n²) for exact recall. This is the documented **recall–throughput /
recall–memory tradeoff** (Zoology; Based).

**Novelty target (HYPOTHESIS — to be validated against prior art first):**

- **Theorem attempt (lower bound):** Any model answering *K* independent associative-recall
  queries over a context containing *D* distinct key→value pairs, with error ≤ ε, requires
  recurrent state of Ω(f(K, D, ε)) bits. Likely route: a communication-complexity / counting
  argument on the state as a bottleneck channel.
  - **Prior-art gate:** related theory exists (e.g., representational limits of SSMs/linear
    attention; recall-vs-state observations in Zoology/Based). The literature review (M1)
    must establish exactly what is already proven. If a tight form exists, the contribution
    shifts to a **tighter regime**, a **different error model**, or the mechanism below.
- **Mechanism (proposed architecture):** augment a subquadratic backbone with a **sparse,
  entropy-gated external associative memory** — writes triggered by a surprise/entropy
  signal, reads via approximate retrieval. **Claim:** recovers associative-recall accuracy
  at sub-O(n²) memory/compute, pushing the empirical recall–memory Pareto frontier.

**Falsification plan.** The lower bound is falsified if a construction answers the queries
with asymptotically less state. The mechanism is falsified if, on MQAR and related tasks, it
fails to beat matched-budget baselines on the accuracy-vs-(memory, compute) frontier.

---

## 4. Architecture

### Repo structure
```
LCMVRSI/
├── pyproject.toml  uv.lock  .gitignore  LICENSE(MIT)  README.md
├── .github/workflows/   ci.yml (ruff + pytest) · paper.yml (tectonic → PDF)
├── configs/             base.yaml · models/*.yaml · benchmarks/*.yaml
├── src/lcmvrsi/
│   ├── models/      base.py (SequenceModel ABC) · transformer.py · linear_attention.py · registry.py
│   ├── benchmarks/  base.py (Benchmark ABC) · mqar.py · registry.py
│   ├── train/       runner.py · metrics.py · logging.py
│   └── utils/       seed.py · config.py (pydantic + YAML) · profiling.py
├── experiments/     run.py (CLI) · sweeps/
├── dashboard/       app.py (Streamlit)
├── docs/            knowledge-map/ · architecture-review/ · problem/ · superpowers/specs/
├── paper/           main.tex · sections/ · refs.bib · figures/ · Makefile
├── results/         figures/ + JSON logs
└── tests/           test_models.py · test_benchmarks.py · test_runner.py
```

### Core interfaces (small, independently testable)
- **`SequenceModel(nn.Module)`** — `forward(input_ids) -> logits`; reports `state_size`
  (recurrent-state bytes where applicable) and `complexity(seq_len) -> {flops, memory}` so
  every model self-documents its memory/compute story. Shared config: vocab, d_model,
  n_layers, etc.
- **`Benchmark`** — `generate(split, n, seq_len, seed) -> (inputs, targets)` and
  `evaluate(model) -> metrics`; defines task-specific metrics (recall accuracy at query
  positions for MQAR).
- **`Runner`** — config-in → trains, evaluates, logs JSON + TensorBoard, records throughput
  and peak memory.
- **Registries** map string names → classes so configs are declarative.

### Conventions
- **Reproducibility:** global seed; deterministic flags where feasible; logged env
  (torch/CUDA versions, GPU name) in every result file.
- **Honesty tags:** every non-trivial claim in docs/paper tagged
  **[PROVEN] / [EMPIRICAL] / [CONJECTURE]**.
- **Citations:** added to `refs.bib` only after web-verification of title/authors/venue/year.
- **Testing:** pytest — model output shapes; benchmark data-gen correctness (targets truly
  require recall); a runner smoke test that trains a tiny model for a few steps on CPU in
  seconds.
- **CI:** ruff + pytest on push/PR; separate workflow builds the paper PDF.
- **Configs:** tiny-by-default (fits 4 GB GPU / CPU) with scale-up configs provided.

### Tech stack
PyTorch + `uv` (Python 3.12); einops, numpy, pandas, pyyaml, pydantic; matplotlib +
TensorBoard + Streamlit; pytest + ruff; LaTeX via `tectonic` in GitHub Actions.

---

## 5. Milestone roadmap (research-first)

| Milestone | Ships |
|---|---|
| **M0 Foundation** | uv project + deps · package skeleton with ABCs · `utils/` (seed/config/profiling) · `configs/base.yaml` · CI + passing test · `paper/` LaTeX skeleton + PDF workflow · `docs/` structure · MIT license · README · `git init`, first commit, push to remote (check remote state first; never force-push; confirm before first push) |
| **M1 Research foundation** | Problem formalization doc (MQAR, recall–state tradeoff, exact open question, hypothesis + theorem statement + falsification plan) · citation-verified literature review · scoped knowledge-map prerequisites · architecture-review derivations (transformer, linear-attn, SSM/Mamba, RWKV: mechanism + complexity + state-size) → populates paper Foundations/Related Work/Problem |
| **M2 Vertical slice** | `SequenceModel` interface · Transformer + linear-attention baselines (tiny) · MQAR benchmark · experiment runner · Streamlit dashboard · tests · reproduce the linear-model recall failure empirically · complexity/memory table |
| **M3 Widen** | SSM/Mamba-style, RWKV-style, Hyena-style baselines · needle-in-haystack, copying, induction-head tasks · memory/throughput profiling · scaling sweeps · map the empirical recall–memory Pareto frontier |
| **M4 Novel contribution** | Theorem attempt (recall–state lower bound) with proof or honest gap · proposed entropy-gated sparse-memory mechanism · ablations · head-to-head vs matched-budget baselines · honest [PROVEN]/[EMPIRICAL]/[CONJECTURE] assessment |
| **M5 Paper + polish** | arXiv-style LaTeX → PDF (CI-built, published) · leaderboard · scaling/Pareto figures · limitations · reproducibility appendix |

**This session targets M0 fully, and the start of M1.**

---

## 6. Risks & mitigations
- **Our bound may already exist.** Mitigated by doing the literature review *before* claiming
  novelty; contribution then shifts to a tighter regime or the mechanism. Honestly reported.
- **Mechanism may not beat baselines.** A rigorous negative result on the Pareto frontier is
  still a publishable contribution; we report it.
- **4 GB VRAM limits scale.** Synthetic MQAR-style tasks isolate the mechanism at small
  scale; scale-up configs document how to reproduce larger.
- **Citation fabrication risk.** Hard rule: no reference enters `refs.bib` unverified.
- **Remote push safety.** Inspect remote before first push; never force-push; confirm first.

---

## 7. Open minor decisions (sensible defaults chosen; change anytime)
- License: **MIT**.
- `claude.md` (the private directive): kept in repo root for now; can be moved to `docs/` or
  gitignored from the public showcase on request.
- Paper PDF: built in CI (`tectonic`); optional local `latexmk`/`tectonic` for iteration.
