**Most "AI research" repos can't be re-run. I built this one rigor-first — so every number is reproducible and every claim is falsifiable.**

Before any model or theorem, LCMVRSI started with the boring, essential stuff that makes research trustworthy:

• **One clean interface.** Every model implements `SequenceModel` and self-reports its memory/compute story (`state_size`, `complexity`). Every task implements `Benchmark`. Swapping models is a one-line config change.
• **Reproducibility by default.** Global seeds, deterministic flags, and the full environment (library versions + git commit) logged into every result JSON.
• **Tests + CI.** 109 tests (model shapes, causality, data-generation correctness, a training smoke test) run on every push via GitHub Actions, alongside linting.
• **An honesty discipline.** Claims are tagged [PROVEN] / [EMPIRICAL] / [CONJECTURE]. No citation enters the bibliography until its title/authors/venue/year are web-verified. Negative results are reported, not buried.

Why it matters: the recent literature shows some recall gaps are *optimization artifacts*, not real limits. The only way to tell the difference is a disciplined, converged, reproducible setup. That foundation is what lets the later results mean something.

Clone it and run `uv run pytest` — it should be green in seconds.

🔗 https://github.com/ruufaayl/LCMVRSI

📎 Attach: a screenshot of `uv run pytest` (109 passing) — or post text-only.

#MLResearch #ReproducibleResearch #SoftwareEngineering #MachineLearning #PyTorch #ContinuousIntegration #OpenScience #AI
