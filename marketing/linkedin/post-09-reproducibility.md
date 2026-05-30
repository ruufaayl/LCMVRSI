**A result you can't reproduce isn't a result. Here's the engineering that makes every number in this project re-runnable on a laptop.**

LCMVRSI is built so anyone can verify it end-to-end — no GPU cluster required:

• **One-command experiments.** Each study (`run.py`, `sweep_recall.py`, `frontier.py`, `h2_compare.py`) regenerates its exact figure and a self-describing JSON (config + metrics + library versions + git commit).
• **Tested, typed, linted.** 109 pytest cases (output shapes, causality/no-future-leak, data-gen correctness, training smoke tests) + ruff, all green in CI on every push.
• **A live dashboard.** `streamlit run dashboard/app.py` reads the results folder and renders the recall curves, the frontier, and the H2 comparison interactively.
• **A paper that builds itself.** The LaTeX report (theory + experiments + results + an honest limitations section) is compiled to PDF by GitHub Actions and published as an artifact — no local TeX needed.
• **Tiny by design.** Everything runs on CPU / a 4 GB GPU, because synthetic tasks isolate the *mechanism* rather than chasing scale.

This is the part that doesn't trend on social media but is the whole point: science you can check.

Clone it, run `uv run pytest`, then `uv run --extra viz streamlit run dashboard/app.py`.

🔗 https://github.com/ruufaayl/LCMVRSI

📎 Attach: a screenshot of the Streamlit dashboard (or reuse images/frontier_mqar.png).

#ReproducibleResearch #MLOps #MachineLearning #PyTorch #Streamlit #ContinuousIntegration #OpenSource #AIResearch #DataScience
