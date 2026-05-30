**Why do long-context AI models forget the middle of a document? I open-sourced a project to pin down the math — and test it honestly.**

Transformers recall anything in their context, but pay O(n²) for it. The efficient alternatives everyone's excited about — Mamba, RWKV, linear attention — compress the past into a small, fixed-size "state." That trade buys speed and long context… but it has a hard limit on *associative recall*: once there are more facts than the state can hold, recall breaks.

LCMVRSI is my from-scratch, fully reproducible study of that recall–memory tradeoff. No hype: every claim is tagged [PROVEN], [EMPIRICAL], or [CONJECTURE], and nothing enters the references until it's verified.

What's inside:
• 6 sequence architectures (Transformer, linear attention, SSM/S4, RWKV, Hyena, + a new mechanism) on ONE tested interface
• 5 synthetic recall tasks (associative recall, copying, induction, needle-in-a-haystack…)
• A proven lower bound on the memory recall needs
• A live dashboard + a paper auto-built in CI

How to use it: `git clone`, `uv sync --group dev`, `uv run pytest`, then run any experiment with one command.

Over the next posts I'll walk through what I built, what it proves, and the honest negative result at the end.

🔗 https://github.com/ruufaayl/LCMVRSI

📎 Attach: images/state_spectrum.png

#MachineLearning #DeepLearning #LLM #Transformers #Mamba #StateSpaceModels #LongContext #AIResearch #OpenSource #PyTorch
