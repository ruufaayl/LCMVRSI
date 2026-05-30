**I raced six architectures on associative recall. The winner reveals something the textbook "fixed state is the bottleneck" story misses.**

This is the empirical **recall–memory frontier**: held-out recall vs. each model's fixed state (left) and throughput (right), all at matched budget on MQAR.

The headline: only the **content-addressed Transformer** solves the task. Every subquadratic model sits near chance. But look closer — two findings the usual "you need a big enough state" framing doesn't explain:

1. **A non-selective SSM is ≈ chance** — far worse than its state size predicts. Content-*independent* dynamics can't route by what the query is asking for. (This is exactly the gap Mamba's selectivity was invented to close.)
2. **Hyena fails *despite having no fixed state at all*.** Its convolution filter is content-independent, so unlimited "memory" doesn't help.

Takeaway: a fixed state is sufficient to fail, but **content-based addressing** is an orthogonal requirement that the classic Ω(n) state bound doesn't capture. Recall needs both *room* and *the right way to look things up*.

This reproduces and sharpens prior art (Zoology, BASED) — I claim the clean, runnable comparison, not the phenomenon.

Reproduce: `uv run --extra viz python experiments/frontier.py --benchmark mqar`

🔗 https://github.com/ruufaayl/LCMVRSI

📎 Attach: images/frontier_mqar.png

#MachineLearning #DeepLearning #Mamba #StateSpaceModels #Transformers #LLM #AIResearch #Hyena #AssociativeRecall #PyTorch
