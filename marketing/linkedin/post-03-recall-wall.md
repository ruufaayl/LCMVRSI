**I trained a Transformer and a linear-attention model on the same recall task. One hits 100%. The other falls off a cliff. Here's the picture.**

The task is MQAR (Multi-Query Associative Recall): the model reads a list of key→value bindings, then has to answer queries about them. Crucially, the bindings are re-randomized every sequence — so the answer can't be memorized in the weights. It has to be *retrieved from context*.

The result (matched width, depth, and training budget):
• **Transformer** — stays at ~1.00 recall as the number of pairs grows. It keeps a growing key–value cache, so it can always look back.
• **Linear attention** — tracks the transformer at first, then collapses once the number of associations exceeds what its fixed-size state can hold.

This isn't a bug; it's the **recall–memory tradeoff**, and reproducing it cleanly (with a properly converged baseline) is the foundation for everything else in the project. Getting the transformer to actually converge took a learning-rate schedule + gradient clipping — without them the "gap" would have been an optimization artifact, not a real limit.

Reproduce it in one command: `uv run --extra viz python experiments/sweep_recall.py`

🔗 https://github.com/ruufaayl/LCMVRSI

📎 Attach: images/recall_vs_pairs.png

#MachineLearning #DeepLearning #Transformers #LinearAttention #AssociativeRecall #LLM #LongContext #AIResearch #PyTorch
