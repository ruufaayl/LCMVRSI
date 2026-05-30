**My new mechanism didn't beat the baselines. I'm posting the negative result anyway — because that's what honest research looks like.**

I built SGSM (surprise-gated memory) expecting it to recall well while keeping a small, novelty-sized state. On the structured-recall task, here's what actually happened:

• The surprise gate **wrote ~98% of tokens** — it did *not* stay sparse.
• So SGSM's state stayed near its worst case, and it **matched, not beat**, the baselines.
• The gate-ablation (forced fully open) performed the same — confirming the gate added nothing here.

Why? A clean, diagnosable cause: the training loss only supervises recall *at query positions*. The backbone is therefore **never trained to predict the filler**, so the surprise signal stays high everywhere and the gate can't learn what's "predictable." The mechanism's premise needs a language-model signal it wasn't given. (A confound on top: no model — even the Transformer at 6,000 steps — fully solved this task at the tiny scale tested, so the comparison couldn't isolate an advantage anyway.)

The fix is specific and testable: **add an auxiliary next-token loss** so surprise becomes informative. That's the next experiment, documented in the repo.

A falsified hypothesis with a diagnosed cause and a concrete next step is a result. I'd rather ship that than a cherry-picked win.

🔗 https://github.com/ruufaayl/LCMVRSI

📎 Attach: images/h2_structured_recall.png

#AIResearch #MachineLearning #DeepLearning #OpenScience #ResearchIntegrity #LLM #NeuralNetworks #ReproducibleResearch
