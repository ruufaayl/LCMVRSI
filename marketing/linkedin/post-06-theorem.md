**How much memory does recall *fundamentally* require? I proved a clean answer — and I'm honest that the proof is textbook, not magic.**

The known result (BASED, ICML 2024) says a fixed-state recurrent model needs Ω(n) bits to solve associative recall in the worst case — where n is the sequence length. True, but pessimistic: real streams aren't worst-case. Most tokens are predictable; only a few bindings are genuinely novel.

So I reparameterized the bound by the *realized entropy* of the key→value map, H(M), instead of raw length n. The result (Proposition 1 in the paper), under the recurrent-bottleneck assumption:

  S ≥ H(M) − D·[ H_b(ε) + ε·log₂(|U|−1) ]   bits

In words: a fixed recurrent state must carry at least the **information content** of the map it answers — minus a small slack that vanishes as error ε → 0.

• When H(M) = Θ(n) (uniform worst case), it recovers the Ω(n) bound.
• When H(M) ≪ n (structured input), the floor is much smaller — so a recurrent model is *not* information-theoretically barred from attention-level recall on structured data.

I present this as a proposition, not a grand theorem: the technique is just the data-processing inequality + Fano. The contribution is the *framing* — and it's the launchpad for a new mechanism.

🔗 https://github.com/ruufaayl/LCMVRSI

📎 Attach: a quote card of the inequality above (clean white background, monospace).

#InformationTheory #MachineLearning #DeepLearning #AItheory #LLM #StateSpaceModels #AIResearch #Mathematics #OpenScience
