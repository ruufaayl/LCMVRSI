# Problem Formalization — Recall vs. Recurrent State in MQAR

> **Honesty tags.** Every non-trivial claim is tagged:
> **[PROVEN]** (a theorem with a proof — ours or, where attributed, prior work),
> **[EMPIRICAL]** (supported only by experiments), **[CONJECTURE]** (believed, not yet
> proven). No reference appears in `paper/refs.bib` until its title/authors/venue/year are
> web-verified. Prior-art search performed 2026-05-29; see §6.

---

## 1. The MQAR task (precise setup)

**Multi-Query Associative Recall (MQAR)** — formalized by Zoology [arora2024zoology].

- Vocabulary `V`. A subset `K ⊂ V` are *keys*, a disjoint subset `U ⊂ V` are *values*.
- A context is a token sequence of length `n` encoding `D` distinct key→value pairs
  `(k_i, v_i)`, `i ∈ [D]`, followed (and/or interleaved) by `Q` *query* positions. At query
  position `t`, the input is a key `k` previously bound in the context; the model must emit the
  associated value `v`.
- **What makes it recall, not memorization:** the binding `k_i ↦ v_i` is resampled per
  sequence, so the answer cannot be stored in the weights — it must be retrieved from the
  context at inference time.
- **Error metric.** `ε` = expected fraction of query positions answered incorrectly under the
  task distribution `𝒟` (equivalently, per-query error probability for i.i.d. queries).
- **Resource of interest.** For a *recurrent* model processing the context causally, the
  **state** is the fixed-width summary carried across positions; its size `S` is measured in
  **bits** (state dimension × bits/entry). For attention, there is no fixed state — it pays
  `O(n²)` compute / `O(n)` KV cache instead.

The central object of this project is the function `S*(D, Q, ε; 𝒟)` = the minimum recurrent
state (bits) needed to achieve query error `≤ ε` on MQAR instances drawn from `𝒟`.

---

## 2. The recall–state phenomenon

Softmax attention answers MQAR **exactly** at any model width because every query attends
back to every key (cost: `O(n²)`). Subquadratic recurrent models (linear attention, SSMs /
Mamba, RWKV) compress the context into a **fixed-size** state, so once `D` exceeds what the
state can hold, recall must degrade. This is the documented **recall–throughput /
recall–memory tradeoff** [arora2024zoology; arora2024based]: you can dial recall up by
enlarging the state, but you give back the efficiency that motivated leaving attention.

---

## 3. What is already proven (this is **not** our contribution)

The prior-art gate (§6) returned a decisive result: **the basic recall-vs-state lower bound
for MQAR already exists and is proven.** We state it here precisely so we never reclaim it.

- **[PROVEN — Arora et al. 2024, "BASED", Thm 3.1 [arora2024based]].** Any model that depends
  causally on a binary input and solves MQAR requires `Ω(n)` bits of recurrent state.
  *Method:* information-theoretic / communication-complexity argument (the state is a
  one-way message bottleneck). *Regime:* worst-case, exact (no error tolerance).
- **[PROVEN — Jelassi et al. 2024, "Repeat After Me", Thm 2.7 [jelassi2024copying]].** A
  generalized SSM with state set `𝒮` has copying error `> 1 − |𝒮|/D^L` on length-`L` uniform
  strings — i.e., state must grow linearly in length to copy. (Copying is a sibling of
  recall; same fixed-state bottleneck.)
- **[PROVEN — Wen et al. 2025 [wen2025rnns]].** Any RNN with `o(n)`-bit memory cannot solve
  certain in-context retrieval tasks **even with chain-of-thought**; "in-context retrieval" is
  the key bottleneck separating RNNs from Transformers.
- **[EMPIRICAL + construction — Zoology [arora2024zoology]].** Recurrent models need hidden
  dimension `≈ n` to solve MQAR; gated-convolution variants need depth growing with the gap.

**Conclusion.** A from-scratch "recurrent models need `Ω(n)` state for recall" theorem is
**closed prior art**. We do not claim it. Per the design spec's prior-art gate, the
contribution must shift to a **tighter / different regime** or to the **mechanism**.

---

## 4. The open question we attack

Every bound in §3 is **worst-case over (near-)uniform inputs**: keys are distinct, bindings
are uniformly random, so the context genuinely carries `Θ(n)` bits and no fixed state can
win. Real associative streams are not like this — they have **low entropy**: keys repeat,
distributions are skewed (Zipfian), most tokens are unsurprising, and the number of *novel*
bindings actually carried, `H`, can be `≪ n`.

> **Open question.** Is the recall–state requirement governed by the **entropy `H` of the
> key→value map actually realized by the input distribution `𝒟`**, rather than by the raw
> length `n`? And can a recurrent mechanism **allocate state proportional to realized `H`** —
> matching attention's recall on structured inputs at sub-`O(n²)` cost — while gracefully
> falling back to the `Ω(n)` worst case only when `H = Θ(n)`?

This regime is explicitly **untouched** by the closest recent work:
[okpekpe2025revisiting] (Okpekpe & Orvieto, 2025) use **uniform synthetic inputs only**,
introduce **no new mechanism**, and — importantly — argue the worst-case gap may be
**confounded by optimization brittleness** rather than pure expressivity. That caveat
*strengthens* the case for an instance-aware, entropy-gated mechanism and a careful empirical
Pareto-frontier study (which is what we can actually run on a laptop).

---

## 5. Hypothesis, theorem attempt, and falsification plan

**H1 — entropy-parameterized lower bound [CONJECTURE].**
For MQAR instances whose realized key→value map has Shannon entropy `H` (bits), achieving
query error `≤ ε` requires recurrent state
`S*(·) = Ω( (1−H_b(ε)) · H )` bits, where `H_b` is binary entropy. This **recovers** BASED's
`Ω(n)` exactly when `H = Θ(n)` (uniform worst case) and is **strictly smaller** when `H ≪ n`.

- *Intended route:* reduction from **augmented-INDEX** with Alice's input drawn from an
  entropy-`H` distribution; the state is the one-way message. Error `ε` enters via the
  information cost of an `ε`-error protocol.
- *Honesty flag.* This refinement may follow with little new machinery from existing
  augmented-index / information-cost results (e.g., the `Ω(1/ε²)` distinct-elements streaming
  bounds and augmented-index tradeoffs). **If so, we will say so explicitly and demote H1 to a
  corollary**, leading instead with the mechanism (H2). We will not dress up a routine
  reduction as a deep theorem.

**H2 — entropy-gated sparse memory (the mechanism) [CONJECTURE → target [EMPIRICAL]].**
Augment a subquadratic backbone with a small **external associative memory** whose **writes
are gated by a surprise/entropy signal** (write only when an incoming binding is novel /
high-surprise) and whose **reads** use approximate nearest-key retrieval. *Claim:* on
**structured** MQAR (`H ≪ n`) it attains attention-level recall at state `≈ Θ(H)`, pushing the
empirical **recall–memory Pareto frontier** past matched-state-budget Mamba / linear-attention
/ BASED baselines — while matching them (no better, no worse) on uniform inputs where `H =
Θ(n)` and H1 says **no** mechanism can help.

**Falsification plan.**
- *H1 is falsified* if a construction solves entropy-`H` MQAR with `o(H)` state (at fixed `ε`).
- *H2 is falsified* if, on structured MQAR and matched (state, compute) budgets, the
  entropy-gated memory **fails to beat** the baselines on the accuracy-vs-resource frontier —
  or if any apparent gain vanishes once optimization is properly tuned (the Okpekpe–Orvieto
  caveat). A rigorous **negative** result here is still reported as a contribution.

---

## 6. Honest novelty positioning (summary of the prior-art gate)

| Claim | Status | Owner |
|---|---|---|
| Recurrent models need `Ω(n)` state for (worst-case) MQAR | **[PROVEN]** | BASED [arora2024based] |
| Copying needs linear state | **[PROVEN]** | Jelassi et al. [jelassi2024copying] |
| `o(n)`-bit RNNs fail in-context retrieval even w/ CoT | **[PROVEN]** | Wen et al. [wen2025rnns] |
| Worst-case gap may be optimization-confounded | **[EMPIRICAL]** | Okpekpe & Orvieto [okpekpe2025revisiting] |
| **Entropy-parameterized** recall–state bound `Ω(g(H,ε))` | **[CONJECTURE]** | **this project (H1)** |
| **Entropy-gated sparse-memory** mechanism beats matched-budget baselines on structured MQAR | **[CONJECTURE]** | **this project (H2)** |

Our defensible novelty, if it survives, is therefore **(i)** the *entropy parameterization /
framing* of an existing bound, **(ii)** the *entropy-gated mechanism* (no prior work in §6
proposes one), and **(iii)** the *structured-input empirical Pareto result*. We lead with
(ii)+(iii) — they are fully reproducible on the target hardware (a 4 GB laptop GPU) — and
treat (i) as theory support to be proven or honestly demoted.

*Adjacent but distinct:* [kawata2026measure] proves a **minimax** lower bound for
**Transformers** as measure-theoretic associative memory (statistical/sample complexity, not
recurrent **state size in bits**); related framing, different quantity. Tracked, not claimed.

---

*Populates `paper/` Problem & Related Work (M1). Mechanism implementation and the empirical
Pareto study are M2–M4.*
