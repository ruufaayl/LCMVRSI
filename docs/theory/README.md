# Theory — An entropy floor on recurrent state for associative recall (H1)

> **Honesty tags.** **[PROVEN]** = theorem with a proof (under explicitly stated assumptions);
> **[CONJECTURE]** = believed, not proven. This document proves a *necessary-condition* (lower
> bound) on recurrent-state size. It does **not** prove that any mechanism *achieves* the bound
> — achievability (H2) is separate and remains [CONJECTURE]/[EMPIRICAL]. The proof technique
> below (data-processing + Fano) is **standard**; the contribution is the *entropy
> reparameterization* of the known length-based bound, not the machinery.

---

## 1. Setup and assumptions

We make the recurrent bottleneck explicit; this is the assumption under which all such bounds
(including BASED [arora2024based]) hold, and it is exactly what separates recurrent models from
attention.

- **(A1) MQAR instance.** A context encodes a key→value map `M`: keys `k_1,…,k_D` (distinct,
  at known positions) bound to values `v_1,…,v_D ∈ U`, `|U|` the value alphabet. Write
  `V = (v_1,…,v_D)`. The map is drawn from a task distribution `𝒟`; let `H := H(M)` be its
  Shannon entropy in bits. With key positions fixed/known, `H(M) = H(V)`.
- **(A2) `S`-bit recurrent model.** The model summarizes the context into a **state**
  `Z = g(context) ∈ {0,1}^S` (so `H(Z) ≤ S`), and every query answer depends on the context
  **only through `Z`**: for a queried key `k`, the prediction is `v̂ = h(Z, k)`. SSMs, linear
  attention, and RWKV satisfy (A2) with `Z` the final recurrent state; softmax attention does
  **not** (its answers read an `O(n)`-bit KV cache, not a fixed `Z`).
- **(A3) Error.** Querying all `D` keys yields predictions `V̂ = (v̂_1,…,v̂_D)`, a function of
  `Z` (keys are known). The average per-query error is `ε := (1/D) Σ_i Pr[v̂_i ≠ v_i] ≤ 1/2`.

---

## 2. Proposition (entropy floor on recurrent state) — [PROVEN under A1–A3]

> **Proposition 1.** Any `S`-bit recurrent model (A2) answering the `D` associative-recall
> queries of an MQAR instance (A1) with average per-query error `ε ≤ 1/2` (A3) must satisfy
>
> **`S ≥ H(M) − D·[ H_b(ε) + ε·log₂(|U|−1) ]`**  bits,
>
> where `H_b` is the binary entropy function.

**Proof.**

*Step 1 — data processing.* The answers form a Markov chain `V → context → Z → V̂` (by A2,
`V̂` is a deterministic function of `Z` once the known keys are fixed). Hence
`I(V; V̂) ≤ I(V; Z) ≤ H(Z) ≤ S.`  (1)

*Step 2 — Fano per query.* For each query, `v̂_i` predicts `v_i ∈ U` with error
`P_{e,i} = Pr[v̂_i ≠ v_i]`. Fano's inequality gives
`H(v_i | v̂_i) ≤ H_b(P_{e,i}) + P_{e,i} · log₂(|U|−1).`
Since conditioning cannot increase entropy, `H(v_i | V̂) ≤ H(v_i | v̂_i)`, so
`H(V | V̂) ≤ Σ_i H(v_i | V̂) ≤ Σ_i [ H_b(P_{e,i}) + P_{e,i} log₂(|U|−1) ].`
By concavity of `H_b` (Jensen) and `(1/D)Σ_i P_{e,i} = ε ≤ 1/2`,
`H(V | V̂) ≤ D·H_b(ε) + D·ε·log₂(|U|−1).`  (2)

*Step 3 — combine.* With keys known, `H(M) = H(V)`, so
`I(V; V̂) = H(V) − H(V | V̂) ≥ H(M) − D·[H_b(ε) + ε log₂(|U|−1)].`  (3)
Chaining (3) with (1) gives `S ≥ H(M) − D·[H_b(ε) + ε log₂(|U|−1)]`. ∎

---

## 3. Corollaries

- **C1 (exact recall).** At `ε = 0` the slack vanishes: **`S ≥ H(M)`** — a fixed recurrent
  state must carry at least the entropy of the map it answers. [PROVEN]
- **C2 (recovers the known length bound).** On the uniform worst case (distinct keys, i.i.d.
  uniform values), `H(M) = D·log₂|U|`; with `D = Θ(n)` this is `Θ(n)`, so Proposition 1 gives
  `S = Ω(n)` for small `ε` — matching BASED Thm 3.1 [arora2024based]. The known result is the
  special case `H(M) = Θ(n)`. [PROVEN]
- **C3 (the opening for a mechanism).** When the realized map is low-entropy (`H(M) ≪ n` — keys
  repeat, values skewed/Zipfian, few novel bindings), the floor is `≪ n`. Proposition 1 then
  **does not forbid** a recurrent model from matching attention's recall with `o(n)` state on
  structured inputs. Whether one *achieves* `Θ(H(M))` is **not** settled here — that is the H2
  achievability question (separate, [CONJECTURE]). [PROVEN that the bound permits it; achiev.
  open]

The bound is informative precisely when the Fano slack is small relative to `H(M)`, i.e. when
`ε` is small (or `|U|` modest); for `ε → 1/2` it degrades to the trivial `S ≥ 0`, as it must.

---

## 4. Honest positioning — what is and is not new

- **Technique: standard.** Steps 1–3 are textbook information theory (data-processing
  inequality + Fano). We claim **no** novelty in the method, and we present the result as a
  **Proposition**, not a headline theorem. This is the demotion the problem doc anticipated.
- **Framing: the contribution.** Prior bounds (BASED [arora2024based], Jelassi
  [jelassi2024copying], Wen [wen2025rnns]) are stated in the raw length `n` / worst case.
  Reparameterizing the floor by the **realized map entropy `H(M)`** makes explicit that the
  obstruction scales with *information content*, not *length* — and therefore that structured
  (low-entropy) streams are a regime where recurrent models are **not** information-theoretically
  barred from attention-level recall. That reframing is what motivates H2.
- **Correction to the earlier guess.** `docs/problem` §5 conjectured the form
  `Ω((1−H_b(ε))·H)`. The honest, proven form is `S ≥ H(M) − D·[H_b(ε)+ε log₂(|U|−1)]`
  (Prop. 1); we adopt the proven form and retire the guessed expression.
- **What remains open (not proven here).**
  - **Achievability** — that a concrete mechanism reaches `S = Θ(H(M))` (this is H2;
    [CONJECTURE], to be tested empirically). A lower bound says nothing about achievability.
  - **Tightness in `ε`** — whether the Fano slack term is order-optimal.
  - **Beyond (A2)** — models that re-read context (attention, chunked/segment-recurrent hybrids)
    are outside the bottleneck assumption and are not constrained by Prop. 1.

---

## 5. Relation to the empirical harness

Proposition 1 is consistent with M2–M3: at uniform MQAR with growing `num_pairs` (high `H(M)`),
the fixed-state baselines (linear attention, SSM, RWKV) degrade while the transformer — which
violates (A2) by keeping an `O(n)` cache — does not. The proposition explains *why* the wall is
fundamental for fixed-state models on high-entropy inputs, and predicts the wall **recedes** as
`H(M)` falls — the prediction H2's experiments (M4.4) are designed to probe.

*Feeds `paper/sections` Theoretical Analysis. Achievability (H2) and its experiments are M4.2–M4.4.*
