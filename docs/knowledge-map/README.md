# Knowledge Map — Prerequisites for Recall vs. Memory

> **Scope.** Only the math the beachhead (recall vs. recurrent state, §`docs/problem`) actually
> uses. Tags: **[PROVEN]** (established theorem), **[EMPIRICAL]**, **[CONJECTURE]**.
> Bibkeys like `[arora2024based]` point at verified `paper/refs.bib` entries. Classical
> textbook results are attributed in prose; they enter `refs.bib` only after web-verification
> during paper writing.

---

## 1. Linear algebra of associative recall

**Outer-product (linear) associative memory.** Store `D` pairs `(k_i, v_i)`, keys `k_i ∈ ℝ^{d}`,
values `v_i ∈ ℝ^{m}`, in a single matrix state

```
S = Σ_{i=1}^{D} v_i k_iᵀ        (m × d)
```

Read with query `q`: `ŷ = S q = Σ_i v_i (k_iᵀ q)`. If the queried key `k_j` is orthonormal to
the rest, `ŷ = v_j` exactly; otherwise

```
ŷ = v_j (k_jᵀ q) + Σ_{i≠j} v_i (k_iᵀ q)     ← crosstalk / interference
```

**[PROVEN]** (classical linear associative memory; Anderson 1972, Kohonen 1972).

**Capacity = rank.** `S` has rank `≤ d`. At most `d` mutually orthogonal keys fit, so a
rank-`d` state stores `≤ d` interference-free associations. **[PROVEN]** (rank argument). This
is the linear-algebra shadow of the information-theoretic `Ω(n)`-state bound: a *fixed* `d`
cannot hold `D ≫ d` clean associations.

**Attention has no fixed-rank bottleneck.** Softmax attention keeps every key in the KV cache
(size `O(n)`), so recall is exact at `O(n)` memory / `O(n²)` compute. Linear attention replaces
the `n`-row cache with the fixed `m×d` matrix `S` above — *that substitution is the entire
recall–memory tradeoff.* **[PROVEN/standard]** [katharopoulos2020transformers]

**SVD & Eckart–Young.** The best rank-`r` approximation of the ideal `n`-pattern memory has
error `Σ_{j>r} σ_j` (tail singular values). A fixed state is a forced low-rank projection;
recall error is governed by the spectrum it discards. **[PROVEN]** (Eckart–Young–Mirsky).

---

## 2. Information theory (the engine of the bound)

- **Entropy / mutual information.** `H(X) = −Σ p log p`; `I(X;Y) = H(X) − H(X|Y)`.
- **Data-processing inequality (DPI).** For the Markov chain `context X → state → answer`,
  `I(answer; X) ≤ I(state; X) ≤ H(state) ≤ |state|` bits. **[PROVEN].** *The whole lower-bound
  program is this line:* to answer queries that depend on `b` bits of context, the state must
  carry `≳ b` bits.
- **Fano's inequality.** `H(X | guess) ≤ H_b(ε) + ε·log(|𝒳|−1)`, with `H_b` the binary entropy.
  **[PROVEN].** This is how error tolerance `ε` enters a lower bound (and how our **H1** picks
  up its `(1 − H_b(ε))` factor).
- **Rate–distortion / MDL.** Reconstructing a source of entropy `H` to distortion `≤ D` needs
  `≥ R(D)` bits. **[PROVEN].** This is the formal version of CLAUDE.md's "compress experience
  near entropy limits" (Q3): the memory budget is lower-bounded by the source's rate, not by
  raw length.
- **Entropy of the key→value map (our `H`).** With `D` pairs and values uniform over `|U|`,
  the map carries `H = D·log|U|` bits (worst case). Repetition, skew (Zipf), and predictable
  bindings make `H ≪ D·log|U|` — the regime our mechanism targets.

---

## 3. Communication complexity (how the bound is actually proven)

**One-way model.** Alice holds `x`, sends *one* message to Bob, who holds `y` and must output
`f(x,y)`. The minimum message length is the one-way communication complexity.

**INDEX.** Alice has `x ∈ {0,1}^N`, Bob has `i ∈ [N]`, output `x_i`. Randomized one-way CC is
`Ω(N)`. **[PROVEN]** [kremer1999randomized]. Intuition: Alice doesn't know which bit Bob will
ask, so her message must essentially retain all of `x`.

**Reduction to recurrent state (the key move).** A causal recurrent model that reads the
context and then the queries *is* a one-way protocol: the **state at the context→query boundary
is Alice's message.** Encode an INDEX instance as a recall context (`x` → key→value bindings,
`i` → the query); a model solving recall solves INDEX, so its state needs `Ω(N)` bits.
**[PROVEN]** — this is precisely the method behind [arora2024based] (BASED Thm 3.1) and
[wen2025rnns].

**Augmented INDEX & `ε`-error.** Information-cost refinements give *error-parameterized*
bounds (e.g., `(1±ε)`-approximating the number of distinct elements in a stream needs
`Ω(1/ε²)` bits). **[PROVEN]** (Bar-Yossef et al. 2004; Indyk–Woodruff). *Relevance & honesty:*
this is the machinery our **H1** would use — and the reason H1 may turn out to be a
near-corollary rather than a hard new theorem. We will report whichever it is.

---

## 4. Associative-memory background

- **Classical Hopfield (1982).** Binary attractor network; storage capacity `≈ 0.138 N`
  patterns before spurious minima dominate (Amit, Gutfreund & Sompolinsky, 1985). **[PROVEN].**
- **Modern Hopfield (2021).** Continuous-state energy whose update rule *equals softmax
  attention*; storage grows **exponentially** in the representation dimension, with one-step
  retrieval and exponentially small error. **[PROVEN/EMPIRICAL]** [ramsauer2021hopfield].
  ⇒ Attention *is* a high-capacity associative memory; the price paid for that capacity is the
  `O(n)` KV store.
- **Why this matters for us.** We want modern-Hopfield-grade recall capacity while paying only
  for the *surprising* (high-entropy) bindings — a **sparse, entropy-gated store** instead of
  the full `O(n)` cache. The capacity theory says such a store *can* recall; the **gating** is
  where the sub-`O(n)` savings — and whatever novelty survives the prior-art gate — actually
  live (mechanism **H2** in `docs/problem`).

---

*Feeds the paper's Mathematical Foundations section (M1). Classical-result citations
(Hopfield 1982; Amit et al. 1985; Anderson/Kohonen 1972; Bar-Yossef et al. 2004) are
web-verified before entering `paper/refs.bib`.*
