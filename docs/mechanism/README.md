# H2 — Surprise-Gated Sparse Memory (SGSM)

> **Status.** Design spec for the mechanism whose *achievability* claim Proposition 1
> (`docs/theory`) leaves open. Tags: the architecture is a concrete proposal; the claim that it
> reaches recall at state `≈ Θ(H(M))` is **[CONJECTURE]**, to be tested empirically (M4.4).

## 1. Idea

Augment a subquadratic backbone with an **external key→value store** whose **writes are gated by
predictive surprise**. Only surprising (novel) tokens are written, so the store grows with the
realized novelty of the stream (`≈ H(M)`), not with length `n`. Reads are differentiable
attention **over the store**. On uniform inputs (`H=Θ(n)`) the gate fires everywhere and the
module **degrades gracefully to attention** (`O(n)` state, `O(n²)` compute) — exactly the regime
Proposition 1 says no mechanism can beat.

## 2. Architecture (a `SequenceModel`)

1. **Embeddings:** `tok_emb + pos_emb`.
2. **Backbone:** `n_layers` causal **linear-attention** blocks (fixed `O(d²)` state) → hidden `h`.
3. **Surprise signal (the gate input):** the backbone predicts the next token via a shared head;
   the per-position surprise is the negative log-likelihood of the *observed* token,
   `s_t = −log p(x_t | x_{<t})` (with `s_0 = 0`). High `s_t` ⇒ token `t` was novel.
4. **Write gate:** `g_t = σ(α·(s_t − τ))`, learnable threshold `τ` and sharpness `α`. Soft in
   `(0,1)` for differentiable training; a hard write is `g_t > ½` (i.e. `s_t > τ`) at inference.
5. **Store & read:** project `h` to `q_t, k_t, v_t`. Read is causal attention over entries `j ≤ t`
   with the write-gate folded into the weights:
   `read_t = Σ_{j≤t} softmax_j( ⟨q_t,k_j⟩/√d + log(g_j+δ) ) · v_j`.
   Low-gate entries are effectively absent, so the read sees only written bindings.
6. **Output:** `out = h + W_o·read`; task logits `= head(LN(out))`. (Same head supplies the
   surprise distribution from `h`.)

## 3. State accounting and complexity

- **Realized state (the quantity of interest):** number of hard writes
  `m = Σ_t 𝟙[s_t > τ]`, each entry `(k,v)` of `2·d` floats ⇒ `m·2d·4` bytes **+** the backbone's
  fixed linear-attention state. `m` is **input-dependent** and tracks novelty; M4.4 measures its
  mean on held-out data and plots recall against it.
- **`state_size` (static property):** we report the **worst-case** bound (backbone state +
  `max_seq_len·2d·4`), since the store *can* grow to `O(n)`. This is honest: SGSM is not a
  fixed-state model; its win is that *realized* state is `≪` worst case on structured inputs.
- **Compute:** training/eval forward is dense `O(n²)` (attention over all past, gate-weighted);
  inference with hard gating reads only `m` entries ⇒ `O(n·m)`, subquadratic when `m ≪ n`. We
  state this split honestly and do not claim subquadratic *training* here.

## 4. What success / failure looks like (falsification)

- **Support [EMPIRICAL] for H2** iff, on **structured** MQAR (`H ≪ n`: repeated keys / skewed
  values), SGSM attains recall close to the transformer while its **realized** store `m ≪ n` and
  below matched fixed-state baselines' capacity — and on **uniform** MQAR it neither beats nor is
  beaten by attention by much (graceful fallback).
- **Falsified** if it fails to beat matched-(state,compute)-budget baselines on structured inputs,
  or if any gain vanishes once optimization is tuned (the Okpekpe–Orvieto caveat), or if the gate
  collapses (writes everything or nothing) regardless of input entropy.

## 5. Ablations (M4.4)

- **Gate on/off:** replace `g_t` by `1` (write-everything) — should recover an attention-like model
  and erase the state savings.
- **Store cap `C`:** cap entries (keep highest-surprise) and sweep `C` — recall vs. `C` is the
  achievability curve; the claim is recall saturates near `C ≈ H(M)`.
- **Structured vs uniform inputs:** vary the key/value entropy of the MQAR generator; SGSM's
  advantage should appear only as `H(M)` drops.

*Implementation is M4.3 (TDD, registered as `sgsm`); experiments are M4.4.*
