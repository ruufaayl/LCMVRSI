# Architecture Review — Complexity & Recurrent-State Derivations

> **Scope.** The four architectures on the recall–memory frontier: softmax-attention
> Transformer, linear attention, SSM/Mamba, RWKV. For each: mechanism → time → memory →
> **recurrent-state size** (the column that decides recall). `n` = sequence length, `d` =
> model width, derivations from each model's definition (tagged **[PROVEN]** where they follow
> directly; standard results). Verified citations in `paper/refs.bib`.

---

## 1. Softmax-attention Transformer [vaswani2017attention]

**Mechanism.** `A = softmax(QKᵀ / √d_k)`, output `A V`, with `Q,K,V ∈ ℝ^{n×d}`.

- **Time.** The `n×n` score matrix costs `O(n² d)`. **[PROVEN/standard].**
- **Memory.** Naive materialization of `A` is `O(n²)`; **FlashAttention** [dao2022flashattention]
  computes the *exact* output in `O(n)` memory by tiling + recomputation. So attention is
  `O(n²)` *time* but need not be `O(n²)` *memory*.
- **Autoregressive inference.** A **KV cache** of size `O(n d)` is kept; per-step cost `O(n d)`.
- **Recurrent state.** *None fixed* — the KV cache **grows with `n`**. There is no fixed-rank
  bottleneck, so recall is **exact** at any width. This is the baseline the subquadratic models
  trade against.

---

## 2. Linear attention [katharopoulos2020transformers]

**Mechanism.** Replace `exp(qᵀk)` with a kernel feature map `φ`:

```
out_t = ( φ(q_t)ᵀ Σ_{i≤t} φ(k_i) v_iᵀ ) / ( φ(q_t)ᵀ Σ_{i≤t} φ(k_i) )
```

Maintain the running **state** `S_t = S_{t-1} + φ(k_t) v_tᵀ` (size `d_φ × d_v`) and normalizer
`z_t = z_{t-1} + φ(k_t)` (size `d_φ`).

- **Time.** `O(n · d_φ d_v)` total — **linear in `n`**. Inference per step: `O(d_φ d_v)`,
  **constant in `n`**. **[PROVEN/standard].**
- **Memory / recurrent state.** Fixed `O(d_φ d_v)`, **independent of `n`**.
- **Recall.** `S_t` is exactly the rank-limited outer-product memory of
  `docs/knowledge-map §1`: it cleanly holds `≲ d_φ` associations, then interferes. **This is
  the bottleneck in its purest form.**

---

## 3. SSM / S4 / Mamba [gu2022s4; gu2023mamba]

**Mechanism.** A (discretized) linear recurrence per channel:

```
h_t = Ā h_{t-1} + B̄ x_t ,   y_t = C h_t (+ D x_t) ,   h_t ∈ ℝ^N
```

- **S4** [gu2022s4]: `(Ā, B̄, C)` are **input-independent**, so the map is a long **convolution**
  computable in `O(n log n)` (FFT) — fast, but content-blind.
- **Mamba** [gu2023mamba]: makes `B, C, Δ` **input-dependent** ("selective"), recovering
  content-based routing at the cost of losing the fixed convolution; computed by a hardware-
  aware **parallel scan** in `O(n)` time.
- **Time.** `O(n · N d)` total (scan); inference per step `O(N d)`, **constant in `n`**.
- **Memory / recurrent state.** Fixed `O(N d)` (`N` = SSM state dim, typically small e.g. 16).
- **Recall.** Selectivity improves *which* content is written, but the state is still
  **fixed-size**, so it is governed by **BASED Thm 3.1** [arora2024based]: solving MQAR needs
  `N d ≳ n`. Zoology [arora2024zoology] confirms the `dim ≈ n` requirement empirically.

---

## 4. RWKV [peng2023rwkv]

**Mechanism.** Token-shift mixing plus a linear-attention-style **WKV** recurrence with
per-channel exponential decay `w` and a current-token bonus `u`:

```
wkv_t = ( Σ_{i<t} e^{-(t-1-i)w + k_i} v_i  +  e^{u + k_t} v_t )
        / ( Σ_{i<t} e^{-(t-1-i)w + k_i}    +  e^{u + k_t} )
```

maintained by carrying running (numerator, denominator) sums with decay — a fixed-size state.

- **Time.** `O(n d)` total; inference per step `O(d)`, **constant in `n`**. **[PROVEN/standard].**
- **Memory / recurrent state.** Fixed `O(d)` per layer.
- **Recall.** Fixed `O(d)` state ⇒ same `Ω(n)` bound applies; worse, the **exponential decay**
  `e^{-(t-i)w}` actively down-weights *old* bindings, so long-range recall degrades fastest.

---

## 5. Summary table

| Architecture | Train time | Infer/step | Memory | **Recurrent state** | Exact recall? |
|---|---|---|---|---|---|
| Transformer [vaswani2017attention] | `O(n² d)` | `O(n d)` | `O(n²)` naive / `O(n)` Flash | **`O(n d)` KV cache — grows** | **Yes** |
| Linear attention [katharopoulos2020transformers] | `O(n d_φ d_v)` | `O(d_φ d_v)` | `O(d_φ d_v)` | **`O(d_φ d_v)` fixed** | No (rank-limited) |
| SSM / Mamba [gu2023mamba] | `O(n N d)` | `O(N d)` | `O(N d)` | **`O(N d)` fixed** | No (BASED Thm 3.1) |
| RWKV [peng2023rwkv] | `O(n d)` | `O(d)` | `O(d)` | **`O(d)` fixed + decay** | No (+ recency bias) |

---

## 6. Synthesis (what this buys the project)

The three subquadratic models differ in *mechanism* but share one structural fact: a **fixed-
size recurrent state** that does not grow with `n`. That single property is what
[arora2024based] Thm 3.1 turns into the worst-case `Ω(n)` recall barrier, and what
[jelassi2024copying] / [wen2025rnns] echo for copying / retrieval. Transformers escape only by
keeping an `O(n)` (growing) KV cache and paying `O(n²)` compute.

So the design space for *beating* the worst case is **not** "make the fixed state cleverer at
fixed size" (the bound forbids it) — it is "**let the state grow with realized entropy `H`, not
with `n`**." A surprise/entropy-gated sparse store (mechanism **H2**, `docs/problem`) sits
exactly between the growing-`O(n)` KV cache and the fixed-`O(1)` recurrent state: it grows to
`O(H)`. On structured inputs (`H ≪ n`) that is the win; on worst-case inputs (`H = Θ(n)`) it
degenerates to attention's `O(n)` and the bound is respected. **[CONJECTURE]** — to be tested
in M2–M4.

---

*Feeds the paper's Architecture Review / Related Work (M1).*
