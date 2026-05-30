**Transformer vs. Mamba vs. RWKV vs. Hyena — implemented from scratch, on one interface, so you can compare them apples-to-apples.**

To study the recall–memory tradeoff fairly, you need the contenders side by side at matched size. LCMVRSI now has six tiny-but-honest architectures, each self-reporting its memory/compute profile:

• **Transformer** — O(n²) time, no fixed state (a growing KV cache). Exact recall.
• **Linear attention** — O(n·d²), fixed O(d²) state.
• **SSM (S4D-style)** — O(n·d·N), fixed O(d·N) state.
• **RWKV** — O(n·d), the smallest fixed state, O(d).
• **Hyena** — O(n·log n) via FFT long convolution, no fixed recurrent state.
• **SGSM** — a new surprise-gated memory whose state *grows with input novelty* (more on that soon).

The attached chart shows the axis the whole field fights over: **how much fixed state each design carries**. That number is exactly what an information-theoretic bound constrains (next posts).

They're deliberately small, honestly-named reimplementations — not the authors' full kernels — so the comparison isolates the *mechanism*, not the engineering.

Add your own model by implementing one class and a decorator. The sweep, the frontier plot, and the complexity table pick it up automatically.

🔗 https://github.com/ruufaayl/LCMVRSI

📎 Attach: images/state_spectrum.png

#Mamba #RWKV #Hyena #StateSpaceModels #Transformers #DeepLearning #MachineLearning #AIResearch #OpenSource #PyTorch
