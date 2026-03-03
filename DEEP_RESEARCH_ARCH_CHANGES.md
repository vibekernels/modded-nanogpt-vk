# Novel optimization paths for the modded-nanogpt speedrun

**The most promising untapped gains lie in three areas: fixing severe GPU underutilization on 768-dim matmuls via Stream-K scheduling, eliminating ~98% of wasted embedding optimizer compute with sparse updates, and removing Python dispatch overhead through Inductor's C++ wrapper mode.** Together these systems-level changes could shave 4–8 seconds from the 88.1s record without touching convergence. On the optimizer side, MARS-M variance reduction on Muon and EMA weight averaging offer the best shot at reducing step count. Below is a deep analysis of every novel technique uncovered, organized by expected impact.

---

## Stream-K GEMM scheduling exposes a 3× utilization gap on 768-dim

The single largest systems-level opportunity is **wave quantization waste** in the model's matmuls. A 768×768 weight matrix tiled at 128×128 produces only **36 output tiles**. The H100 has 132 SMs. In standard data-parallel GEMM scheduling, this means a single partial wave at **~27% SM utilization** — nearly three-quarters of the GPU sits idle during these operations.

**Stream-K** (Osama et al., NVIDIA, 2023) is a work-centric scheduling approach that distributes fractional tiles across all SMs by splitting work along the K-dimension. Instead of assigning whole tiles to CTAs, each persistent CTA receives an even share of total inner-loop iterations. For the 768×768 shape, Stream-K can approach **~95% utilization** versus ~27% for data-parallel. The 768×3072 FFN shapes produce 144 tiles (~1.1 waves), where Stream-K eliminates the ~10% tail wave waste.

Stream-K is fully supported in CUTLASS 3.x for Hopper (sm90) with examples and a hybrid heuristic scheduler that auto-selects between Stream-K for tail waves and data-parallel for full waves. It is **not** exposed in cuBLAS as a user option. Integration requires replacing cuBLAS calls with CUTLASS-based kernels for the specific 768-dim shapes, which can be done through `torch._inductor`'s max-autotune CUTLASS backend or custom kernel wrappers.

**Estimated savings:** 2–4 seconds (the Q/K/V/O projections at 768×768 run ~44 times per step in forward+backward across 11 layers; even partial utilization improvement on these kernels compounds heavily over 1490 steps). **Complexity:** Medium — CUTLASS integration with proper tile-size autotuning. **Risk:** Stream-K adds a small GMEM workspace for partial reductions (~70% more device memory for the GEMM workspace, but this is negligible at 768-dim). No convergence risk.

---

## Sparse embedding optimizer updates waste 98% of compute

The model's ~50K-token vocabulary embedding has a severe inefficiency: standard Adam updates all 50,000 rows every step, even though only **~1,000 unique tokens** appear per batch. This means **~98% of embedding optimizer compute — moment updates, parameter writes, and gradient communication — is pure waste**.

PyTorch's built-in `torch.optim.SparseAdam` solves this by operating only on non-zero gradient rows when `nn.Embedding(sparse=True)` is enabled. For the speedrun's embedding layer (50,304 × 768), this reduces optimizer compute from **38.4M parameters to ~768K** — a 50× reduction. The allreduce communication for embedding gradients also shrinks proportionally if sparse representations are used.

The modded-nanogpt Discussion #23 (Dec 2025) already identified this as "very reasonable" but it remains unimplemented. There are practical complications: SparseAdam does not support weight decay (a limitation for regularization), moment estimates for unseen tokens become stale (approximation rather than exact Adam), and it requires a separate optimizer instance for the embedding layer. The 5 value embeddings in the current architecture multiply the benefit but also the integration complexity.

**Estimated savings:** 0.5–1.5 seconds from reduced optimizer compute and communication for embedding parameters. **Complexity:** Low-medium — dual optimizer setup, sparse gradient handling. **Risk:** Stale moments for rare tokens could marginally affect embedding quality; no weight decay on embeddings may require compensating regularization. **Leaderboard compliance:** Yes, purely an optimizer change.

---

## Inductor's cpp_wrapper mode eliminates Python dispatch overhead for free

The simplest win is a single configuration line: `torch._inductor.config.cpp_wrapper = True`. This replaces Python wrapper code that manages kernel launches with compiled C++ code that calls `cuLaunchKernel` directly, bypassing the Python interpreter, GIL, and PyTorch dispatcher overhead between every kernel.

Intel benchmarks show **5–17% speedup for BF16 small workloads** (iteration time ≤ 40ms) — exactly the regime of a 768-dim transformer where per-step time is ~59ms and many kernels (LayerNorm, activations, element-wise ops) have sub-microsecond durations. The overhead of Python dispatch between these short kernels becomes a measurable fraction of total time.

One important caveat: cpp_wrapper **conflicts with CUDA graphs** (the code explicitly skips cudagraph capture when cpp_wrapper is enabled). Since CUDA graph capture is already on the researcher's radar, the decision is between cpp_wrapper (simpler, no static-memory constraints) and CUDA graphs (more aggressive, eliminates launch overhead entirely but requires static memory layout). For the heterogeneous batch size schedule (8→16→24), cpp_wrapper may be more practical since it doesn't require separate graph captures per batch size.

**Estimated savings:** 1–3 seconds. **Complexity:** Trivial — one config line. Stable since PyTorch 2.5. **Risk:** Minor compatibility issues with some ops; conflicts with CUDA graph approach. **Leaderboard compliance:** Yes, no extra torch.compile flags needed (this is an Inductor config, not a compile flag).

---

## CUTLASS warp-specialized ping-pong kernels for Hopper matmuls

Beyond Stream-K scheduling, the Hopper architecture's warp specialization enables a fundamentally different kernel design pattern. The **ping-pong pattern** (CUTLASS 3.x `sm90_gemm_tma_warpspecialized_pingpong`) uses three warp groups per thread block: one lightweight producer (128 threads dedicated to TMA data movement) and two consumer warp groups performing tensor core MMA. The consumers alternate between epilogue and computation — while Consumer0 writes results, Consumer1 computes, and vice versa — achieving near-perfect **overlap of epilogue with computation**.

Custom CUTLASS ping-pong kernels achieve **~280 TFLOPS on H100**, competitive with or exceeding cuBLAS. The Tawa compiler (arXiv 2510.14719) demonstrates that warp specialization can be automated from high-level specifications, achieving 96% of handwritten FlashAttention-3 throughput. Critically, **Triton does not support warp specialization** — it retains SIMT-centric abstractions that preclude explicit warp role assignment. Exploiting this requires CUTLASS C++ or inline PTX.

For 768-dim, the MLP matmuls (768→3072 and 3072→768) have sufficient N/M dimensions to benefit from ping-pong scheduling. The smaller 768×768 attention projections may see muted gains because epilogue costs aren't fully amortized. Combining ping-pong with Stream-K scheduling would address both the wave quantization and epilogue overlap problems simultaneously.

**Estimated savings:** 1–3 seconds on matmul-heavy operations. **Complexity:** High — requires CUTLASS C++ expertise and integration. **Risk:** Custom kernel maintenance burden; cuBLAS may already use similar scheduling internally for some shapes. **Leaderboard compliance:** Yes, but code readability requirement may be a concern.

---

## EMA weight averaging could replace the cooldown phase entirely

**Exponential Moving Average (EMA) of weights** maintains a running average of model parameters during training. At evaluation, the EMA weights are used instead of the raw trained weights. The AlgoPerf benchmark (Feb 2025) demonstrated that **EMA reduces steps-to-target by ~18% on average** across diverse workloads when used with NadamW, and even improves Distributed Shampoo results.

The key insight for this speedrun: EMA can effectively **replace the learning rate cooldown phase**. The cooldown exists to suppress oscillations and reveal accumulated progress. EMA accomplishes the same thing by averaging out oscillations without requiring a decay phase. This means all 1490 steps could run at high learning rate with the EMA model achieving comparable or better final loss. At the current schedule, the cooldown phase likely consumes **~15–20% of total steps** (~220–300 steps).

Implementation requires only one extra weight buffer (124M params × 2 bytes ≈ 250MB, trivial on 80GB H100s) and one line of EMA update per step. The decay rate (typically 0.999 for long training, ~0.99 or lower for 1490 steps) needs tuning. A closely related approach is **LAWA** (Latest Weight Averaging), which averages the last K checkpoints — simpler but similar effect.

**Estimated savings:** 30–75 steps to the same loss target, equivalent to 1.5–3.5 seconds. **Complexity:** Very low — ~5 lines of code. **Risk:** Decay rate tuning is critical for short runs; too aggressive averaging loses recent learning, too conservative provides no benefit. The evaluated model must be the EMA model, which changes the evaluation semantics slightly. **Leaderboard compliance:** Yes, as long as the timed run includes the EMA computation.

---

## MARS-M adds variance reduction to Muon with proven gains on GPT-2

MARS-M (arXiv 2510.21800, Oct 2025) integrates MARS-style stochastic recursive momentum with Muon's matrix orthogonalization. Instead of using raw gradients, it maintains a variance-reduced estimator that blends the current gradient with a correction from the previous step's gradient. This provably improves convergence rate from Muon's Õ(T⁻¹/⁴) to **Õ(T⁻¹/³)**.

Empirically on GPT-2 (768-dim — an exact architecture match), MARS-AdamW achieves lower training and validation loss at every checkpoint and **saves 22B tokens to reach target loss** for GPT-2 Large. The combination with Muon (MARS-M) is specifically designed to be compatible with the orthogonalization update rule.

The main cost is storing one extra gradient buffer for the previous-step gradients, increasing optimizer memory by ~33%. For a 124M parameter model on 80GB H100s, this is ~250MB — negligible. The `gamma` hyperparameter controlling variance reduction strength needs tuning, and at large batch sizes (262K tokens/step), gradient variance is already relatively low, potentially reducing the marginal benefit.

**Estimated savings:** 5–15% convergence improvement, translating to 75–220 fewer steps or equivalently 3–8 seconds. **Complexity:** Medium — requires tracking previous gradients and the recursive momentum computation. Code is available at github.com/AGI-Arena/MARS. **Risk:** Short-run behavior (1490 steps) may not allow variance reduction estimates to stabilize fully. Interaction with NorMuon's existing normalization is untested. **Leaderboard compliance:** Yes, requires p<0.01 significance.

---

## Three lower-effort convergence optimizations worth testing

**Cooldown shape optimization** is the simplest experiment: replace the current linear LR cooldown with sqrt, cosine, or 1-sqrt decay. Recent work (TMLR Aug 2025) on cooldown training dynamics shows different shapes create distinct bias-variance tradeoffs. For 1490 steps with ~200 steps of cooldown, the shape matters more than one might expect. Estimated gain: 10–30 steps (0.5–1.5s), trivial to implement, zero risk.

**Lookahead optimizer wrapper** maintains slow weights φ and fast weights θ, interpolating φ toward θ every k steps. It wraps around the existing NorMuon+Adam without modifying them. With k=5 and α=0.5, it typically provides 5–10% convergence acceleration. For 1490 steps, there would be ~298 slow-weight updates — borderline whether this is enough for the smoothing effect to manifest. Estimated gain: 1–3 seconds. ~20 lines of code.

**AdaMuon** (arXiv 2507.11005, Jul 2025) extends Muon with element-wise second-moment adaptivity on top of orthogonalized directions. Since NorMuon already has neuron-wise normalization, AdaMuon adds finer-grained (element-wise) adaptivity. Claims **>40% training efficiency over AdamW** and 23.87% wall-clock reduction for GPT-2 XL. The incremental gain over NorMuon is harder to estimate but likely 3–8%. Requires maintaining per-element second moments within the existing Muon infrastructure. Medium complexity.

---

## NVLS buffer alignment and copy engine offloading for communication

Two communication optimizations complement the NCCL tuning already on the radar. First, **ncclMemAlloc for NVLS-compatible buffers**: the DGX H100's third-gen NVSwitch has embedded ALUs (400 GFLOPS FP32) capable of performing allreduce in-fabric via the NVLS algorithm. However, NVLS requires buffers aligned to CU_MULTICAST_GRANULARITY (2MB). If PyTorch's caching allocator splits or misaligns a buffer, NCCL silently falls back to slower ring allreduce. Using `ncclMemAlloc` for communication buffers or PyTorch's new MemPool API (`torch.cuda.use_mem_pool`) ensures correct alignment. NVLS achieves **~450 GB/s allreduce bandwidth** on DGX H100, roughly 2× ring for small messages. Estimated savings: 0.5–1.5 seconds.

Second, **copy engine offloading**: instead of running NVLink data transfers on SMs (as default NCCL does), the H100's dedicated DMA copy engine achieves ~81% of the 900 GB/s theoretical NVLink bandwidth without consuming any SMs. Hazy Research's "One Kernel for All Your GPUs" demonstrates this approach. For 8×H100 with NVSwitch, offloading allreduce data movement to copy engines frees SMs entirely for computation. This is conceptually simple but requires custom communication code bypassing NCCL for some operations. Estimated savings: 1–2 seconds. High complexity.

---

## Fused GEMM epilogues could eliminate 22+ kernel launches per step

Beyond the already-known fused residual+norm and fused lm_head+CE, there's an unexploited fusion opportunity: **attention output projection + residual add + next layer's norm** in a single CUTLASS epilogue. Currently these are three separate kernel launches per layer — the GEMM for the output projection, an element-wise add for the residual, and a reduction for LayerNorm. The residual and norm operations are memory-bound; fusing them into the GEMM epilogue eliminates two round-trips to HBM per layer.

CUTLASS supports custom epilogue operations that can perform arbitrary element-wise and reduction operations after the GEMM mainloop. The fused `LayerNorm(residual + GEMM_output)` would be a custom epilogue computing the sum, mean, and variance in-register before writing normalized output. Across 11 layers × 2 directions (forward + backward), this eliminates ~44 kernel launches per step.

A related fusion target is **GEMM + ReLU²** for the MLP layers. cuBLASLt supports epilogue fusions like GEMM+GELU, but ReLU² (squared ReLU) requires a custom epilogue. The Triton max-autotune mode may already attempt this, but manually optimized CUTLASS kernels for the specific 768-dim shapes could do better.

**Estimated savings:** 1–2 seconds from eliminated kernel launches and reduced memory traffic. **Complexity:** Medium — requires CUTLASS epilogue programming or Triton template modifications. **Risk:** May conflict with existing torch.compile fusions.

---

## TMA descriptor pre-computation and other per-step overhead reductions

Several smaller optimizations target the **per-step fixed overhead** that compounds across 1490 steps:

**TMA descriptor pre-computation:** Triton passes TMA descriptors via global memory with H2D copies per kernel launch. CUTLASS passes them by-value as kernel parameters, avoiding this overhead. Pre-computing all TMA descriptors at initialization and reusing them eliminates a small but per-launch cost. With hundreds of kernels per step, this saves **~0.5–1 second** total. The fix is straightforward if CUDA graphs are used (descriptors set up once); otherwise requires manual descriptor management.

**Lambda parameter count alignment:** The modded-nanogpt PR #140 revealed that ensuring scalar parameter counts are multiples of **64** (8 GPUs × 8 parameters each) enables coalesced memory access in the optimizer. When counts were 56 or 72, runtime increased meaningfully versus 64. Ensuring all small parameter groups have aligned counts is trivial and saves **~0.5–1 second**.

**Graph break elimination:** Using `TORCH_LOGS=graph_breaks` to identify and fix all torch.compile graph breaks enables full-graph compilation. Each break introduces Python fallback and prevents cross-break fusion. The heterogeneous batch size schedule likely causes breaks if implemented with Python conditionals. Refactoring to static control flow could save **0.3–1 second**.

---

## Progressive layer dropping saves compute in early training

**Switchable-BERT Layer Dropping (SBLD)** (AAAI 2023) progressively drops transformer layers during early training using Bernoulli sampling with layer-wise sensitivity-guided drop rates. The insight: deeper layers contribute less during early training when representations are still forming. SBLD claims **19.67% end-to-end training time reduction** for GPT-3 Medium with +1.65% accuracy improvement.

For the modded-nanogpt 11-layer model, a schedule that drops 2–3 MLP layers during the first third of training (while the attention windows are still small and learning is focused on local patterns) and progressively enables all layers could save significant FLOPs. The current codebase already drops the first MLP layer entirely, validating the principle. Going further — temporarily dropping 2–3 additional MLPs during warmup — could compound the benefit.

**Estimated savings:** 2–4 seconds from reduced FLOPs in early training. **Complexity:** Medium — need to implement switchable blocks with a drop schedule and validate convergence. **Risk:** With only 1490 steps, aggressive dropping during the first 500 steps means those layers have only 990 steps to learn, which may be insufficient. The already-dropped MLP-1 suggests diminishing returns. **Leaderboard compliance:** Yes, requires p<0.01 significance.

---

## Speculative but high-ceiling ideas for further exploration

**Thread block clusters with TMA multicast** could reduce HBM bandwidth pressure by loading shared weight tiles once and multicasting them across cluster CTAs via the SM-to-SM network. For 768-dim weight matrices shared across batch tokens, multicast could cut memory traffic by up to the cluster size (8×). However, Triton does not support thread block clusters in mainline — only the experimental TLX fork. Implementation requires CUDA C++ and is extremely complex. Estimated ceiling: 10–20% on bandwidth-bound kernels.

**Megakernel architecture** (from Hazy Research's "Look Ma, No Bubbles!") fuses the entire forward pass into a single persistent kernel with an on-GPU instruction interpreter. This eliminates all kernel launch overhead and memory-bandwidth bubbles between operations. Adapting this for backward passes would be the hard part. Estimated ceiling: 3–6 seconds, but implementation complexity is extreme.

**Weight sharing → untying**: train the model with weight sharing across pairs of layers initially (reducing effective depth to 5-6 unique layers), then untie at an automatically determined point based on gradient divergence statistics. An OpenReview paper claims 50% training time reduction for BERT. For 11 layers with existing weight tying between embedding/lm_head (untied at 2/3 of training), extending this to hidden layers is architecturally natural but risky at 768-dim where capacity is already limited.

**OT regularization**: a plug-and-play optimal transport regularization term reported 46% final loss reduction and 42% parameter count reduction on character-level nanoGPT. If these results transfer to token-level training, the convergence improvement would be massive. However, the original experiments were on character-level models, which have very different loss landscapes. Worth a quick experiment.

---

## Conclusion

The modded-nanogpt speedrun at 88.1 seconds is already heavily optimized, but three categories of untapped opportunity remain. **On the systems side**, the 768-dim model suffers from wave quantization (Stream-K fix), Python dispatch overhead (cpp_wrapper fix), and wasted embedding optimizer compute (SparseAdam fix) — collectively worth an estimated **4–9 seconds** with moderate implementation effort. **On the optimizer side**, MARS-M variance reduction and EMA weight averaging are the strongest candidates, potentially worth **3–8 seconds in reduced step count**. **On the kernel side**, CUTLASS warp-specialized ping-pong kernels and fused GEMM epilogues offer another **2–5 seconds** but require significant engineering. The single easiest experiment is enabling `cpp_wrapper = True` — a one-line change that Intel benchmarks suggest yields 5–17% on small BF16 workloads. The highest-ceiling single technique is Stream-K GEMM scheduling, which directly addresses the fundamental problem that 768-dim matmuls produce fewer tiles than the H100 has SMs.