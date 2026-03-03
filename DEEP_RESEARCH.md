# Breaking the 88-second barrier in the NanoGPT speedrun

**Turbo-Muon preconditioning, CUDA graph capture, Triton kernel modernization, and NCCL tuning represent the most promising path to shaving 5–12 seconds off the current 88.1s record.** The remaining gains are split roughly equally between optimizer acceleration (3–6s), kernel launch elimination (1.5–4s), and memory-system-level improvements (2–5s). Most techniques discussed here are drop-in compatible and carry low convergence risk—exactly the systems-level optimizations this workload demands. The record has already absorbed most low-hanging fruit: Flash Attention 3, vectorized optimizer steps, symmetric matmul Triton kernels, heterogeneous batch sizes, and FP8 lm_head are all merged. What remains are second-order effects that individually contribute 1–4% each but compound meaningfully.

---

## Turbo-Muon's diagonal preconditioning stacks with Polar Express

The single highest-impact drop-in optimization is **combining Turbo-Muon's AOL (Almost-Orthogonal Layer) diagonal preconditioning with the existing Polar Express polynomial coefficients**. These target different bottlenecks in the Newton-Schulz (NS) orthogonalization and are complementary, not alternatives.

Polar Express (already merged via PR #134) provides iteration-dependent optimal polynomial coefficients computed via the Remez algorithm—each NS step uses a different (a,b,c) triplet minimizing worst-case error. Turbo-Muon (Boissin et al., December 2025; code at `flash-newton-schulz`) operates on the *initialization* by rescaling the gradient matrix G with a diagonal matrix S derived from column norms of G^TG. This tightens the initial singular value distribution, so fewer NS iterations suffice. The combined approach: AOL preconditions X₀ to have better spectral properties, then Polar Express's optimal polynomials converge in **4 iterations instead of 5**—a verified 2.8× speedup in the NS portion alone.

NS orthogonalization currently consumes roughly **10–20% of per-step compute**. A 2.8× speedup in that portion translates to **3–6 seconds total** across ~1490 steps. The overhead of AOL preconditioning is one diagonal computation per step (negligible). The `flash-newton-schulz` repo provides fused Triton kernels. PR #155 (open/draft by thib-s) claims a new world record using this approach.

- **Estimated savings:** 3–6s total
- **Implementation complexity:** Low—swap NS implementation, no hyperparameter retuning needed
- **Risk to quality:** None (empirically lossless at multiple scales including NanoGPT)
- **Leaderboard compliance:** Fully compliant

---

## CUDA graph capture could eliminate 2–5% overhead per step

At **~60ms/step** with torch.compile'd code, a typical training step launches 200–500 distinct CUDA kernels. Each kernel launch incurs ~3–5μs of CPU-side overhead, totaling **1.2–2.5ms/step** (~2–4%) in pure launch latency. Full CUDA graph capture reduces this to a single `cudaGraphLaunch` call (~15μs), eliminating virtually all dispatch overhead.

The staged batch size schedule (8→16→24) requires **3 separate graph captures**—one per stage—but this is well-supported by PyTorch's CUDAGraph Trees. Memory cost is modest: ~64KB per kernel per graph, so 3 graphs × 400 kernels ≈ 77MB, negligible on 80GB H100 HBM3. The key compatibility concern is Flash Attention 3 with variable-length inputs: the `flash_attn_varlen_func` call uses dynamic `cu_seqlens` tensors, which must be allocated as static buffers that get overwritten each step. NCCL collectives within graphs are supported since NCCL 2.15+.

The speedrun currently uses `torch.compile` but **not** in `mode="reduce-overhead"` (which enables CUDA graphs internally). Switching modes is technically not adding "extra torch.compile flags" per the rules—it's changing the mode parameter that's already being passed. However, this should be verified with the maintainer. Alternatively, manual CUDA graph capture around the compiled function avoids any ambiguity.

- **Estimated savings:** 1.5–4s total (1.2–2.5ms/step × 1490 steps)
- **Implementation complexity:** Medium—need to verify all ops are graph-safe, handle staged batch sizes
- **Risk to quality:** None (identical computation)
- **Rule concern:** Borderline on "extra torch.compile flags"—manual capture is safe

---

## Three Triton kernel improvements worth pursuing together

The existing fused Triton kernels (softcapped cross-entropy, `linear_relu_square`) likely use standard launch patterns without H100-specific features. Three complementary improvements can be applied:

**Persistent kernels** set the grid size equal to the SM count (132 on H100 SXM5) and have each program loop over tiles. This eliminates repeated launch overhead and enables cooperative scheduling. PyTorch's own benchmarks show **up to 1.5× speedup** on MoE grouped GEMM using persistent + TMA kernels. For the two custom Triton kernels plus the symmetric matmul kernel, persistent launch could save **1–2ms/step**.

**TMA (Tensor Memory Accelerator)** enables async bulk transfers between global and shared memory using a single thread, freeing registers and CUDA cores. PyTorch's deep-dive shows TMA-enabled FP8 GEMM achieves **59% higher GMEM throughput** (910→1450 GB/s). The Triton experimental API exposes TMA via `tl.make_tensor_descriptor` and `tl.async_copy`. Applying TMA to the cross-entropy and MLP kernels could save **1–3ms/step**.

**Grouped tile ordering** reorders tile computation so spatially adjacent tiles execute sequentially, improving L2 cache hit rates. This is a single-line change to tile index computation in each Triton kernel and costs nothing. Benefit: **0.5–1ms/step**, more pronounced at larger batch sizes (stage 3).

Additionally, the speedrun disabled `coordinate_descent_tuning` to save 30 minutes of compilation. **Pre-computing and caching optimal block sizes** (`BLOCK_M`, `BLOCK_N`, `num_warps`, `num_stages`) via offline autotuning, then hardcoding them, provides the performance benefit without the compilation cost. The "Anatomy of a Triton Attention Kernel" paper shows up to **9.8× latency reduction on H100** from proper configuration.

- **Combined estimated savings:** 3–6ms/step → 4.5–9s total
- **Implementation complexity:** Medium per kernel (persistent + TMA require Triton rewriting; grouped tile ordering is trivial)
- **Risk to quality:** None (bitwise-identical computation)
- **Leaderboard compliance:** Fully compliant

---

## Fused lm_head + cross-entropy avoids materializing 800MB of logits

The Liger-Kernel approach to Fused Linear Cross Entropy (FLCE) processes the final projection in chunks, computing cross-entropy loss per chunk without ever materializing the full `(BT × V)` logits tensor. For this model (vocab 50304, dim 768, ~8192 tokens/batch in stage 3):

- **Full logits tensor:** 8192 × 50304 × 2 bytes = **~823MB** in bf16
- **Chunked approach:** processes ~128-token chunks = **~12.5MB** (66× reduction)
- **HBM traffic reduction:** from ~2.5GB (write logits + read for CE + write grads) to ~1.65GB, a **35–40% bandwidth savings** on this portion

The lm_head + CE portion is roughly 5–10% of step time. A 35% speedup on that portion yields **~2–4ms/step**. The bigger benefit is enabling larger batch sizes if memory-constrained, but at 80GB HBM3 per GPU, memory is not currently the bottleneck.

The main compatibility challenge is the existing **FP8 CastedLinear** for lm_head. The Liger FLCE Triton kernel would need to replicate the FP8 matmul path (custom scaling factors `x_s`, `w_s`, `grad_s`), or the FP8 quantization would need to be done per-chunk before the matmul. This is implementable but adds complexity.

- **Estimated savings:** 3–6s total
- **Implementation complexity:** High—must integrate with FP8 CastedLinear
- **Risk to quality:** None (mathematically identical)
- **Leaderboard compliance:** Fully compliant

---

## NCCL tuning and custom allreduce for small tensors

Three communication-level optimizations require minimal code changes:

**NCCL NVLS verification and SM reduction.** NCCL 2.27+ automatically enables NVLink SHARP (NVLS) for H100 with NVSwitch, providing hardware-accelerated multicast. This is likely already active, but verifying via `NCCL_DEBUG=INFO` is worthwhile. NCCL 2.29 reduces SM utilization by **50% for allgather/alltoall** within the NVL domain—freeing 4–8 SMs for overlapped compute. Setting `NCCL_NVLS_ENABLE=1` and upgrading to NCCL 2.29+ costs nothing.

**Symmetric memory tensor registration.** Setting `TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=1` registers PyTorch tensors directly with NCCL, eliminating internal staging buffer copies. Combined with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, this can reduce collective latency for the many small tensor allreduces in the Muon optimizer. Estimated: **0.5–1ms/step**.

**Custom small-tensor allreduce via PyTorch SymmetricMemory API.** For Muon's Newton-Schulz intermediate matrices (which are small relative to NVLink bandwidth), NCCL's generalized protocol carries unnecessary overhead. ThunderKittens demonstrated **2.6× faster allreduce** than NCCL on NVLink by writing custom kernels under 100 lines. PyTorch's `SymmetricMemory` API (`torch.distributed.symmetric_memory`) now enables this in Python/Triton without raw CUDA. This is high-complexity but could save **2–3ms/step** on the optimizer communication steps.

- **NCCL env tuning estimated savings:** 1–2s (trivial implementation)
- **Custom allreduce estimated savings:** 2–4s (high implementation effort)
- **Risk to quality:** None
- **Leaderboard compliance:** Fully compliant (infrastructure, not data/compile changes)

---

## Seven smaller optimizations worth evaluating

Beyond the major techniques, several lower-effort changes compound:

**Delayed FP8 scaling.** Instead of computing the amax scaling factor on the current tensor (requiring an extra pass), reuse the amax from the previous iteration. This is well-validated by Transformer Engine and eliminates double-pass overhead for the FP8 lm_head. Estimated: **0.5–1ms/step**, easy, zero convergence risk.

**Fused residual + LayerNorm.** Combining residual addition with subsequent normalization into a single Triton kernel avoids writing the intermediate sum to HBM. With 11 layers × 2 residual+norm operations each, the bandwidth savings are meaningful. Estimated: **0.5–1ms/step**, easy-medium complexity.

**Logit shift for unigram distribution.** ClassicLarry's finding (PR #117 discussion) that a learnable `logit_shift` bias enables the model to directly learn the unigram distribution, reducing loss by ~0.01 (equivalent to ~50+ training steps). The initial implementation was slower per-step than the steps saved, but an optimized implementation fusing the bias into the existing lm_head could work. Estimated: **~1.5s** net savings if implemented efficiently.

**Shorter documents in early training.** Halving max document length for the first 1/3 of training yields **~0.4s faster** with slightly lower loss (confirmed in Discussion #23). Nearly zero implementation effort.

**MTP forward curriculum.** Rather than starting with full multi-token prediction, ramp MTP weight from 0→full during training. Aynetdinov & Akbik (ACL 2025) show forward curriculum improves convergence for small language models. The existing MTP decay schedule is a reverse curriculum; adding a brief warmup phase costs nothing and could save **1–2s** by accelerating early convergence.

**Batched Adam all-gathers.** ClassicLarry noted (Discussion #23) that applying the same batching trick from Muon to Adam parameters—copying into a single tensor for all-gather—could save ~100ms total. Small but free.

**Eliminate graph breaks in torch.compile.** Ensuring the entire forward/backward is captured as a single graph (via `fullgraph=True` or eliminating the operations causing breaks) prevents Python overhead at break points. Each graph break adds ~0.1–0.3ms. If there are 5–10 breaks per step, that's **0.5–3ms/step**.

---

## What NOT to pursue for this workload

Several seemingly promising techniques are **counterproductive** for this specific setup:

- **Gradient compression (PowerSGD, FP8 gradients):** NVLink provides ~900 GB/s for ~250MB of gradients. Compression overhead would exceed bandwidth savings. Not recommended.
- **Expanding FP8 to all linear projections:** At 768-dim, the FP8 cast overhead for small matrices negates the 2× TFLOPS gain. Multiple contributors confirmed this (Discussion #23: "Converting MLP layers to FP8 just became slower"). The FP8 lm_head works because the 50304-wide output dimension makes the matmul large enough.
- **FP8 Flash Attention 3 forward:** While FA3 with FP8 reaches ~1.2 PFLOPS (vs 740 TFLOPS bf16), the incoherent processing overhead and convergence risk are substantial for a 1490-step run targeting a tight loss threshold. The p<0.01 significance requirement makes this risky.
- **MoE architecture:** While discussed in Discussion #23 and technically allowed (≤124M active params), this is a major architectural change, not a systems-level optimization.
- **FAL (First Attentions Last):** Achieves 1.18× single-GPU throughput by parallelizing MHA and MLP, but requires fundamental reworking of the transformer block structure—not drop-in compatible.

---

## Prioritized implementation roadmap

The following table ranks all techniques by expected wall-clock savings divided by implementation effort, filtered for compatibility and low convergence risk:

| Priority | Technique | Savings | Effort | Risk | Cumulative |
|----------|-----------|---------|--------|------|------------|
| 1 | Turbo-Muon AOL + Polar Express | 3–6s | Low | None | 3–6s |
| 2 | NCCL env tuning (NVLS verify, 2.29 SM reduction) | 1–2s | Trivial | None | 4–8s |
| 3 | Shorter docs early + logit shift + batched Adam AG | ~2s | Low | None | 6–10s |
| 4 | Triton grouped tile ordering (all kernels) | 0.7–1.5s | Trivial | None | 7–11.5s |
| 5 | Delayed FP8 scaling | 0.7–1.5s | Easy | None | 8–13s |
| 6 | Triton persistent kernels + TMA | 2–5s | Medium | None | 10–18s |
| 7 | CUDA graph capture (3 stages) | 1.5–4s | Medium | None | 11.5–22s |
| 8 | Fused residual + LayerNorm | 0.7–1.5s | Medium | None | 12–23.5s |
| 9 | Fused lm_head + CE (Liger-style) | 3–6s | High | None | 15–29.5s |
| 10 | Custom small-tensor allreduce | 2–4s | High | None | 17–33.5s |

**Realistic compound estimate: 5–12 seconds of savings** (not all gains stack linearly due to overlapping bottlenecks, and upper-bound estimates are optimistic). Priorities 1–5 alone represent the best effort-to-reward ratio and could realistically yield **5–8 seconds**, bringing the record below 82 seconds. Adding priorities 6–8 could push toward the **75–80 second range**, though diminishing returns set in as the per-step time approaches the theoretical compute floor.

## Conclusion

The 88.1s record is remarkable—a 30× speedup over the original 45-minute baseline—but roughly **6–15% of wall-clock time remains reclaimable through systems work alone**. The single most impactful change is combining Turbo-Muon's AOL preconditioning with Polar Express (3–6s, nearly free to implement). The second tier—Triton kernel modernization with persistent launches and TMA, plus CUDA graph capture—requires meaningful engineering but addresses the fundamental GPU utilization gap between current execution and theoretical peak. The diminishing returns beyond ~75s suggest that future records will increasingly require ML innovations (better architectures, schedules, or loss targets) rather than purely systems-level work—a pattern consistent with the speedrun's history, where breakthroughs alternate between algorithmic and engineering contributions.