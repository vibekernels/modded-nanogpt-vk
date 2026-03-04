# Research Log

Experiments and optimization attempts for modded-nanogpt-vk. All runs on 1xH100 SXM 80GB (RunPod) unless noted.

**Current baseline:** val_loss 3.2802, train_time 668s (`94639dc` on master)

---

## Turbo-Muon AOL Preconditioning

**Date:** 2026-03-04
**Branch:** `turbo-muon-aol` (`a1bf7d8`)
**Status:** No improvement

Replaced 5-iteration Polar Express Newton-Schulz with 4-iteration [Turbo-Muon](https://arxiv.org/abs/2512.04632) AOL (Almost-Orthogonal Layer) diagonal preconditioning from [flash-newton-schulz](https://github.com/thib-s/flash-newton-schulz).

**Hypothesis:** AOL rescales the gradient matrix using Gram matrix row norms (`s = rsqrt(abs(A).sum(dim=-1))`), tightening the initial singular value distribution so 4 NS iterations suffice instead of 5. Expected 3-6s savings (~10-20% of NS compute eliminated).

**Changes:**
- Removed global spectral normalization (`X / (X.norm() * 1.02)`)
- Added AOL preconditioning on first iteration (both tall and wide matrix paths)
- Swapped Polar Express 5-iter coefficients for Turbo-Muon 4-iter coefficients: `(3.9505, -6.3029, 2.6377), (3.7418, -5.5913, 2.3037), (2.8769, -3.1427, 1.2046), (2.8366, -3.0525, 1.2012)`

**Results:**

| | Val Loss | Train Time | Step Avg (stage 1) |
|---|----------|------------|-------------------|
| Baseline | 3.2802 | 668s | — |
| Turbo-Muon AOL | 3.2785 | 681.9s | 262.4ms |

**Outcome:** +13.9s slower (2.1%). Loss is marginally better but within noise. The per-step AOL overhead (abs, sum, rsqrt, two broadcast multiplies on the Gram matrix) likely exceeds the savings from removing one fused Triton NS iteration. On these small matrices (768x768 Gram), materializing new tensors for diagonal scaling dominates.

**Possible follow-ups:**
- Fuse AOL preconditioning into the existing XTX/XXT Triton kernels to eliminate overhead
- Try 5 iterations + AOL (better convergence without time penalty)
- Try AOL only for the MLP bank (larger matrices where overhead is amortized)

---

## Turbo-Muon AOL 5-iter (same iterations, better init)

**Date:** 2026-03-04
**Branch:** `turbo-muon-aol-5iter` (`26fa27b`)
**Status:** No improvement

Keep 5 NS iterations but use AOL preconditioning + Turbo-Muon coefficients instead of Polar Express coefficients. Tests whether AOL improves convergence quality without reducing iteration count.

**Hypothesis:** AOL provides a better starting point. With 5 iterations (same as baseline), the extra preconditioning quality should improve convergence, yielding a lower loss for the same compute.

**Changes:**
- Same AOL preconditioning as above (removes spectral normalization)
- Full 5-iteration Turbo-Muon coefficients: `(4.0848, -6.8946, 2.9270), (3.9505, -6.3029, 2.6377), (3.7418, -5.5913, 2.3037), (2.8769, -3.1427, 1.2046), (2.8366, -3.0525, 1.2012)`

**Results:**

| | Val Loss | Train Time | Step Avg (stage 1) |
|---|----------|------------|-------------------|
| Baseline | 3.2802 | 668s | — |
| AOL 4-iter (prev) | 3.2785 | 681.9s | 262.4ms |
| AOL 5-iter | 3.2822 | 680.2s | 265.9ms |

**Outcome:** +12.2s slower (1.8%), loss slightly worse than baseline (3.2822 vs 3.2802). The Turbo-Muon coefficients with 5 iterations don't improve convergence over Polar Express. The overhead of AOL preconditioning (~3.5ms/step in stage 1 vs the 4-iter variant) matches the expected cost of one additional NS iteration. Convergence quality is no better — Polar Express's Remez-optimized coefficients may simply be better-suited for this workload.

---

## Fused Triton AOL + 4-iter

**Date:** 2026-03-04
**Branch:** `turbo-muon-aol-fused` (`d6621e4`)
**Status:** No improvement (fixed NaN, still slower than baseline)

Fuse AOL preconditioning into dedicated Triton kernels to eliminate PyTorch dispatch overhead. Four kernels: `_aol_prescale_A_kernel` (compute row sums + rsqrt), `_aol_scale_A_kernel` (scale Gram matrix), `_aol_scale_X_{cols,rows}_kernel` (scale X). Combined with 4-iter Turbo-Muon coefficients.

**Hypothesis:** The unfused AOL's overhead (5 PyTorch ops) dominates the savings from removing one NS iteration. Fusing into Triton eliminates dispatch overhead, making the 4-iter approach net positive.

**Changes:**
- Added 4 new Triton kernels in `triton_kernels.py` for fused AOL
- Restructured `polar_express` loop: compute Gram matrix once for AOL before iteration loop
- 4-iter Turbo-Muon coefficients (same as first experiment)

**Bug fixes (2 iterations):**
1. Initial version stored scaling vector `s` in bf16 → NaN divergence. Fixed by using fp32 for all `s` storage.
2. `_aol_prescale_A_kernel` used `tl.zeros([1], ...)` accumulator, creating a block/scalar mismatch on `tl.store`. Under `torch.compile`, `identify_mutated_tensors` failed, causing incorrect codegen. Fixed by using a `[BLOCK_K]` accumulator with `tl.where` masking + `tl.sum` reduction to produce a proper scalar. Also added `clamp(max=1e4)` matching the non-fused version.

**Results:**

| | Val Loss | Train Time | Step Avg (stage 1) |
|---|----------|------------|-------------------|
| Baseline | 3.2802 | 668s | — |
| Fused AOL 4-iter (bf16 s) | **NaN** | 637.2s | 256.2ms |
| Fused AOL 4-iter (fp32 s, fixed) | 3.2808 | 678.4s | 258.0ms |

**Outcome:** After fixing the Triton kernel, training converges correctly (val_loss 3.2808, essentially matching baseline). The fused approach is ~3.5s faster than unfused AOL (678s vs 682s) but still ~10s slower than baseline (668s). The AOL preconditioning overhead, even when fused into Triton, exceeds the savings from removing one NS iteration on these matrix sizes.

**Key learnings:**
- Triton `tl.zeros([1])` creates a block type that fails under `torch.compile` when stored to a scalar pointer — use `tl.zeros([BLOCK_K])` + `tl.sum` instead
- Scaling vectors must be stored in fp32 even when the matrices are bf16
- On small Gram matrices (768x768), even a single Triton kernel launch for preconditioning costs more than it saves by eliminating one fused NS iteration

---

## Swizzled Tile Ordering for MLP GEMM Kernel

**Date:** 2026-03-04
**Branch:** `triton-swizzle-mlp` (`b2334db`)
**Status:** No improvement

Added `tl.swizzle2d` grouped tile ordering to the `linear_relu_square_kernel` (MLP GEMM) and increased `GROUP_SIZE_M` from 1 to 8. The symmetric matmul kernels (XXT, XTX, ba_plus_cAA) already used swizzling; this was the only GEMM-like kernel without it.

**Hypothesis:** Swizzled tile ordering makes consecutive thread blocks process spatially adjacent tiles, improving L2 cache hit rates for weight matrix columns. Expected 0.7–1.5s savings across 1490 steps.

**Changes:**
- Added `tl.swizzle2d(pid_m, pid_n, num_pid_m, num_pid_n, GROUP_SIZE_M)` after both pid_m/pid_n computations in the persistent kernel loop
- Changed `GROUP_SIZE_M=1` to `GROUP_SIZE_M=8` in the kernel launch

**Results:**

| | Val Loss | Train Time | Step Avg (stage 1) |
|---|----------|------------|-------------------|
| Baseline | 3.2802 | 668s | ~265ms |
| Swizzled MLP | 3.2814 | 675.9s | 258.7ms |

**Outcome:** +7.9s slower overall despite stage 1 being ~6ms/step faster (258.7 vs ~265). The kernel already uses a persistent launch pattern (`grid = min(NUM_SMS, num_tiles)` with `tl.range` loop), which inherently provides good SM utilization. Swizzling within the persistent loop may actually hurt by breaking the sequential tile access pattern that TMA descriptors optimize for. The overhead likely comes from later stages where the batch size grows and the swizzle pattern interacts poorly with the TMA prefetch pipeline.

**Key learnings:**
- Persistent kernels with TMA descriptors may not benefit from swizzled tile ordering — TMA's hardware prefetch already handles memory access scheduling
- The `GROUP_SIZE_M=1` was likely intentional for this kernel's persistent + TMA design
- Swizzling is most beneficial for non-persistent kernels with explicit pointer arithmetic (like the symmetric matmul kernels that already use it)

---

## Inductor cpp_wrapper Mode

**Date:** 2026-03-04
**Branch:** N/A (tested on RunPod only)
**Status:** Failed (crash)

Enabled `torch._inductor.config.cpp_wrapper = True` to replace Python wrapper code managing kernel launches with compiled C++ code calling `cuLaunchKernel` directly, bypassing the Python interpreter and GIL between every kernel.

**Hypothesis:** Intel benchmarks show 5-17% speedup for BF16 small workloads (iteration time <=40ms). Our 768-dim model has many short kernels where Python dispatch overhead is a measurable fraction of total time. Expected 1-3s savings.

**Changes:**
- Single line: `torch._inductor.config.cpp_wrapper = True` after line 34

**Results:**

| | Val Loss | Train Time |
|---|----------|------------|
| Baseline | 3.2792 | 680s |
| cpp_wrapper | CRASH | N/A |

**Outcome:** Crashed during compilation with `RuntimeError: Failed to run autotuning code block: invalid decimal literal (<string>, line 7077)`. The C++ codegen is incompatible with this codebase's custom FP8 operators (`mm_t_op`, `mm_op`) and Triton kernel autotuning blocks. The cpp_wrapper's code generation produces invalid C++ for the complex autotuning paths used by FP8 `_scaled_mm` and custom Triton kernels.

**Possible follow-ups:**
- May work in a future PyTorch version with better cpp_wrapper + custom op support
- Could selectively apply cpp_wrapper only to sub-graphs that don't use FP8 ops (not currently supported)

---

## EMA Weight Averaging

**Date:** 2026-03-04
**Branch:** N/A (tested on RunPod only)
**Status:** No improvement

Maintained an exponential moving average of model parameters during training, swapping in EMA weights for validation. Tested decay rates of 0.995 and 0.99.

**Hypothesis:** EMA smooths out parameter oscillations, effectively replacing the LR cooldown phase. Could allow all steps to train at high LR while EMA captures the underlying progress. Expected 30-75 fewer effective steps needed (1.5-3.5s savings).

**Changes:**
- After warmup: `ema_state = {k: v.clone() for k, v in model.state_dict().items()}`
- After each optimizer step: `ema_state[k].lerp_(v, 1 - ema_decay)`
- Swap EMA weights in for validation, restore live weights after

**Results:**

| | Val Loss | Train Time |
|---|----------|------------|
| Baseline (this pod) | 3.2792 | 680.3s |
| EMA decay=0.995 | 3.4500 | 679.4s |
| EMA decay=0.99 | 3.3456 | 680.4s |

**Outcome:** Both decay rates produced significantly worse val loss. The 0.995 decay over-smooths dramatically (+0.17 loss), while 0.99 is better but still +0.066 above baseline. The ~8ms/step overhead from state_dict copies and lerp updates is modest but adds up.

The fundamental issue: EMA conflicts with the aggressive multi-stage training schedule. The codebase uses sharp batch size transitions (8->16->24), MTP head pruning, embed/lm_head splitting at 2/3, and YaRN window extensions. EMA weights lag behind these transitions, blending pre-transition and post-transition parameters in a way that hurts rather than helps. The existing 60% linear cooldown is already well-tuned for these dynamics.

**Possible follow-ups:**
- EMA only during the cooldown phase (not the full run) to avoid transition interference
- Per-parameter-group EMA with different decay rates for banks vs scalars
- LAWA (Latest Weight Averaging) — average last K checkpoints instead of exponential

---

## Cosine Cooldown Shape

**Date:** 2026-03-04
**Branch:** N/A (tested on RunPod only)
**Status:** No improvement (neutral)

Replaced the linear LR cooldown with cosine decay: `t_cos = 0.5 * (1 - cos(pi * t))` providing slower initial decay and faster decay at the end.

**Hypothesis:** Different cooldown shapes create distinct bias-variance tradeoffs. Cosine cooldown maintains higher LR longer in the early cooldown phase, allowing more learning before rapid convergence at the end. Expected 10-30 step equivalent gain (0.5-1.5s).

**Changes:**
- In `get_lr()`: replaced `lr = lr * (1 - t) + 0.15 * t` with cosine-shaped interpolation

**Results:**

| | Val Loss | Train Time |
|---|----------|------------|
| Baseline (this pod) | 3.2792 | 680.3s |
| Cosine cooldown | 3.2841 | 680.9s |

**Outcome:** Val loss +0.005 worse, well within run-to-run noise. Training time identical (as expected — no compute change). The linear cooldown with 60% fraction is already well-optimized for this codebase. The cosine shape's slower initial decay means higher LR for longer, which doesn't help when the batch size schedule already provides the right effective LR progression.

**Possible follow-ups:**
- Try sqrt cooldown (`t_sqrt = sqrt(t)`) — faster initial decay, may help stabilize the large-batch stage 3
- Try 1-sqrt cooldown — even slower initial decay than cosine
- Cooldown fraction tuning (55% vs 60% vs 65%) may matter more than the shape

---

<!-- Template for new experiments:

## Experiment Name

**Date:** YYYY-MM-DD
**Branch:** `branch-name` (`commit`)
**Status:** Improvement / No improvement / Inconclusive

Brief description of what was tried and why.

**Hypothesis:** What we expected to happen.

**Changes:**
- Change 1
- Change 2

**Results:**

| | Val Loss | Train Time |
|---|----------|------------|
| Baseline | 3.2802 | 668s |
| This run | X.XXXX | XXXs |

**Outcome:** Summary of what happened and why.

**Possible follow-ups:**
- Ideas for future work

---
-->
