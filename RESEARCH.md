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

## Fix CWD Mask in NorMuon Path (>= vs >)

**Date:** 2026-03-04
**Branch:** N/A (tested on RunPod only)
**Status:** No improvement (neutral)
**Source:** [PR #172](https://github.com/KellerJordan/modded-nanogpt/pull/172) — ClassicLarry

The Adam CWD path uses `>` (correct — zero gradients don't trigger weight decay). The NorMuon CWD path in `_cautious_wd_and_update_inplace` uses `>=`, which means parameters with zero gradients still get weight decay applied.

**Hypothesis:** For sparse embeddings processed by NorMuon, `>=` drives rarely-seen token embeddings toward zero. Fixing to `>` should preserve rare token embeddings better. Expected 0.3-0.9s from better embedding quality.

**Changes:**
- Line 922: `mask = (grad * p_precise) >= 0` changed to `mask = (grad * p_precise) > 0`

**Results:**

| | Val Loss | Train Time |
|---|----------|------------|
| Baseline (this pod) | 3.2789 | 681.7s |
| CWD >= to > fix | 3.2796 | 682.4s |

**Outcome:** Val loss +0.0007, well within noise. No measurable effect. The NorMuon path primarily handles the large weight banks (`attn_bank`, `mlp_bank`), which don't have sparse gradients — they always receive dense gradient updates. The sparse embedding parameters are handled by Adam (which already uses `>`), so the fix has minimal impact.

---

## Gate Learning Rate Multiplier Tuning

**Date:** 2026-03-04
**Branch:** N/A (tested on RunPod only)
**Status:** Possible small improvement at lr_mul=5.0
**Source:** [PR #117](https://github.com/KellerJordan/modded-nanogpt/pull/117), [PR #146](https://github.com/KellerJordan/modded-nanogpt/pull/146) — varunneal, YouJiacheng

Added `lr_mul` to `attn_gate_bank` and `ve_gate_bank` in param_table. These gates currently inherit the default Adam LR (0.008). YouJiacheng found 0.1x was critical for the 2.92 track under Muon, but they were never re-tuned after moving to Adam.

**Hypothesis:** Gates may learn too slowly at the default LR, or the optimal rate under Adam differs from the Muon-era setting.

**Changes:**
- Added `"lr_mul": X` to both `attn_gate_bank` and `ve_gate_bank` entries

**Results:**

| | Val Loss | Train Time |
|---|----------|------------|
| Baseline (this pod) | 3.2789 | 681.7s |
| lr_mul=0.1 | 3.2800 | 679.2s |
| lr_mul=5.0 | **3.2771** | 684.3s |

**Outcome:** lr_mul=0.1 (slower gates) was neutral. lr_mul=5.0 (faster gates) showed -0.0018 improvement — small but the intermediate checkpoints were also consistently better than baseline at every validation point. The faster gate learning helps the model more quickly identify which heads should attend and which should be suppressed. The 5.0x multiplier matches the `scalars` and `resid_lambdas` lr_mul values, suggesting these small control parameters benefit from faster learning.

**Possible follow-ups:**
- Test lr_mul=10.0 to see if further acceleration helps
- Combine with other improvements once a clear win is established
- Test on 8xH100 to see if effect holds at the competition configuration
- Statistical significance: needs multiple runs to confirm -0.0018 is real

---

## GPT-OSS Attention Sinks (Simplified)

**Date:** 2026-03-04
**Branch:** N/A (tested on RunPod only)
**Status:** No improvement / Implementation blocked
**Source:** [PR #117](https://github.com/KellerJordan/modded-nanogpt/pull/117) — ClassicLarry, byronxu99

GPT-OSS sinks multiply attention output by `sigmoid(LSE - learned_sink)`, equivalent to adding a learned null key to the softmax denominator. This lets heads express "I found nothing useful" independently of the existing sparse gate.

**Hypothesis:** The sparse gate and sinks are complementary — gate decides "should I look?", sinks decide "did I find anything?". Expected 0.6-1.8s from combining both.

**Implementation challenges:**
1. Flash Attention 3's `flash_attn_varlen_func` wrapper doesn't expose `return_softmax_lse` — the internal `FlashAttnVarlenFunc` supports `return_softmax=True` but the wrapper maps it via `return_attn_probs` which conflicts with `torch.compile` tracing
2. Adding kwargs to the attention function's `forward()` breaks `torch.compile` (dynamo) — had to route through `AttnArgs` dataclass instead
3. LSE shape from varlen FA3 is `(num_heads, total_seqlen)`, requiring transpose + reshape that varies between paired/non-paired head configurations

Due to these issues, tested a simplified version: per-head learned scale `y * sigmoid(-sink)` without LSE dependency. This is a weaker version that doesn't capture position-dependent attention confidence.

**Changes:**
- Added `attn_sinks` parameter (10, num_heads) initialized to zeros
- Applied `sigmoid(-attn_sink)` as per-head scale before existing gate
- Added to param_table, work_order, and bf16 conversion

**Results:**

| | Val Loss | Train Time |
|---|----------|------------|
| Baseline (this pod) | 3.2789 | 681.7s |
| Simplified sinks | 3.2823 | 687.6s |

**Outcome:** Slightly worse (+0.0034 loss, +6s time). The simplified version without LSE is redundant with the existing sparse gate — both are just learned per-head scales. The real GPT-OSS mechanism requires position-dependent gating via LSE, which is blocked by the FA3 varlen API + torch.compile interaction.

**Possible follow-ups:**
- Patch `flash_attn_varlen_func` wrapper to pass through `return_softmax` correctly
- Use `torch.compiler.allow_in_graph` to wrap the FA3 call and handle the tuple return
- Try computing LSE approximation from attention output statistics (e.g., per-head output norm)

---

## Less-Cautious Weight Decay

**Date:** 2026-03-04
**Branch:** N/A (tested on RunPod only)
**Status:** No improvement
**Source:** [PR #172](https://github.com/KellerJordan/modded-nanogpt/pull/172) — shenberg

Standard cautious weight decay (CWD) blocks all weight decay when the update direction disagrees with the parameter sign. A less-cautious variant allows weight decay unless it would both flip the sign AND exceed the update magnitude.

**Hypothesis:** Standard CWD is too conservative — it prevents weight decay even when the decay is small relative to the update. Less-cautious WD should improve regularization quality. Expected 0.3-0.9s.

**Changes:**
- Adam CWD mask changed from `(update * p_slice) > 0` to `((update * p_slice) > 0) | ((update * p_slice) < p_slice.square() * (-eff_wd))`

**Results:**

| | Val Loss | Train Time |
|---|----------|------------|
| Baseline (this pod) | 3.2789 | 681.7s |
| Less-cautious WD | 3.2824 | 681.7s |

**Outcome:** Val loss +0.0035 worse, within noise. The existing CWD behavior is well-tuned for this training setup. With only 1490 steps and aggressive batch size scaling, the conservative WD approach that avoids conflicting with gradient direction appears optimal. The additional weight decay from the less-cautious variant may slightly harm the rapid schedule transitions.

---

## Parallelize MLP and Attention in Parallel Region

**Date:** 2026-03-04
**Branch:** N/A (tested on RunPod only)
**Status:** No improvement (mixed — faster but worse loss)
**Source:** [PR #230](https://github.com/KellerJordan/modded-nanogpt/pull/230) — ClassicLarry, msisovic

In the parallel residual region (layers 7-10), snapshot lane1 before attention modifies it and feed the pre-attn version to MLP. This removes the sequential dependency between attention and MLP, potentially allowing CUDA to schedule them concurrently.

**Hypothesis:** Lambda values show MLP largely ignores the sequential dependency. Parallelization should yield 0.3-1.0s speedup with minimal convergence impact.

**Changes:**
- Added `lane1_pre = lane1` snapshot before attention
- MLP reads `norm(lane1_pre)` instead of `norm(lane1)` (which includes attention's update)

**Results:**

| | Val Loss | Train Time |
|---|----------|------------|
| Baseline (this pod) | 3.2789 | 681.7s |
| Parallel MLP/attn | 3.2826 | 679.0s |

**Outcome:** Mixed result. Training was 2.7s faster (consistent with some kernel overlap), but val loss regressed by +0.0037. On 1xGPU, CUDA has limited ability to truly parallelize two large operations, so the speedup is modest. The loss regression suggests that despite lambda analysis showing weak dependency, MLP does extract some value from seeing attention-modified lane1 — particularly in the later layers where attention amplifies lane1 by ~2.2x.

**Possible follow-ups:**
- Only parallelize layers 7-8 (earliest parallel layers where dependency is weakest)
- Try on 8xH100 where communication overhead may make the speedup more valuable
- Statistical significance test — the loss difference may be within noise

---

## Separate Residual Lambdas Per Lane

**Date:** 2026-03-04
**Branch:** N/A (tested on RunPod only)
**Status:** No improvement
**Source:** [PR #230](https://github.com/KellerJordan/modded-nanogpt/pull/230) — msisovic

Currently both lanes share the same residual lambda per layer. Expanded `resid_lambdas` from `(num_layers, 2)` to `(num_layers, 2, 2)` — [layer, attn/mlp, lane0/lane1] — so each lane gets independent residual scaling.

**Hypothesis:** Logged lambda values show lanes develop very different characteristics in the parallel region. Independent lambdas should let each lane find its optimal residual scaling. Only adds 22 extra scalar parameters.

**Changes:**
- Changed `resid_lambdas` shape from `(num_layers, 2)` to `(num_layers, 2, 2)`
- Split unbinding into per-lane variants: `resid_lambdas_attn_ln0`, `resid_lambdas_attn_ln1`, etc.
- Updated all usage sites in sequential and parallel regions

**Results:**

| | Val Loss | Train Time |
|---|----------|------------|
| Baseline (this pod) | 3.2789 | 681.7s |
| Separate lambdas | 3.2812 | 680.4s |

**Outcome:** Val loss +0.0023, within noise. The extra degrees of freedom don't help — the model has enough flexibility through `post_lambdas` (which already has per-lane scaling) that per-lane residual lambdas are redundant. The `post_lambdas` already control per-lane contribution from attention and MLP outputs, while `resid_lambdas` controls the residual connection strength, which should logically be the same for both lanes since they share the same residual stream initially.

---

## Combined: Gate lr_mul=5.0 + Parallel MLP/Attn

**Date:** 2026-03-04
**Branch:** N/A (tested on RunPod only)
**Status:** No improvement (not additive)

Combined the two most promising individual changes: gate lr_mul=5.0 (best loss: -0.0018) and parallel MLP/attn (best speed: -2.7s).

**Hypothesis:** If the improvements are independent, we should get both better loss and faster training.

**Changes:**
- `lr_mul: 5.0` for `attn_gate_bank` and `ve_gate_bank`
- Snapshot `lane1_pre` before attention, feed to MLP for concurrent execution

**Results:**

| | Val Loss | Train Time |
|---|----------|------------|
| Baseline (this pod) | 3.2789 | 681.7s |
| Gate lr_mul=5.0 only | 3.2771 | 684.3s |
| Parallel MLP only | 3.2826 | 679.0s |
| Combined | 3.2805 | 680.4s |

**Outcome:** Not additive. The parallel MLP's loss regression (+0.0037) partially cancelled the gate lr improvement (-0.0018), yielding a net +0.0016 — within noise. The faster gate learning may be less effective when MLP receives stale lane1, since the gates need to compensate for the different information flow.

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
