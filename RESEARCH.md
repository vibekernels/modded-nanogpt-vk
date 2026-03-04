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
**Branch:** `turbo-muon-aol-fused` (`aa203a6`)
**Status:** Failed (NaN)

Fuse AOL preconditioning into dedicated Triton kernels to eliminate PyTorch dispatch overhead. Three kernels: `_aol_prescale_A_kernel` (compute row sums + rsqrt), `_aol_scale_A_kernel` (scale Gram matrix), `_aol_scale_X_{cols,rows}_kernel` (scale X). Combined with 4-iter Turbo-Muon coefficients.

**Hypothesis:** The unfused AOL's overhead (5 PyTorch ops) dominates the savings from removing one NS iteration. Fusing into Triton eliminates dispatch overhead, making the 4-iter approach net positive.

**Changes:**
- Added 4 new Triton kernels in `triton_kernels.py` for fused AOL
- Restructured `polar_express` loop: compute Gram matrix once for AOL before iteration loop
- 4-iter Turbo-Muon coefficients (same as first experiment)

**Results:**

| | Val Loss | Train Time | Step Avg (stage 1) |
|---|----------|------------|-------------------|
| Baseline | 3.2802 | 668s | — |
| Fused AOL 4-iter | **NaN** | 637.2s | 256.2ms |

**Outcome:** Training diverged to NaN by step 250. The fused kernel is numerically incorrect — likely a bug in the Triton AOL scaling kernels (possibly bf16 precision loss during the in-place rsqrt + multiply chain, or incorrect stride handling for batched tensors). The step_avg of 256ms confirms the fused approach is faster (vs 262ms unfused, 265ms baseline estimate), but the computation is wrong.

**Possible follow-ups:**
- Debug the Triton kernel: test with fp32 intermediates, verify stride math for batched tensors
- Check if the in-place bf16 rsqrt accumulates too much error — the unfused PyTorch path may silently use fp32 intermediates
- Simpler approach: just keep AOL ops in PyTorch but use `torch.compile` to let it fuse automatically

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
