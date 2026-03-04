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
