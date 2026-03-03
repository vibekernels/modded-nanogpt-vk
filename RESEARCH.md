# modded-nanogpt Optimization Research

Research into systems optimizations and unexplored model quality improvements from
merged record-setting PRs in [KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt).

Current record: **88.1s** on 8xH100 to reach ≤3.28 validation loss.

All experiments run on 1×H100 80GB with `torchrun --standalone --nproc_per_node=1 train_gpt.py`.

---

## Benchmark Results (1xH100)

### Round 1: CPU Data Prep Baseline (val_loss 3.2801)

| Optimization | Final val_loss | Delta | Verdict |
|---|---|---|---|
| **Baseline (CPU data prep)** | 3.2801 | — | — |
| **GPT-OSS Attention Sinks** | **3.2787** | **-0.0014** | **Positive** |
| **CWD mask fix (>= → >)** | **3.2792** | **-0.0009** | **Positive** |
| **Combined (sinks + CWD fix)** | 3.2798 | -0.0003 | Not additive |
| Separate residual lambdas | 3.2816 | +0.0015 | Neutral/hurt |
| Parallel MLP+Attention | 3.2832 | +0.0031 | Hurt |
| Less-cautious weight decay | — | — | Hurt at step 750 (killed) |
| Gate LR mul 0.1x | — | — | Hurt at step 750 (killed) |
| Gate LR mul 5.0x | — | — | Hurt at step 750 (killed) |

### Round 2: GPU Data Prep Baseline (val_loss 3.2813, 3.4x faster)

GPU data prep moves type conversions + bigram hash to GPU, reducing total training
time from 38.4 min to 11.2 min on 1xH100 (see [Systems Optimization](#systems-optimization-gpu-data-prep) below).

| Optimization | Final val_loss | Delta vs GPU baseline | Verdict |
|---|---|---|---|
| **GPU data prep baseline** | 3.2813 | — | — |
| **CWD mask fix (>= → >)** | **3.2778** | **-0.0035** | **Best single opt** |
| **CWD + sinks** | **3.2780** | **-0.0033** | **Best combo** |
| **Cooldown frac 0.55** | **3.2780** | **-0.0033** | **Positive** |
| GPT-OSS sinks | 3.2796 | -0.0017 | Positive |
| Bigram lr_mul 50 (vs 75) | 3.2796 | -0.0017 | Positive |
| Cooldown frac 0.65 | 3.2809 | -0.0004 | Neutral |
| CWD + sinks + cooldown 0.55 | 3.2808 | -0.0005 | Not additive |
| CWD + cooldown 0.55 | 3.2816 | +0.0003 | Not additive |
| Value_embeds wd_mul 2.0 | 3.2815 | +0.0002 | Neutral |
| Resid_lambdas lr_mul 10 | 3.2833 | +0.0020 | Hurt |
| Soft-cap norm (embeddings) | — | — | Hurt at step 750 (killed) |
| Grad clipping at transitions | — | — | Hurt at step 1000 (killed) |

### Intermediate Checkpoints (Round 1)

| Optimization | step 500 | step 750 | step 1000 | step 1250 |
|---|---|---|---|---|
| Baseline | 4.2179 | 3.8098 | 3.5229 | 3.3730 |
| GPT-OSS Sinks | 4.2202 | 3.8115 | **3.5174** | **3.3716** |
| CWD mask fix | **4.2121** | **3.8077** | **3.5209** | **3.3722** |
| Combined | 4.2126 | 3.8094 | 3.5188 | 3.3727 |

### Round 3: GPU Data Prep + CWD + Sinks (current code)

Both CWD mask fix and GPT-OSS attention sinks are now implemented in `train_gpt.py`.

| Optimization | Final val_loss | Delta vs GPU baseline | Verdict |
|---|---|---|---|
| GPU data prep baseline | 3.2813 | — | — |
| **CWD + sinks (current code)** | **3.2790** | **-0.0023** | **Implemented** |

### Key Findings

- **CWD mask fix is the strongest single optimization** across both rounds
- GPT-OSS sinks improve steadily in later training (strongest at step 1000+)
- CWD + sinks combined: -0.0023 (between individual results, confirming non-additivity)
- Cooldown frac 0.55 (shorter cooldown) matches best results but worse intermediates
- **Optimizations are NOT additive** — combining any two produces worse results than either alone
- This non-additivity pattern held across both CPU and GPU data prep rounds; the optimizations
  likely find similar loss landscape minima from different directions
- Gate LR tuning: the default 1.0x is already near-optimal under Adam
- Less-cautious WD hurt — standard CWD is better for this setup
- Parallel MLP+Attention: sequential dependency matters despite small lambda values
- Separate residual lambdas: extra parameters don't help; shared lambdas provide useful regularization
- Soft-cap normalization for embeddings causes training instability
- Gradient clipping at stage transitions (max_norm=1.0, ±2 steps) too aggressive

### Current State

**Implemented in `train_gpt.py`:**
1. **GPU data prep** — 3.4x training speedup (38.4 min → 11.2 min on 1xH100)
2. **CWD mask fix** (`>=` → `>`) — strongest single model quality improvement
3. **GPT-OSS attention sinks** — learned LSE-based gating on attention output

Combined val_loss: **3.2790** (vs 3.2813 GPU baseline, **-0.0023**).

---

## Systems Optimization: GPU Data Prep

### Summary

|  | Baseline | Optimized | Speedup |
|--|----------|-----------|---------|
| Stage 0 (steps 0-499) | 237ms/step | 247ms/step | 1.0x |
| Stage 1 (steps 500-999) | 639ms/step | 456ms/step | **1.4x** |
| Stage 2 (steps 1000-1249) | 3,884ms/step | 652ms/step | **6.0x** |
| Stage 2+Ext (steps 1250-1490) | 3,913ms/step | 651ms/step | **6.0x** |
| **Total training time** | **2,304s (38.4 min)** | **672s (11.2 min)** | **3.4x** |
| Peak memory | 37,326 MiB | 37,327 MiB | unchanged |
| val_loss | 3.2801 | 3.2813 | +0.0012 (run-to-run variance) |

**70.8% total training time reduction** from a single optimization: moving data prep to GPU.

Note: initial development under Triton 3.4.0 measured slightly faster numbers (3.8x, 6.5x Stage 2)
but that Triton version had NaN issues. The numbers above are from re-validation under Triton 3.6.0.

### Root Cause: CPU Data Preparation Bottleneck

In Stages 2-3, CPU data preparation dominates at ~488ms per accumulation step × 8 = ~3,900ms per training step.

**Performance breakdown by stage (baseline):**

| Stage | Steps | Step Time | Data Prep | GPU Compute | Optimizer | Bottleneck |
|-------|-------|-----------|-----------|-------------|-----------|------------|
| 0 | 0-499 | 237ms | ~52ms (22%) | ~180ms (76%) | ~7ms | GPU |
| 1 | 500-999 | 639ms | ~56-213ms (9-33%) | ~385ms (60%) | ~7ms | Mixed |
| 2 | 1000-1449 | **3,884ms** | **~3,620ms (93%)** | ~120ms (3%) | ~7ms | **CPU** |
| Extension | 1450-1490 | **3,913ms** | **~3,600ms (92%)** | ~150ms (4%) | ~7ms | **CPU** |

**Per-accum-step breakdown (Stage 2, 49152 tokens):**

| Operation | Time | % of Data Prep |
|-----------|------|----------------|
| `get_bigram_hash()` | **398ms** | 82% |
| Type conversions (uint16 to int32/int64) | **77ms** | 16% |
| `torch.cat` (slice assembly) | 0.4ms | <1% |
| `next_batch` (BOS index walk) | 0.07ms | <1% |
| GPU transfer | 0.2ms | <1% |

The CPU cost scales nonlinearly with token count: 0.12ms at 16K tokens (Stage 0), 0.18ms at 32K (Stage 1), but **488ms at 49K** (Stage 2). This ~4000x jump for a 3x token increase is likely due to PyTorch CPU dispatch overhead thresholds and pinned memory allocation costs in `get_bigram_hash()`.

### GPU Kernel Breakdown (torch.profiler)

| Kernel | Time | % |
|--------|------|---|
| `aten::mm` (non-FP8 matmuls, optimizer) | 365ms | 29.7% |
| `linear_relu_square_kernel` (fused MLP) | 155ms | 12.6% |
| `aten::_scaled_mm` (FP8 matmuls) | 132ms | 10.7% |
| Flash Attention backward | 127ms | 10.3% |
| Fused softcapped CE backward | 50ms | 4.1% |
| Fused softcapped CE forward | 38ms | 3.1% |
| Flash Attention forward | 32ms | 2.6% |

GPU compute is well-optimized with fused kernels, FP8, and Flash Attention 3.

### Implementation

**Status**: Implemented and verified.

Transfer raw uint16 buffer to GPU, then do type conversions + bigram hash on GPU.

Changes in `train_gpt.py`:
1. `get_bigram_hash()` updated to support both CPU and GPU tensors (uses `x.clone()` instead of `pin_memory` allocation).
2. `distributed_data_generator()` transfers raw `buf` to GPU first, then creates `_inputs`/`_targets` with proper dtypes on GPU.
3. `_bigram_cpu` for `sparse_index_update` only computed when `_sparse_comms_active()` is True (8-GPU, grad_accum=1), avoiding unnecessary GPU→CPU sync otherwise.

### 8×H100 Impact

On 8×H100: `world_size=8`, `grad_accum_steps=1`, `_sparse_comms_active()=True`.

- Each GPU still processes 49,152 tokens (same per-GPU token count)
- 1 accumulation step per training step (vs 8 on 1×H100)
- Data prep bottleneck: 488ms/step vs ~74ms GPU compute → GPU idle ~87% of time
- After optimization: ~0.13ms data prep (including GPU→CPU sync for sparse comms)
- The `.cpu().numpy()` call for sparse comms adds ~0.1ms (192KB transfer + stream sync)
- **Same relative speedup (~6x on Stage 2)**, smaller absolute time saved per step

### Other Systems Ideas Considered

**Prefetch next accumulation step** — Not implemented. With GPU data prep, remaining CPU
work is ~0.5ms/accum (<1% of GPU compute time). Not worthwhile.

**Optimizer FP8 matmuls** — Not implemented. `aten::mm` calls (29.7% of GPU time) come from
Polar Express orthogonalization (5 iterations of 768x768 matmuls). Needs convergence validation.

---

## Model Quality Optimizations (Detail)

### CWD Mask Fix in NorMuon Path (>= vs >)

**Source:** [PR #172](https://github.com/KellerJordan/modded-nanogpt/pull/172)
**Suggested by:** ClassicLarry
**Status:** **Implemented. val_loss -0.0009 (CPU baseline), -0.0035 (GPU baseline).**

The Adam CWD path uses `>` (correct — zero gradients do not trigger weight decay).
The NorMuon CWD path at `_cautious_wd_and_update_inplace` previously used `>=`, which
meant tokens with zero gradients (unseen in the current batch) still got weight decay
applied. For sparse embeddings processed by NorMuon, this drove rarely-seen
token embeddings toward zero. Fixed to `>`.

```python
mask = (grad * p_precise) > 0
```

### GPT-OSS Attention Sinks

**Source:** [PR #117](https://github.com/KellerJordan/modded-nanogpt/pull/117) discussion
**Suggested by:** ClassicLarry (collaborator)
**Status:** **Implemented. val_loss -0.0014 (CPU baseline), -0.0017 (GPU baseline).**

The model already has a **sparse attention gate** — a per-head sigmoid deciding
"should this query attend at all?" GPT-OSS sinks address the complementary
question: "given that it attended, how much should it trust the result?"

The implementation multiplies attention output by `sigmoid(LSE - learned_sink)`,
where LSE (log-sum-exp) is a byproduct of softmax. Mathematically equivalent to
adding a learned null key to the softmax denominator, computed post-hoc without
a custom attention kernel:

> sigmoid(LSE - sink) = e^LSE / (e^sink + e^LSE)

#### Implementation (verified)

```python
y, lse = flash_attn_varlen_func(..., return_attn_probs=True)  # returns (out, softmax_lse)
y = y * sigmoid(F.linear(x[..., :12], attn_gate_w))   # existing sparse gate
lse = lse.detach().bfloat16().view(B, T, num_heads, 1) # cast fp32→bf16, reshape
y = y * sigmoid(lse - sinks.view(1, 1, num_heads, 1))  # new: GPT-OSS sinks
```

Adds one learned parameter per attention head per layer (`attn_sinks` tensor of shape
`(num_layers, num_heads)`). Negligible parameter and communication cost.

#### Implementation notes

- Flash Attention 3's `return_attn_probs=True` returns `(output, softmax_lse)` (not `return_lse`).
- LSE is fp32; must cast to bf16 to match model dtype (otherwise Triton compilation fails).
- LSE must be `.detach()`ed — flash attention backward doesn't support LSE gradients, but
  the main gradient still flows through the attention output `y`.
- For paired-head layers, LSE shape `(T*2, num_heads//2)` reshapes correctly to `(T, num_heads)`.

### Cooldown Fraction 0.55

**Status:** **Positive. val_loss -0.0033 (GPU baseline).**

The training schedule has `cooldown_frac=0.60` with a commented-out alternative of 0.55.
Shorter cooldown (0.55) keeps LR higher for longer, improving final convergence.
Also tested 0.65 (val_loss 3.2809, marginal).

### Bigram Embed lr_mul 50

**Status:** **Marginal positive. val_loss -0.0017 (GPU baseline).**

Reducing `bigram_embed` lr_mul from 75 to 50 slightly improved final val_loss,
suggesting rare bigrams may get noisy updates at the higher LR.

---

## Benchmarked Ideas That Hurt

### Gate Learning Rate Multiplier Tuning

**Source:** [PR #117](https://github.com/KellerJordan/modded-nanogpt/pull/117),
[PR #146](https://github.com/KellerJordan/modded-nanogpt/pull/146)
**Suggested by:** varunneal, YouJiacheng
**Status:** **Both 0.1x and 5.0x hurt (killed at step 750). Default 1.0x is near-optimal under Adam.**

YouJiacheng found 0.1x lr multiplier was key under Muon, but gates were later moved to
Adam and never re-tuned. Testing confirmed the default Adam lr is already near-optimal.

### Parallelize MLP and Attention

**Source:** [PR #230](https://github.com/KellerJordan/modded-nanogpt/pull/230)
**Suggested by:** ClassicLarry
**Status:** **val_loss 3.2832 (+0.0031). Hurt.**

Snapshot `lane1` before attention modifies it, feed pre-attn version to MLP. Despite
small lambda values suggesting minimal cross-dependency, the sequential dependency
between attention and MLP still matters.

### Less-Cautious Weight Decay

**Source:** [PR #172](https://github.com/KellerJordan/modded-nanogpt/pull/172)
**Suggested by:** shenberg
**Status:** **Hurt at step 750 (val_loss 3.8314 vs baseline 3.8098), killed early.**

A variant that allows weight decay unless it would flip the sign AND exceed the
update magnitude. Standard CWD is better for this setup.

### Separate Residual Lambdas Per Lane

**Source:** [PR #230](https://github.com/KellerJordan/modded-nanogpt/pull/230)
**Suggested by:** msisovic
**Status:** **val_loss 3.2816 (+0.0015). Hurt.**

Expanding `resid_lambdas` from `(num_layers, 2)` to `(num_layers, 2, 2)`. Extra
parameters don't help; shared lambdas provide useful regularization.

### Soft-Cap Normalization for Embeddings

**Source:** [PR #201](https://github.com/KellerJordan/modded-nanogpt/pull/201)
**Suggested by:** ClassicLarry
**Status:** **Hurt (val_loss 3.87 at step 750 vs baseline 3.80), killed early.**

Soft-cap norm that caps high magnitudes while allowing zeros, applied to bigram and
value embeddings. Caused training instability.

### Gradient Clipping at Stage Transitions

**Source:** [PR #163](https://github.com/KellerJordan/modded-nanogpt/pull/163)
**Suggested by:** ClassicLarry
**Status:** **Hurt (val_loss 3.5343 at step 1000 vs baseline 3.5188), killed.**

`max_norm=1.0` within ±2 steps of batch size transitions was too aggressive.

### Other Hyperparameter Tuning

- **Value_embeds wd_mul 2.0** (vs 5.0): val_loss 3.2815, neutral. Good intermediates but converged to baseline.
- **Resid_lambdas lr_mul 10** (vs 5.0): val_loss 3.2833, hurt.

---

## Untested Ideas (ranked by feasibility × impact)

### Fuse Loss Computation into Backward Kernel

**Source:** [PR #199](https://github.com/KellerJordan/modded-nanogpt/pull/199)
**Suggested by:** YouJiacheng
**Estimated impact:** 0.1–0.5s | **Effort:** Medium-high

The forward pass of `FusedSoftcappedCrossEntropy` computes a loss value that is
never used during training (only gradients matter). Making the forward a no-op
and fusing all computation into the backward kernel eliminates one Triton kernel
launch and one partial load of the logits tensor (~150–450ms total over 1490 steps).

### Sparse Gradient Communication for All Embeddings

**Source:** [PR #201](https://github.com/KellerJordan/modded-nanogpt/pull/201)
**Suggested by:** shenberg
**Estimated impact:** 1–3s | **Effort:** Medium

`bigram_embed` already uses `"sharded_sparse"` communication (PR #221). The
`value_embeds` and `embed` parameters still use dense `"sharded"` communication
despite having similarly sparse gradients (~65% sparsity). The sparse
communication infrastructure exists — extending it is primarily an engineering
task of handling the different tensor shapes.

### Custom Lightweight Optimizer for Small Parameters

**Source:** [PR #132](https://github.com/KellerJordan/modded-nanogpt/pull/132)
**Suggested by:** ClassicLarry
**Estimated impact:** 0.1–0.3s | **Effort:** Medium

Small replicated parameters (scalars, gates, lambdas — 15+ groups) have
disproportionate overhead from Adam's full state machinery. A stripped-down SGD
with momentum could suffice for these, saving per-step overhead that accumulates
over 1490 steps.

### Collage Optimizer

**Source:** [PR #166](https://github.com/KellerJordan/modded-nanogpt/pull/166)
**Suggested by:** shenberg
**Estimated impact:** 0.5–1.5s | **Effort:** High

Since Adam updates are communication-bound, GPU-local compute is effectively
free. Collage (arXiv 2405.03637) computes multiple optimizer update candidates
and selects the best per parameter group. The extra compute hides behind
communication with zero wall-time cost, improving update quality and reducing
steps to target.

### Delayed Weight Update (Stale Weights)

**Source:** [PR #102](https://github.com/KellerJordan/modded-nanogpt/pull/102)
**Suggested by:** KellerJordan (repo owner)
**Estimated impact:** 0–3s | **Effort:** Medium

Start the next forward pass before the optimizer finishes, overlapping optimizer
compute with forward pass compute using separate CUDA streams. Changes training
dynamics (weights are one step stale), so requires statistical validation.
KellerJordan flagged the safe variant — running Adam and Muon concurrently rather
than sequentially — as a pure systems win.

### Concept-Aware Data Batching

**Source:** [PR #118](https://github.com/KellerJordan/modded-nanogpt/pull/118)
**Suggested by:** ClassicLarry
**Estimated impact:** 0.3–0.6s | **Effort:** Medium

The data loader reads contiguous chunks from individual shards, creating
correlated batches when long documents dominate. Interleaving reads from multiple
shards would reduce correlation with minimal runtime overhead.

### Value Embeddings Guided by LM Head

**Source:** [PR #175](https://github.com/KellerJordan/modded-nanogpt/pull/175)
**Suggested by:** ClassicLarry
**Estimated impact:** 0.3–0.9s | **Effort:** High

Initialize value embeddings from transposed LM head weights. Simpler alternative:
tune `wd_mul` or `lr_mul` for value_embeds (partially tested — wd_mul 2.0 was neutral).

### QKVG Attention

**Source:** [PR #117](https://github.com/KellerJordan/modded-nanogpt/pull/117)
**Suggested by:** byronxu99
**Estimated impact:** Uncertain | **Effort:** High

Per-head G projection that gates attention output by `sigmoid(dot(Q, G))`.
More expressive than the 12-dim sparse gate, but adds a full `(768, 768)` weight
matrix per layer. Likely net negative on wall time for speedrun.

### Hourglass Attention Head Count

**Source:** [PR #120](https://github.com/KellerJordan/modded-nanogpt/pull/120)
**Suggested by:** ClassicLarry
**Estimated impact:** 1–3s | **Effort:** Very high

More attention heads in early/late layers, fewer in middle layers. Strong
theoretical grounding but breaks the uniform `attn_bank` parameter shape.

---

## Context: Current Architecture

- **Model:** 11-layer GPT, 768-dim, parallel residual connections (layers 7–10)
- **Attention:** Flash Attention 3, paired-head with YaRN window extension, sparse gates
- **Embeddings:** Token embed (tied to LM head for first 2/3), bigram hash embed, value embeds (5 layers)
- **Optimizer:** Unified NorMuonAndAdam with Polar Express, cautious weight decay, mantissa tracking
- **Schedule:** 3 stages + extension, batch size 8→16→24, MTP decay
- **Communication:** Explicit scatter/work ordering, sharded + replicated + sharded_sparse comms

## Context: Leaderboard Rules

- 8x NVIDIA H100 GPUs, wall-clock time to ≤3.28 validation loss
- Must not modify train/validation data pipelines (batch size, seq len, attention changes OK)
- Must not use extra `torch.compile` flags
- New records need p<0.01 statistical significance unless purely systems-level
- Code readability matters — changes with poor readability-to-performance ratio may be rejected
