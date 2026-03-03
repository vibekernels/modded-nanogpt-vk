# modded-nanogpt Optimization Research

Research into unexplored optimizations from merged record-setting PRs in
[KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt).

Current record: **88.1s** on 8xH100 to reach ≤3.28 validation loss.

---

## Benchmark Results (1xH100)

### Round 1: CPU Data Prep Baseline (val_loss 3.2801)

| Optimization | Final val_loss | Delta | Verdict |
|---|---|---|---|
| **Baseline (CPU data prep)** | 3.2801 | — | — |
| **GPT-OSS Attention Sinks** | **3.2787** | **-0.0014** | **Best single opt** |
| **CWD mask fix (>= → >)** | **3.2792** | **-0.0009** | **Positive** |
| **Combined (sinks + CWD fix)** | 3.2798 | -0.0003 | Not additive |
| Separate residual lambdas | 3.2816 | +0.0015 | Neutral/hurt |
| Parallel MLP+Attention | 3.2832 | +0.0031 | Hurt |
| Less-cautious weight decay | — | — | Hurt at step 750 (killed) |
| Gate LR mul 0.1x | — | — | Hurt at step 750 (killed) |
| Gate LR mul 5.0x | — | — | Hurt at step 750 (killed) |

### Round 2: GPU Data Prep Baseline (val_loss 3.2813, 3.4x faster)

GPU data prep moves type conversions + bigram hash to GPU, reducing total training
time from 38.4 min to 11.2 min on 1xH100. See RESEARCH.md for details.

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

**Intermediate checkpoints (Round 1, step val_loss):**

| Optimization | step 500 | step 750 | step 1000 | step 1250 |
|---|---|---|---|---|
| Baseline | 4.2179 | 3.8098 | 3.5229 | 3.3730 |
| GPT-OSS Sinks | 4.2202 | 3.8115 | **3.5174** | **3.3716** |
| CWD mask fix | **4.2121** | **3.8077** | **3.5209** | **3.3722** |
| Combined | 4.2126 | 3.8094 | 3.5188 | 3.3727 |

**Key findings:**
- CWD mask fix is the strongest single optimization across both rounds
- GPT-OSS sinks improve steadily in later training (strongest at step 1000+)
- Cooldown frac 0.55 (shorter cooldown) matches best results but worse intermediates
- Optimizations are generally NOT additive — combining any two works worse than individual
- Gate LR tuning: the default 1.0x is already near-optimal under Adam
- Less-cautious WD hurt — standard CWD is better for this setup
- Parallel MLP+Attention: sequential dependency matters despite small lambda values
- Separate residual lambdas: extra parameters don't help; shared lambdas provide useful regularization
- Soft-cap normalization for embeddings causes NaN/training instability
- Gradient clipping at stage transitions is too aggressive, interferes with training

---

## Top Recommendation: GPT-OSS Attention Sinks + Existing Sparse Gate

**Source:** [PR #117](https://github.com/KellerJordan/modded-nanogpt/pull/117) discussion
**Suggested by:** ClassicLarry (collaborator)
**Status:** **Benchmarked — val_loss improved by 0.0014 (3.2801 → 3.2787)**

### What it is

The model already has a **sparse attention gate** — a per-head sigmoid deciding
"should this query attend at all?" GPT-OSS sinks address the complementary
question: "given that it attended, how much should it trust the result?"

The implementation multiplies attention output by `sigmoid(LSE - learned_sink)`,
where LSE (log-sum-exp) is a byproduct of softmax. As byronxu99 explained in
PR #117, this is mathematically equivalent to adding a learned null key to the
softmax denominator, but computed post-hoc without a custom attention kernel:

> sigmoid(LSE - sink) = e^LSE / (e^sink + e^LSE)

Each mechanism independently improved loss on the harder 2.92 track. ClassicLarry's
reasoning for why the combination should be additive:

> "sparse attn gate lets a query decide if it wants to look for a key, independent
> of other tokens. gpt_oss assumes that a query does want to look for a key, and
> then modulates based on if it finds one. Ideally, a conceptually complete pairing
> mechanism should do both."

### Implementation (verified)

Approximately 3 lines on top of the existing attention function:

```python
y, lse = flash_attn_varlen_func(..., return_attn_probs=True)  # returns (out, softmax_lse)
y = y * sigmoid(F.linear(x[..., :12], attn_gate_w))   # existing sparse gate
lse = lse.detach().bfloat16().view(B, T, num_heads, 1) # cast fp32→bf16, reshape
y = y * sigmoid(lse - sinks.view(1, 1, num_heads, 1))  # new: GPT-OSS sinks
```

Adds one learned parameter per attention head per layer (`attn_sinks` tensor of shape
`(num_layers, num_heads)`). Negligible parameter and communication cost.

### Implementation notes

- Flash Attention 3's `return_attn_probs=True` returns `(output, softmax_lse)` (not `return_lse`).
- LSE is fp32; must cast to bf16 to match model dtype (otherwise Triton compilation fails).
- LSE must be `.detach()`ed — flash attention backward doesn't support LSE gradients, but
  the main gradient still flows through the attention output `y`.
- For paired-head layers, LSE shape `(T*2, num_heads//2)` reshapes correctly to `(T, num_heads)`.
- Not additive with CWD mask fix (combined: 3.2798, worse than either alone).

### Estimated impact

0.6–1.8 seconds (10–30 fewer training steps). This is a modeling improvement,
which tends to have outsized impact because every step saved compounds across
forward, backward, optimizer, and communication.

---

## Quick Wins (independent, can test in parallel)

### 1. Gate Learning Rate Multiplier Tuning

**Source:** [PR #117](https://github.com/KellerJordan/modded-nanogpt/pull/117),
[PR #146](https://github.com/KellerJordan/modded-nanogpt/pull/146)
**Suggested by:** varunneal, YouJiacheng
**Estimated impact:** 0.3–1.2s
**Effort:** One-line param_table change
**Status:** **Benchmarked — both 0.1x and 5.0x hurt (killed at step 750). Default 1.0x is near-optimal under Adam.**

YouJiacheng found that a 0.1x lr multiplier was the difference between the sparse
attention gate working or failing on the 2.92 track (under Muon). When the gates
were later moved from Muon to Adam, the lr was never re-tuned. The current
param_table specifies no `lr_mul` for `attn_gate_bank` or `ve_gate_bank`, so they
inherit the default Adam lr of 0.008.

**Change:** Add `"lr_mul": X` to the `attn_gate_bank` and `ve_gate_bank` entries
in the param_table. Test values: 0.1, 0.5, 2.0, 5.0.

### 2. Parallelize MLP and Attention in Parallel Region

**Source:** [PR #230](https://github.com/KellerJordan/modded-nanogpt/pull/230)
**Suggested by:** ClassicLarry
**Estimated impact:** 0.3–1.0s
**Effort:** Small forward pass refactor
**Status:** **Benchmarked — val_loss 3.2832 (+0.0031). Hurt. Sequential dependency between attention and MLP matters even with small lambda values.**

msisovic's logged lambda values from PR #230 show that in the parallel residual
region (layers 7–10), attention consistently amplifies lane1 to ~2.2x while MLP
outputs are more uniform. The model is learning to largely ignore the sequential
dependency between MLP_9 and Attn_10.

Currently, within each parallel layer, attention runs before MLP, and MLP reads
from the attention-modified lane1. To parallelize, save `lane1` before the
attention update and feed the pre-attn version to MLP:

```python
# Current (sequential):
attn_out = attn(norm(lane0), ...)
lane0 = resid_lambdas_attn[i] * lane0 + post_lambdas_attn_ln0[i] * attn_out + x0_inject[i]
lane1 = resid_lambdas_attn[i] * lane1 + post_lambdas_attn_ln1[i] * attn_out
mlp_out = ReLUSqrdMLP(norm(lane1), c_fc, c_proj)  # reads attn-modified lane1

# Proposed (parallel):
lane1_pre = lane1  # snapshot before attn modifies it
attn_out = attn(norm(lane0), ...)
mlp_out = ReLUSqrdMLP(norm(lane1_pre), c_fc, c_proj)  # reads original lane1
# then apply both updates to lanes
```

This lets CUDA schedule the attention and MLP kernels concurrently on layers
where the lambda values show minimal cross-dependency.

### 3. Less-Cautious Weight Decay

**Source:** [PR #172](https://github.com/KellerJordan/modded-nanogpt/pull/172)
**Suggested by:** shenberg
**Estimated impact:** 0.3–0.9s
**Effort:** Replace mask formula
**Status:** **Benchmarked — hurt at step 750 (val_loss 3.8314 vs baseline 3.8098), killed early. Standard CWD is better for this setup.**

shenberg's insight: standard cautious weight decay (CWD) blocks all weight decay
when the update direction disagrees with the parameter sign. A less-cautious
variant allows weight decay unless it would both flip the sign of the update AND
exceed the update magnitude:

```python
# Current:
mask = (update * p_slice) > 0

# Proposed:
mask = ((update * p_slice) > 0) | ((update * p_slice) < p_slice.square() * (-eff_wd))
```

shenberg tested this and found it "seems to be a beneficial change" but never
submitted a follow-up PR. Under Adam (communication-bound), the extra compute
is free. Under NorMuon (compute-bound), it may add measurable overhead.

### 4. Fix CWD Mask in NorMuon Path (>= vs >)

**Source:** [PR #172](https://github.com/KellerJordan/modded-nanogpt/pull/172)
**Suggested by:** ClassicLarry
**Estimated impact:** 0.3–0.9s
**Effort:** One-character change
**Status:** **Benchmarked — val_loss 3.2792 (-0.0009). Positive. Immediate improvement from step 500 onward.**

The Adam CWD path uses `>` (correct — zero gradients do not trigger weight decay).
The NorMuon CWD path at `_cautious_wd_and_update_inplace` uses `>=`, which means
tokens with zero gradients (unseen in the current batch) still get weight decay
applied. For sparse embeddings processed by NorMuon, this drives rarely-seen
token embeddings toward zero.

```python
# Current (NorMuon path):
mask = (grad * p_precise) >= 0

# Proposed:
mask = (grad * p_precise) > 0
```

---

## Other Unexplored Ideas (ranked by feasibility x impact)

### 5. Fuse Loss Computation into Backward Kernel

**Source:** [PR #199](https://github.com/KellerJordan/modded-nanogpt/pull/199)
**Suggested by:** YouJiacheng
**Estimated impact:** 0.1–0.5s | **Effort:** Medium-high

The forward pass of `FusedSoftcappedCrossEntropy` computes a loss value that is
never used during training (only gradients matter). Making the forward a no-op
and fusing all computation into the backward kernel eliminates one Triton kernel
launch and one partial load of the logits tensor (~150–450ms total over 1490 steps).

### 6. Sparse Gradient Communication for All Embeddings

**Source:** [PR #201](https://github.com/KellerJordan/modded-nanogpt/pull/201)
**Suggested by:** shenberg
**Estimated impact:** 1–3s | **Effort:** Medium

`bigram_embed` already uses `"sharded_sparse"` communication (PR #221). The
`value_embeds` and `embed` parameters still use dense `"sharded"` communication
despite having similarly sparse gradients (~65% sparsity). The sparse
communication infrastructure exists — extending it is primarily an engineering
task of handling the different tensor shapes.

### 7. Soft-Cap Normalization for Embeddings

**Source:** [PR #201](https://github.com/KellerJordan/modded-nanogpt/pull/201)
**Suggested by:** ClassicLarry
**Estimated impact:** 0.3–0.9s | **Effort:** Medium
**Status:** **Benchmarked — hurt (val_loss 3.87 at step 750 vs baseline 3.80), killed early. Training instability likely from normalizing sparse embeddings.**

Standard RMSNorm prevents embeddings from reaching zero magnitude, which is
semantically meaningful for bigram and value embeddings (zero = "don't contribute").
A soft-cap norm that caps high magnitudes while allowing zeros:

```python
def soft_cap_norm(x, threshold=1.0):
    norms = x.norm(dim=-1, keepdim=True)
    scale = torch.clamp(threshold / norms, max=1.0)
    return x * scale
```

### 8. Custom Lightweight Optimizer for Small Parameters

**Source:** [PR #132](https://github.com/KellerJordan/modded-nanogpt/pull/132)
**Suggested by:** ClassicLarry
**Estimated impact:** 0.1–0.3s | **Effort:** Medium

Small replicated parameters (scalars, gates, lambdas — 15+ groups) have
disproportionate overhead from Adam's full state machinery. A stripped-down SGD
with momentum could suffice for these, saving per-step overhead that accumulates
over 1490 steps.

### 9. Separate Residual Lambdas Per Lane

**Source:** [PR #230](https://github.com/KellerJordan/modded-nanogpt/pull/230)
**Suggested by:** msisovic
**Estimated impact:** 0.1–0.5s | **Effort:** Low
**Status:** **Benchmarked — val_loss 3.2816 (+0.0015). Hurt. Shared lambdas provide useful regularization; extra parameters don't help.**

Currently both lanes share the same residual lambda per layer. The logged lambda
values show lanes develop very different characteristics. Expanding
`resid_lambdas` from `(num_layers, 2)` to `(num_layers, 2, 2)` adds only 8 extra
scalars with negligible overhead.

### 10. Collage Optimizer

**Source:** [PR #166](https://github.com/KellerJordan/modded-nanogpt/pull/166)
**Suggested by:** shenberg
**Estimated impact:** 0.5–1.5s | **Effort:** High

Since Adam updates are communication-bound, GPU-local compute is effectively
free. Collage (arXiv 2405.03637) computes multiple optimizer update candidates
and selects the best per parameter group. The extra compute hides behind
communication with zero wall-time cost, improving update quality and reducing
steps to target.

### 11. Delayed Weight Update (Stale Weights)

**Source:** [PR #102](https://github.com/KellerJordan/modded-nanogpt/pull/102)
**Suggested by:** KellerJordan (repo owner)
**Estimated impact:** 0–3s | **Effort:** Medium

Start the next forward pass before the optimizer finishes, overlapping optimizer
compute with forward pass compute using separate CUDA streams. Changes training
dynamics (weights are one step stale), so requires statistical validation.
KellerJordan flagged the safe variant — running Adam and Muon concurrently rather
than sequentially — as a pure systems win.

### 12. Gradient Clipping at Stage Transitions

**Source:** [PR #163](https://github.com/KellerJordan/modded-nanogpt/pull/163)
**Suggested by:** ClassicLarry
**Estimated impact:** 0–0.5s | **Effort:** Medium
**Status:** **Benchmarked — hurt (val_loss 3.5343 at step 1000 vs baseline 3.5188), killed. max_norm=1.0 within ±2 steps of transitions was too aggressive.**

Visible loss spikes at the 1/3 batch size transition (131072 → 262144 tokens)
suggest gradient instability. Targeted gradient clipping for Adam parameters
during transition steps could smooth recovery. NorMuon's polar express already
normalizes gradient magnitude, so clipping there may be redundant.

### 13. Concept-Aware Data Batching

**Source:** [PR #118](https://github.com/KellerJordan/modded-nanogpt/pull/118)
**Suggested by:** ClassicLarry
**Estimated impact:** 0.3–0.6s | **Effort:** Medium

The data loader reads contiguous chunks from individual shards, creating
correlated batches when long documents dominate. Interleaving reads from multiple
shards would reduce correlation with minimal runtime overhead. ClassicLarry's
analysis showed that single long documents (court cases, fan fiction) can dominate
entire batches, making gradients poor estimates of the true distribution.

### 14. Value Embeddings Guided by LM Head

**Source:** [PR #175](https://github.com/KellerJordan/modded-nanogpt/pull/175)
**Suggested by:** ClassicLarry
**Estimated impact:** 0.3–0.9s | **Effort:** High

Value embeddings face the same sparse gradient problem as token embeddings (~1/50000
activation probability per token). Initializing them from transposed LM head
weights (tiled to 5 layers) could give rare tokens a meaningful starting point.
Simpler alternative: tune `wd_mul` or `lr_mul` for value_embeds based on
per-token frequency.

### 15. QKVG Attention

**Source:** [PR #117](https://github.com/KellerJordan/modded-nanogpt/pull/117)
**Suggested by:** byronxu99
**Estimated impact:** Uncertain | **Effort:** High

Add a per-head G projection that gates attention output by `sigmoid(dot(Q, G))`.
More expressive than the current 12-dim sparse gate, but adds a full
`(768, 768)` weight matrix per layer. Likely net negative on wall time for a
speedrun due to the extra matmul, but could be viable for larger models with GQA.

### 16. Hourglass Attention Head Count

**Source:** [PR #120](https://github.com/KellerJordan/modded-nanogpt/pull/120)
**Suggested by:** ClassicLarry
**Estimated impact:** 1–3s | **Effort:** Very high

More attention heads in early/late layers (data retrieval and prediction), fewer
in middle layers (processing). Strong theoretical grounding but breaks the
uniform `attn_bank` parameter shape that the entire optimizer pipeline depends on.
A workaround: use the existing attention gate with per-layer, per-head bias terms
initialized to suppress certain heads, approximating the hourglass pattern
without changing parameter shapes.

---

## Context: Current Architecture (as of latest merged record)

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
