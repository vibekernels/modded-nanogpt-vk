# CUDA C Kernel Rewrite Progress

Pure CUDA C reimplementation of the modded-nanogpt-vk training pipeline, eliminating all Python/PyTorch/Triton dependencies.

## File Structure

| File | Lines | Purpose |
|------|-------|---------|
| `train_gpt.cu` | ~3,500 | Model, forward/backward, optimizer, training loop, data loading |
| `kernels.cu` | ~2,850 | Custom CUDA kernels (ports of triton_kernels.py + extras) |
| `train_gpt.h` | ~420 | Shared structs, constants, forward declarations |
| `kernels.h` | ~290 | Kernel launch wrapper declarations |
| `Makefile` | ~60 | Build system (nvcc sm_90a, links cublas/cudnn/curand) |

## Current Status

### Performance

| Metric | CUDA C | Python Baseline |
|--------|--------|-----------------|
| Step time | ~1,400ms | ~450ms |
| Val loss (step 0) | 10.8302 | 10.8305 |
| Val loss (step 250) | ~5.68 | ~4.52 |
| Val loss (step 500) | ~5.53 | ~4.25 |
| Target val loss | ~3.28 | 3.2802 |

**Speed**: ~3x slower than Python (down from 13x before cuDNN flash attention).
**Quality**: ~1.2+ nat gap in val_loss that grows over training. Root cause under investigation.

### What Works

#### Forward Pass (verified correct at step 0)
- Embedding lookup + bigram hash + smear gate
- RMS norm (custom kernel)
- 11-layer transformer with dual-lane residual streams (layers 7-10 parallel)
- Attention: QKV projection (cuBLAS), QK RMS norm, YaRN RoPE, key offset, value embed gating, cuDNN SDPA (flash attention), output gating, output projection (cuBLAS)
- MLP: c_fc matmul (cuBLAS), fused ReLU^2 (custom kernel), c_proj matmul (cuBLAS)
- Skip connection (layer 3 save, layer 6 inject)
- Backout subtraction, lane averaging
- Final RMS norm, lm_head matmul (BF16 via cuBLAS)
- Softcapped cross-entropy loss with multi-token prediction (custom kernel)
- cuDNN SDPA with power-of-2 num_seqs bucketing and ragged sequence offsets

#### Backward Pass
- Softcapped CE backward (BF16 variant, custom kernel)
- lm_head backward (cuBLAS BF16: grad_x via gemm_bf16_Bt, grad_w via gemm_bf16_At)
- Final RMS norm backward (custom kernel)
- Backout/lane-merge backward with scalar gradient accumulation
- Per-layer backward (10 -> 0): cuDNN SDPA backward, MLP backward, residual/lambda/gate scalar gradients
- Embedding backward: scatter_add with native bf16 atomicAdd (SM 9.0)
- GPU-side scalar gradient accumulation kernel (maps float accumulators to bf16 gradient buffers)

#### Optimizer
- **NorMuon** (attn_bank, mlp_bank): Nesterov momentum (EMA-style, matching Python's `lerp_`), Polar Express orthogonalization (5 iterations), NorMuon variance reduction, cautious weight decay with mantissa-tracked BF16 updates
- **Adam** (scalars, gates, embeddings, lm_head): bias-corrected Adam with per-param LR multipliers, betas, weight decay. Steps on odd iterations only.
- Embed/lm_head tying with transpose_add/transpose_copy, split at step ~967
- All hyperparameters verified matching Python's param_table

#### Training Loop
- 4-stage schedule with YaRN window extensions
- MTP weight interpolation per step
- BOS-aligned batch construction from mmap'd .bin shards
- Cached scalar parameters (host-side copies constant across microsteps)
- num_seqs passed from batch construction to forward/backward (avoids D2H sync)

### Custom CUDA Kernels (kernels.cu)
- RMS norm forward/backward
- XXT, XTX, ba+cAA (symmetric matmul for Polar Express)
- Fused ReLU^2 forward/backward
- Softcapped CE forward/backward (BF16 variant)
- Transpose copy/add (32x32 tiled)
- RoPE apply + YaRN table computation
- Embedding gather/scatter_add, bigram hash
- Sigmoid gate, fused_add_scale, fused_add3, smear forward/backward
- Adam update (cautious, with mask), Muon cautious update (mantissa-tracked BF16)
- NorMuon variance reduction (per-row/col second momentum EMA)
- Nesterov momentum, norm divide, tensor norm, scale tensor
- BF16 dot product (multi-block atomic), bf16 L2 norm
- Key offset shift forward/backward
- GPU-side scalar gradient accumulation

## Quality Gap Investigation

### Verified Correct
- **Forward pass**: Step 0 val_loss matches Python exactly (10.8302 vs 10.8305)
- **Model initialization**: Same random distributions as Python (attn_bank uniform, mlp c_fc uniform, c_proj zeros, lm_head normal(0,0.005), etc.)
- **Parameter norms at init**: attn_bank ~87.6, mlp_bank ~96.0, scalars ~4.03 (identical)
- **Nesterov momentum formula**: EMA-style `buf = m*buf + (1-m)*g; out = m*buf + (1-m)*g` (matches Python's `lerp_`)
- **All optimizer hyperparameters**: LR multipliers, betas, weight decay multipliers match Python's param_table exactly
- **Gradient clearing pattern**: NorMuon grads zeroed every step, Adam grads zeroed on even steps (equivalent to Python clearing at end of odd steps)
- **Polar Express**: XTX/XXT kernels, ba+cAA kernel, batched GEMM strides all verified correct
- **cuBLAS GEMM**: Row-major wrapper correctly translates to column-major cuBLAS calls

### Key Diagnostic Findings

**Gradient norm explosion between steps (CUDA):**
- Step 0: x0_lambdas=2.97, resid_lambdas=2.67 (reasonable)
- Step 1: x0_lambdas=3061, resid_lambdas=4028 (1000x increase)
- Step 2: x0_lambdas=20213, resid_lambdas=24162 (further 7x increase)

**Raw float scalar_grad_acc values per microstep:**
- Step 0, micro 0: x0_lambda[0]=0.27 (reasonable)
- Step 1, micro 0: x0_lambda[0]=191.9 (720x increase for a single microstep!)
- Step 2, micro 0: x0_lambda[0]=779.8 (further 4x increase)

**This means the backward pass itself produces rapidly growing gradients** -- it's NOT an accumulation or gradient clearing bug. The float accumulators are zeroed correctly each microstep, and the accumulate_scalar_grads kernel correctly maps them to bf16 buffers.

**Meanwhile, grad_lane0 norms are stable** (~1.78 at layer 0 across all steps), and **x0 (RMS-normed embeddings) are unchanged** between steps 0-1 (embed weights not updated by NorMuon). So the 720x increase in `dot(grad_lane0, x0)` must come from grad_lane0's *direction* becoming much more aligned with x0 after the first NorMuon step.

### Unresolved Question
Whether this gradient growth pattern is normal (Python shows similar behavior) or pathological (CUDA-specific bug). The Python diagnostic script crashes at step 1 due to `torch.compile` issues, preventing direct comparison. Step 0 Python gradient norms are in a similar range to CUDA but not identical (different random seeds/data).

### Remaining Suspects
1. **Backward pass produces wrong gradient directions** after model update -- subtle bug in attention backward, MLP backward, or norm backward that only manifests after weights change
2. **cuDNN SDPA backward** receiving or producing subtly wrong results
3. **key_offset_shift uses overlapping cudaMemcpy2D** -- undefined behavior per CUDA spec, though forward pass appears correct
4. **Data pipeline differences** -- different batch construction could cause systematically different gradient statistics

## Optimizations Applied

| Optimization | Sync ops eliminated/step | Status |
|--------------|--------------------------|--------|
| Pass num_seqs from batch construction | ~16 | Done |
| Cache scalar copies once per step | ~96 | Done |
| GPU-side scalar gradient accumulation | ~88 | Done |
| Pre-allocate h_inputs/h_targets | CPU overhead | Done |
| CUDA event timing | Accuracy | Done |
| cuDNN flash attention | Speed (13x -> 3x) | Done |

## Bugs Fixed

| Bug | Root Cause | Fix |
|-----|-----------|-----|
| val_loss ~532K | gpu_reduce_sum not divided by T | Added `*loss_out /= T` |
| cuBLAS error 15 (NOT_SUPPORTED) | FP8 matmul unsupported | Replaced with BF16 path |
| Misaligned address in scatter_add | `atomicAdd(reinterpret_cast<float*>(&bf16_ptr))` | Native bf16 atomicAdd |
| OOM 143 GB | Allocating val-sized (262K tokens) saved activations | Use train-sized (49K) |
| cuBLAS error 7 (INVALID_VALUE) | Optimizer state not allocated (commented out) | Uncommented alloc |
| Output buffering | C stdio buffering over SSH | `setlinebuf(stdout/stderr)` |
| 13x slowdown | Naive O(T^2) attention | cuDNN SDPA with graph caching |
| ~225 GPU pipeline stalls/step | Blocking cudaMemcpy for scalars/seqlens | Cached host copies + GPU kernel |

## Build & Run

```bash
# On H100 (sm_90a) with CUDA 12.x, cuDNN 9.x
# Requires cudnn-frontend headers: git clone https://github.com/NVIDIA/cudnn-frontend.git
export PATH=/usr/local/cuda/bin:$PATH
make

# Needs data shards in ./data/fineweb10B/
python data/cached_fineweb10B.py 9
./train_gpt
```

## RunPod

Current pod: `ssh -i ~/.ssh/id_ed25519 -p 11096 root@216.243.220.230`
Code location on pod: `/root/modded-nanogpt-vk/`
Pod has 2 data shards (enough for ~750 steps). Temporary diagnostic modifications on pod (VAL_LOSS_EVERY=5, early exit at step 10, extra DIAG prints). Local `train_gpt.cu` is the clean version.
