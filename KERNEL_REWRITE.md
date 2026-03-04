# CUDA C Kernel Rewrite Progress

Pure CUDA C reimplementation of the modded-nanogpt-vk training pipeline, eliminating all Python/PyTorch/Triton dependencies.

## File Structure

| File | Lines | Purpose |
|------|-------|---------|
| `train_gpt.cu` | ~2,750 | Model, forward/backward, optimizer, training loop, data loading |
| `kernels.cu` | ~1,970 | Custom CUDA kernels (ports of triton_kernels.py + extras) |
| `train_gpt.h` | ~410 | Shared structs, constants, forward declarations |
| `kernels.h` | ~270 | Kernel launch wrapper declarations |
| `Makefile` | ~60 | Build system (nvcc sm_90a, links cublas/cudnn/curand) |

## What Works

### Forward Pass
- Embedding lookup + bigram hash + smear gate
- RMS norm (custom kernel)
- 11-layer transformer with dual-lane residual streams (layers 7-10 parallel)
- Attention: QKV projection (cuBLAS), QK RMS norm, YaRN RoPE, key offset, value embed gating, attention (naive kernel), output gating, output projection (cuBLAS)
- MLP: c_fc matmul (cuBLAS), fused ReLU^2 (custom kernel), c_proj matmul (cuBLAS)
- Skip connection (layer 3 save, layer 6 inject)
- Backout subtraction, lane averaging
- Final RMS norm, lm_head matmul (BF16 via cuBLAS)
- Softcapped cross-entropy loss with multi-token prediction (custom kernel)
- Validation: 10.48M tokens, val_loss=10.8301 at init (matches ln(50304)=10.826)

### Backward Pass
- Softcapped CE backward (BF16 variant, custom kernel)
- lm_head backward (cuBLAS BF16: grad_x via gemm_bf16_Bt, grad_w via gemm_bf16_At)
- Final RMS norm backward (custom kernel)
- Backout/lane-merge backward with scalar gradient accumulation
- Per-layer backward (10 → 0): attention backward (naive kernel), MLP backward, residual/lambda/gate scalar gradients via bf16_dot_product
- Embedding backward: scatter_add with native bf16 atomicAdd (SM 9.0)
- All 8 gradient accumulation micro-batches complete without errors

### Optimizer
- **NorMuon** (attn_bank, mlp_bank): Nesterov momentum, Polar Express orthogonalization (4 iterations with Turbo-Muon AOL coefficients), variance reduction, cautious weight decay with mantissa-tracked BF16 updates
- **Adam** (scalars, gates, embeddings, lm_head): bias-corrected Adam with per-param LR multipliers, betas, weight decay. Steps on odd iterations only.
- Embed/lm_head tying with transpose_add/transpose_copy

### Training Loop
- 4-stage schedule with YaRN window extensions
- MTP weight interpolation per step
- BOS-aligned batch construction from mmap'd .bin shards
- Loss confirmed dropping: 18.94 → 11.9 over 58 steps (18.94 = 1.75 * ln(50304), correct for 3-token MTP weights [1.0, 0.5, 0.25])

### Custom CUDA Kernels (kernels.cu)
- RMS norm forward/backward
- XXT, XTX, ba+cAA (symmetric matmul for Polar Express)
- Fused ReLU^2 forward/backward
- Softcapped CE forward/backward (FP8 e5m2 and BF16 variants)
- Transpose copy/add (32x32 tiled)
- RoPE apply + YaRN table computation
- Embedding gather/scatter_add, bigram hash
- Sigmoid gate, fused_add_scale, fused_add3, smear forward
- Adam update, Muon cautious update, variance reduction
- Nesterov momentum, norm divide, tensor norm, scale tensor
- BF16/FP8 conversions, GPU reduce sum, bf16 dot product
- Elementwise multiply broadcast, fused gate add
- Naive varlen attention forward/backward (placeholder)
- Key offset shift

## Performance

| Metric | CUDA C | Python Baseline |
|--------|--------|-----------------|
| Step time | ~5.96s | ~0.45s |
| Slowdown | 13x | — |
| Bottleneck | Naive O(T^2) attention | Flash attention (cuDNN) |
| GPU memory | ~35 GB | ~37 GB |

## What Doesn't Work Yet

### cuDNN Flash Attention (Critical)
The naive attention kernel is O(T^2) per position — accounts for the entire 13x slowdown. Need to replace with cuDNN frontend flash attention API (varlen + causal mask + sliding window). The cuDNN handles and headers are already linked; just need the descriptor setup and launch code.

### FP8 lm_head Matmul
`cublasLtMatmul` with FP8 e4m3 returned `CUBLAS_STATUS_NOT_SUPPORTED` on the test pod (CUDA 12.6). Currently using BF16 fallback. May need different cuBLAS LT descriptor configuration or a newer CUDA toolkit.

### Full Training Validation
- Only tested with 1 of 9 data shards (runs out of data around step ~750)
- Need all 9 shards to complete 1490 steps
- Target: val_loss ~3.2802 (baseline)

## Bugs Fixed

| Bug | Root Cause | Fix |
|-----|-----------|-----|
| val_loss ~532K | gpu_reduce_sum not divided by T | Added `*loss_out /= T` |
| cuBLAS error 15 (NOT_SUPPORTED) | FP8 matmul unsupported | Replaced with BF16 path |
| Misaligned address in scatter_add | `atomicAdd(reinterpret_cast<float*>(&bf16_ptr))` | Native bf16 atomicAdd |
| OOM 143 GB | Allocating val-sized (262K tokens) saved activations | Use train-sized (49K) |
| cuBLAS error 7 (INVALID_VALUE) | Optimizer state not allocated (commented out) | Uncommented alloc |
| Output buffering | C stdio buffering over SSH | `setlinebuf(stdout/stderr)` |

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
