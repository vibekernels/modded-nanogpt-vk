# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A high-performance GPT training fork of [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) optimized for 1xH100 (and up to 8xH100). Trains on FineWeb 10B tokens with aggressive speed optimizations — the entire codebase lives in two files: `train_gpt.py` (~2000 lines) and `triton_kernels.py` (~850 lines).

## Commands

### Data Preparation
```bash
python data/cached_fineweb10B.py <num_shards>  # Download shards (use 9 for full training)
```

### Training
```bash
# Single GPU
torchrun --standalone --nproc_per_node=1 train_gpt.py

# Multi-GPU (8x)
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Docker
Docker image is built via GitHub Actions on push to `master` and published to `ghcr.io/vibekernels/modded-nanogpt-vk`. The image targets RunPod with SSH access via `PUBLIC_KEY` env var. When creating pods with `runpodctl`, use `--ports "22/tcp"` to get full SSH (with SCP/SFTP support) rather than RunPod's proxied basic SSH. Always create a new pod rather than connecting to an existing one — other pods may belong to other agents running their own experiments. Prefer US regions for RunPod pods — container images load significantly faster there. Always terminate RunPod pods once they're no longer needed as they are expensive.

## Architecture

### Two-File Design

Everything is intentionally in two files — there are no module directories or package structure. This is a deliberate design choice inherited from the modded-nanogpt competition format.

- **`train_gpt.py`**: Model definition, optimizer, training loop, data pipeline, and all configuration
- **`triton_kernels.py`**: Custom Triton kernels (XXT, XTX, fused ReLU², softcapped cross-entropy, transpose ops)

### Model Architecture (GPT class)

- 11 transformer layers, 6 heads, head_dim=128, model_dim=768
- **Parameter banks**: `attn_bank` and `mlp_bank` store all layer weights as stacked tensors (not per-layer modules) to reduce gradient accumulation overhead
- **Dual-lane residual streams** (layers 7-10): Lane 0 (attn) + Lane 1 (mlp) run in parallel with `post_lambdas` scaling
- **Value embeddings**: 5 unique spherical-Gaussian-initialized gates (`ve_gate_bank`) with per-head normalization
- **Rotational position embeddings (YaRN)**: Dynamic window sizes, paired-head variant on layers 0, 2, 5, 9
- **FP8 quantization** (`CastedLinearT`): Transposed weight layout for faster gradient accumulation
- **Multi-token prediction (MTP)**: Weights transition across stages from 3-token to 1-token prediction
- **Bigram hash embeddings**: Sparse vocabulary of `50304 * 5` for character-pair features
- Attention skipped at layer 6

### Training Pipeline

**4-stage schedule** (1,490 total steps):
1. Stage 1 (1/3): seq_len=896, batch=8×2048×8, window=(1,3)
2. Stage 2 (1/3): seq_len=2048, batch=16×2048×8, window=(3,7)
3. Stage 3 (1/3): seq_len=2048, batch=24×2048×8, window=(5,11)
4. Extension (40 steps): window=(6,13), then post-YaRN extension to window=20

Key transitions: embed/lm_head split at 2/3 of training, MTP heads pruned across stages, LR tied to batch size with 60% cooldown.

### Optimizer: NorMuonAndAdam (TrainingManager)

Hybrid optimizer — each named parameter has its own config in `param_table`:
- **Muon** (Newton-Schulz orthogonalization): Used for `attn_bank` and `mlp_bank` (the large weight matrices)
- **Adam**: Used for scalars, gates, embeddings, lm_head — each with specific LR multipliers, betas, and weight decay
- Communication: `sharded` (reduce_scatter/all_gather), `replicated` (all_reduce), or `sharded_sparse`
- Adam params only step on odd iterations
- Work order processes small params first while large reduce ops overlap

### Data Pipeline

`distributed_data_generator` is a Python generator that yields `(inputs, targets, cum_lengths, bigram_inputs, bigram_cpu)`. It loads `.bin` shards asynchronously, supports BOS-aligned batching, and does GPU-side type conversion + bigram hashing when sparse comms are not active.

### Warmup Phase

The training loop has a ~7-minute warmup that compiles the model and warms Triton kernels before resetting optimizer state and starting timed training. This is normal behavior, not a hang.

## Key Conventions

- `records/track_1_short/` contains 85+ dated experiment records documenting each incremental optimization
- Logs go to `logs/{run_id}/` with the full source code snapshot
- Validation runs every 250 steps on 10.48M fixed tokens
- `grad_accum_steps = 8 // world_size` — always 8 effective accumulation steps regardless of GPU count
- The codebase uses `torch.compile` and `torch._dynamo` extensively; `recompile_limit` is set to 64
