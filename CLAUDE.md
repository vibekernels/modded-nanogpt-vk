# CLAUDE.md

## Project Overview

Modded-NanoGPT: competitive speedrun to train a GPT-2-scale language model to 3.28 val loss on FineWeb using 8×H100 GPUs. Current record is under 90 seconds.

Single file training script: `train_gpt.py`. Custom Triton kernels in `triton_kernels.py`. Flash Attention 3 loaded via `kernels` package from HuggingFace.

## Setup

### Requirements

- NVIDIA H100 GPU(s) with driver supporting CUDA 12.6+
- Python 3.11+

### Installation

```bash
pip install -r requirements.txt
```

This installs `torch==2.10` (ships with CUDA 12.8 and Triton 3.2). If triton's `TensorDescriptor` import fails, install `triton==3.4.0` separately.

### Data

```bash
# Full dataset (10B tokens, 103 shards, ~20GB):
python data/cached_fineweb10B.py

# Subset (first N shards of 100M tokens each):
python data/cached_fineweb10B.py 9
```

Data downloads to `data/fineweb10B/` as `.bin` files (uint16 GPT-2 tokens with 1024-byte header).

### Running

```bash
# 8×H100 (production, ~90 seconds):
./run.sh

# 1×H100 (development/profiling, ~10 minutes with GPU data prep optimization):
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

## Architecture

- Single-file model + training loop in `train_gpt.py` (~2040 lines)
- `torch.compile` with `fullgraph=True` — kernel warmup takes ~7 minutes on first run
- 4 training stages with increasing batch size (8→16→24 ×2048×8 tokens), sequence length (896→2048), and sliding window sizes
- `grad_accum_steps = 8 // world_size` (8 accum steps on 1 GPU, 1 on 8 GPUs)
- FP8 matmuls, Flash Attention 3, fused Triton kernels for MLP and cross-entropy
- Muon optimizer with Polar Express orthogonalization

## Key Code Locations

- **Data pipeline**: `distributed_data_generator()` (~line 1506), `Shard` class (~line 1418), `get_bigram_hash()` (~line 1485)
- **Model**: GPT class (~line 1100-1400)
- **Training loop**: ~line 1976-2035
- **Training stages**: `TRAINING_STAGES` list (~line 1675)
- **Optimizer**: `TrainingManager` class (~line 1780)
- **Sparse comms** (8-GPU bigram gradient reduce-scatter): `_sparse_comms_active()` (~line 252), only active when `world_size==8` and `grad_accum_steps==1`

## Development Notes

- `expandable_segments:True` is set at line 22 for CUDA memory allocator
- Validation loss is evaluated every 250 steps
- Logs go to `logs/<run_id>.txt`
- The training script reads itself and logs its own source code for reproducibility
- When modifying data prep, be aware that `_sparse_comms_active()` controls whether `_bigram_cpu` (GPU→CPU copy) is needed; it's only True on 8-GPU runs
