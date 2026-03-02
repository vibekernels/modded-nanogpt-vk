"""
Benchmark: CPU vs GPU data preparation in distributed_data_generator.
Measures per-accumulation-step data prep time for each training stage config.
"""
import time
import torch
import numpy as np
from pathlib import Path

torch.set_float32_matmul_precision("high")

# ── Config matching train_gpt.py ──────────────────────────────────────────────
BOS_ID = 50256
BIGRAM_VOCAB_SIZE = 50304 * 5

STAGES = [
    {"name": "Stage 0", "batch_size": 8 * 2048 * 8, "max_seq_len": 896,  "grad_accum": 8},
    {"name": "Stage 1", "batch_size": 16 * 2048 * 8, "max_seq_len": 2048, "grad_accum": 8},
    {"name": "Stage 2", "batch_size": 24 * 2048 * 8, "max_seq_len": 2048, "grad_accum": 8},
]

# ── Data loading (copied from train_gpt.py) ──────────────────────────────────
def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32)
    assert header[0] == 20240520
    num_tokens = int(header[2])
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True)
        f.seek(256 * 4)
        f.readinto(tokens.numpy())
    return tokens

# ── Original CPU bigram hash ──────────────────────────────────────────────────
def get_bigram_hash_cpu(x):
    mod = BIGRAM_VOCAB_SIZE - 1
    x = x.to(torch.int32)
    out = torch.empty_like(x, pin_memory=True)
    out.copy_(x)
    out[0] = mod
    out[1:] = torch.bitwise_xor(36313 * out[1:], 27191 * out[:-1]) % mod
    return out

# ── New GPU bigram hash ───────────────────────────────────────────────────────
def get_bigram_hash_gpu(x):
    mod = BIGRAM_VOCAB_SIZE - 1
    x = x.to(torch.int32)
    out = x.clone()
    out[0] = mod
    out[1:] = torch.bitwise_xor(36313 * out[1:], 27191 * out[:-1]) % mod
    return out

# ── Correctness check ────────────────────────────────────────────────────────
def verify_correctness(tokens, n=49152):
    """Verify GPU path produces identical results to CPU path."""
    buf = tokens[:n + 1]

    # CPU path (original)
    inputs_cpu = buf[:-1].to(dtype=torch.int32)
    targets_cpu = buf[1:].to(dtype=torch.int64)
    bigram_cpu = get_bigram_hash_cpu(inputs_cpu)

    # GPU path (new)
    buf_gpu = buf.to(device="cuda")
    inputs_gpu = buf_gpu[:-1].to(dtype=torch.int32)
    targets_gpu = buf_gpu[1:].to(dtype=torch.int64)
    bigram_gpu = get_bigram_hash_gpu(inputs_gpu)

    assert torch.equal(inputs_cpu, inputs_gpu.cpu()), "inputs mismatch!"
    assert torch.equal(targets_cpu, targets_gpu.cpu()), "targets mismatch!"
    assert torch.equal(bigram_cpu, bigram_gpu.cpu()), "bigram hash mismatch!"
    print("Correctness verified: CPU and GPU paths produce identical results.\n")

# ── Benchmark functions ───────────────────────────────────────────────────────
def bench_cpu_data_prep(buf, warmup=3, iters=10):
    """Original CPU data prep: type conversions + bigram hash on CPU, then transfer."""
    # Warmup
    for _ in range(warmup):
        _inputs = buf[:-1].to(dtype=torch.int32)
        _targets = buf[1:].to(dtype=torch.int64)
        _bigram = get_bigram_hash_cpu(_inputs)
        _inputs.to(device="cuda", non_blocking=True)
        _targets.to(device="cuda", non_blocking=True)
        _bigram.to(device="cuda", non_blocking=True)
        _ = _bigram.numpy()
        torch.cuda.synchronize()

    # Timed
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        _inputs = buf[:-1].to(dtype=torch.int32)
        _targets = buf[1:].to(dtype=torch.int64)
        _bigram = get_bigram_hash_cpu(_inputs)
        _inputs.to(device="cuda", non_blocking=True)
        _targets.to(device="cuda", non_blocking=True)
        _bigram.to(device="cuda", non_blocking=True)
        _ = _bigram.numpy()

        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    return times

def bench_gpu_data_prep(buf, warmup=3, iters=10):
    """New GPU data prep: transfer raw uint16, then type conversions + bigram hash on GPU."""
    # Warmup
    for _ in range(warmup):
        buf_gpu = buf.to(device="cuda", non_blocking=True)
        _inputs = buf_gpu[:-1].to(dtype=torch.int32)
        _targets = buf_gpu[1:].to(dtype=torch.int64)
        _bigram = get_bigram_hash_gpu(_inputs)
        torch.cuda.synchronize()

    # Timed
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        buf_gpu = buf.to(device="cuda", non_blocking=True)
        _inputs = buf_gpu[:-1].to(dtype=torch.int32)
        _targets = buf_gpu[1:].to(dtype=torch.int64)
        _bigram = get_bigram_hash_gpu(_inputs)

        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    return times

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading data shard...")
    tokens = _load_data_shard(Path("data/fineweb10B/fineweb_train_000001.bin"))
    print(f"Loaded {tokens.numel():,} tokens\n")

    verify_correctness(tokens)

    print(f"{'Stage':<12} {'Tokens/accum':>14} {'CPU (ms)':>12} {'GPU (ms)':>12} {'Speedup':>10}")
    print("-" * 65)

    for stage in STAGES:
        num_tokens_per_accum = stage["batch_size"] // stage["grad_accum"]
        # Simulate single-rank: num_tokens_local = num_tokens_per_accum (world_size=1)
        n = num_tokens_per_accum

        # Create a buffer of the right size (simulating torch.cat output)
        buf = tokens[:n + 1].contiguous()

        cpu_times = bench_cpu_data_prep(buf, warmup=5, iters=20)
        gpu_times = bench_gpu_data_prep(buf, warmup=5, iters=20)

        cpu_med = np.median(cpu_times) * 1000
        gpu_med = np.median(gpu_times) * 1000
        speedup = cpu_med / gpu_med

        print(f"{stage['name']:<12} {n:>14,} {cpu_med:>11.2f} {gpu_med:>11.2f} {speedup:>9.1f}x")

    # Also estimate total step time impact for Stage 2
    print("\n--- Stage 2 Total Step Time Estimate ---")
    n = STAGES[2]["batch_size"] // STAGES[2]["grad_accum"]
    accum_steps = STAGES[2]["grad_accum"]
    buf = tokens[:n + 1].contiguous()

    cpu_times = bench_cpu_data_prep(buf, warmup=5, iters=20)
    gpu_times = bench_gpu_data_prep(buf, warmup=5, iters=20)

    cpu_per_accum = np.median(cpu_times) * 1000
    gpu_per_accum = np.median(gpu_times) * 1000
    gpu_compute_per_accum = 120  # ~120ms from profiling (RESEARCH.md)

    cpu_step = (cpu_per_accum + gpu_compute_per_accum) * accum_steps
    gpu_step = (gpu_per_accum + gpu_compute_per_accum) * accum_steps

    print(f"Data prep per accum:  CPU={cpu_per_accum:.1f}ms  GPU={gpu_per_accum:.2f}ms")
    print(f"GPU compute per accum: ~{gpu_compute_per_accum}ms")
    print(f"Estimated step time:  CPU={cpu_step:.0f}ms  GPU={gpu_step:.0f}ms")
    print(f"Stage 2 speedup:      {cpu_step/gpu_step:.1f}x")
