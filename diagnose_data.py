"""Diagnose data loading bottleneck."""
import time, torch, glob, numpy as np, threading
from pathlib import Path

BOS_ID = 50256

def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32)
    num_tokens = int(header[2])
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True)
        f.seek(256 * 4)
        f.readinto(tokens.numpy())
    return tokens

def next_multiple_of_n(v, *, n):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)

def get_bigram_hash(x, bigram_vocab_size=50304*5):
    mod = bigram_vocab_size - 1
    x = x.to(torch.int32)
    out = torch.empty_like(x, pin_memory=True)
    out.copy_(x)
    out[0] = mod
    out[1:] = torch.bitwise_xor(36313 * out[1:], 27191 * out[:-1]) % mod
    return out

# Load a shard
files = sorted(glob.glob("data/fineweb10B/fineweb_train_*.bin"))
print(f"Loading {files[0]}...")
t0 = time.perf_counter()
tokens = _load_data_shard(Path(files[0]))
print(f"Loaded {tokens.numel():,} tokens in {time.perf_counter()-t0:.3f}s")

# Build BOS index
t0 = time.perf_counter()
bos_idx = (tokens == BOS_ID).nonzero(as_tuple=True)[0].to(torch.int64).cpu().numpy()
print(f"BOS index ({len(bos_idx):,} entries) built in {time.perf_counter()-t0:.3f}s")

# Simulate data loading for different batch sizes
configs = [
    ("Stage 0", 16384, 896),   # tokens_per_accum, max_seq_len
    ("Stage 1", 32768, 2048),
    ("Stage 2", 49152, 2048),
]

for name, num_tokens_local, max_seq_len in configs:
    print(f"\n{'='*60}")
    print(f"{name}: {num_tokens_local} tokens/accum, max_seq_len={max_seq_len}")
    print(f"{'='*60}")

    times = {}

    # 1. next_batch equivalent
    t0 = time.perf_counter()
    starts, ends = [], []
    idx = 0
    cur_len = 0
    n = len(bos_idx)
    while cur_len <= num_tokens_local:
        cur = bos_idx[idx]
        starts.append(cur)
        end = min(bos_idx[idx + 1] if idx + 1 < n else tokens.numel(),
                  cur + max_seq_len,
                  cur + num_tokens_local - cur_len + 1)
        ends.append(end)
        cur_len += end - cur
        idx += 1
    times['next_batch'] = (time.perf_counter() - t0) * 1000
    print(f"  next_batch: {times['next_batch']:.3f}ms ({len(starts)} docs)")

    # 2. torch.cat
    start_idxs = torch.tensor(starts)
    end_idxs = torch.tensor(ends)

    t0 = time.perf_counter()
    buf = torch.cat([tokens[i:j] for i, j in zip(start_idxs, end_idxs)])
    times['torch_cat'] = (time.perf_counter() - t0) * 1000
    print(f"  torch.cat: {times['torch_cat']:.3f}ms ({buf.numel()} tokens from {len(starts)} slices)")

    _inputs = buf[:-1]
    _targets = buf[1:]

    # 3. Type conversions
    t0 = time.perf_counter()
    _inputs_i32 = _inputs.to(dtype=torch.int32)
    _targets_i64 = _targets.to(dtype=torch.int64)
    times['type_conv'] = (time.perf_counter() - t0) * 1000
    print(f"  type_conv: {times['type_conv']:.3f}ms")

    # 4. Bigram hash
    t0 = time.perf_counter()
    _bigram = get_bigram_hash(_inputs_i32)
    times['bigram_hash'] = (time.perf_counter() - t0) * 1000
    print(f"  bigram_hash: {times['bigram_hash']:.3f}ms")

    # 5. cum_lengths
    t0 = time.perf_counter()
    max_num_docs = next_multiple_of_n(num_tokens_local // 300, n=128)
    cum_lengths = (end_idxs - start_idxs).cumsum(0)
    _cum_lengths = torch.full((max_num_docs,), num_tokens_local)
    _cum_lengths[0] = 0
    _cum_lengths[1:len(cum_lengths) + 1] = cum_lengths
    _cum_lengths = _cum_lengths.to(dtype=torch.int32)
    times['cum_lengths'] = (time.perf_counter() - t0) * 1000
    print(f"  cum_lengths: {times['cum_lengths']:.3f}ms")

    # 6. GPU transfer
    t0 = time.perf_counter()
    _inputs_gpu = _inputs_i32.to(device="cuda", non_blocking=True)
    _targets_gpu = _targets_i64.to(device="cuda", non_blocking=True)
    _cum_gpu = _cum_lengths.to(device="cuda", non_blocking=True)
    _bigram_gpu = _bigram.to(device="cuda", non_blocking=True)
    torch.cuda.synchronize()
    times['gpu_transfer'] = (time.perf_counter() - t0) * 1000
    print(f"  gpu_transfer: {times['gpu_transfer']:.3f}ms")

    total = sum(times.values())
    print(f"  TOTAL: {total:.3f}ms per accum step")
    print(f"  x8 accum steps: {total*8:.3f}ms per training step")
