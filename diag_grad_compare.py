"""
Diagnostic: Save first batch data from Python model's first training step,
run forward+backward, report gradient norms per parameter.
Run with: torchrun --standalone --nproc_per_node=1 diag_grad_compare.py
"""
import torch
import torch.nn.functional as F
import torch.distributed as dist
import os, sys, struct

os.environ.setdefault("RANK", "0")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")

sys.path.insert(0, os.path.dirname(__file__))
exec(open("train_gpt.py").read().split("# start the clock")[0])

print("=== Gradient Comparison Diagnostic ===", flush=True)

# Get 8 microsteps of data (matching a full training step)
train_loader2 = distributed_data_generator(
    args.train_files, TRAINING_STAGES[0].batch_size,
    TRAINING_STAGES[0].train_max_seq_len, grad_accum_steps=grad_accum_steps
)

model.train()
model.zero_grad(set_to_none=True)

# Run 8 microsteps (one full training step)
send_args = training_manager.train_loader_send_args
fwd_args = training_manager.get_forward_args()

for accum in range(grad_accum_steps):
    inputs, targets, cum_seqlens, bigram_inputs, bigram_cpu = train_loader2.send(send_args)
    training_manager.sparse_index_update(0, bigram_cpu)
    fwd_args = training_manager.get_forward_args()
    loss = model(inputs, targets, cum_seqlens, bigram_inputs, fwd_args) * grad_scale
    loss.backward()

    if accum == 0:
        print(f"  microstep 0: loss={loss.item()/grad_scale:.6f}, T={inputs.shape[0]}", flush=True)
        # Save first microstep data for CUDA replay
        inputs_np = inputs.cpu().numpy()
        targets_np = targets.cpu().numpy()
        cum_seqlens_np = cum_seqlens.cpu().numpy()
        bigram_inputs_np = bigram_inputs.cpu().numpy()
        with open('/tmp/diag_batch0_inputs.bin', 'wb') as f:
            f.write(inputs_np.astype('int32').tobytes())
        with open('/tmp/diag_batch0_targets.bin', 'wb') as f:
            f.write(targets_np.astype('int64').tobytes())
        with open('/tmp/diag_batch0_cum_seqlens.bin', 'wb') as f:
            f.write(cum_seqlens_np.astype('int32').tobytes())
        with open('/tmp/diag_batch0_bigram_inputs.bin', 'wb') as f:
            f.write(bigram_inputs_np.astype('int32').tobytes())
        with open('/tmp/diag_batch0_meta.txt', 'w') as f:
            f.write(f"T={inputs.shape[0]}\n")
            f.write(f"num_seqs={cum_seqlens.shape[0]-1}\n")
        print(f"  Saved batch to /tmp/diag_batch0_*.bin", flush=True)

# Print gradient norms
print("\n=== Step 0 gradient norms (after 8 microsteps) ===", flush=True)
name_map = {
    '_orig_mod.attn_bank': 'attn_bank',
    '_orig_mod.mlp_bank': 'mlp_bank',
    '_orig_mod.embed.weight': 'embed',
    '_orig_mod.lm_head.weight': 'lm_head',
    '_orig_mod.scalars': 'scalars',
    '_orig_mod.post_lambdas': 'post_lambdas',
    '_orig_mod.x0_lambdas': 'x0_lambdas',
    '_orig_mod.resid_lambdas': 'resid_lambdas',
    '_orig_mod.attn_gate_bank': 'attn_gate_bank',
    '_orig_mod.ve_gate_bank': 've_gate_bank',
    '_orig_mod.smear_gate.weight': 'smear_gate',
    '_orig_mod.skip_gate.weight': 'skip_gate',
    '_orig_mod.bigram_lambdas': 'bigram_lambdas',
    '_orig_mod.bigram_embed.weight': 'bigram_embed',
    '_orig_mod.value_embeds': 'value_embeds',
}

for name, param in model.named_parameters():
    if param.grad is not None:
        short = name_map.get(name, name)
        norm = param.grad.float().norm().item()
        print(f"  grad_norm {short:30s} = {norm:.6f}", flush=True)
    else:
        short = name_map.get(name, name)
        print(f"  grad_norm {short:30s} = None", flush=True)

# Also print parameter norms for sanity
print("\n=== Parameter norms ===", flush=True)
for name, param in model.named_parameters():
    short = name_map.get(name, name)
    norm = param.float().norm().item()
    print(f"  param_norm {short:30s} = {norm:.6f}", flush=True)

print("\n=== End diagnostic ===", flush=True)
dist.destroy_process_group()
