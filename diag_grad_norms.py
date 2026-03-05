"""
Diagnostic: Print gradient and parameter norms after step 0.
Run with: torchrun --standalone --nproc_per_node=1 diag_grad_norms.py
"""
import torch
import torch.distributed as dist
import os, sys

# Patch to run on single GPU
os.environ.setdefault("RANK", "0")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")

# Import everything from train_gpt
sys.path.insert(0, os.path.dirname(__file__))
exec(open("train_gpt.py").read().split("# start the clock")[0])

# Now run one training step and print norms
print("=== Running 1 training step for gradient norm diagnostics ===", flush=True)

train_loader2 = distributed_data_generator(
    args.train_files, TRAINING_STAGES[0].batch_size,
    TRAINING_STAGES[0].train_max_seq_len, grad_accum_steps=grad_accum_steps
)

model.train()
# Zero grads
model.zero_grad(set_to_none=True)

# Run 8 microsteps (one full training step)
for idx in range(grad_accum_steps):
    send_args = training_manager.train_loader_send_args
    inputs, targets, cum_seqlens, bigram_inputs, bigram_cpu = train_loader2.send(send_args)
    training_manager.sparse_index_update(0, bigram_cpu)
    loss = model(inputs, targets, cum_seqlens, bigram_inputs, training_manager.get_forward_args()) * grad_scale
    training_manager.sparse_index_share(0)
    loss.backward()
    del loss

print("=== Step 0 gradient norms (after 8 microsteps) ===", flush=True)
for name, param in model.named_parameters():
    label = getattr(param, 'label', name)
    if param.grad is not None:
        norm = param.grad.float().norm().item()
        print(f"  grad_norm {label:20s} = {norm:.6f}")
    else:
        print(f"  grad_norm {label:20s} = None (no grad)")

# Per-element scalar gradient values
scalars_grad = model.scalars.grad
if scalars_grad is not None:
    print("=== Per-element scalar gradients ===", flush=True)
    for j in range(min(scalars_grad.numel(), 25)):
        if j < 22:
            layer_idx = j // 2
            which = "sa_lambda0" if j % 2 == 0 else "sa_lambda1"
            print(f"  scalars[{j}] ({which}[{layer_idx}]) = {scalars_grad[j].item():.6f}")
        elif j == 22:
            print(f"  scalars[{j}] (smear_lambda) = {scalars_grad[j].item():.6f}")
        elif j == 23:
            print(f"  scalars[{j}] (backout_lambda) = {scalars_grad[j].item():.6f}")
        elif j == 24:
            print(f"  scalars[{j}] (skip_lambda_raw) = {scalars_grad[j].item():.6f}")

print("=== Step 0 parameter norms (before optimizer step) ===", flush=True)
for name, param in model.named_parameters():
    label = getattr(param, 'label', name)
    norm = param.data.float().norm().item()
    print(f"  param_norm {label:20s} = {norm:.6f}")

print("=== End diagnostics ===", flush=True)
dist.destroy_process_group()
