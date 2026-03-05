"""
Diagnostic: Print gradient norms after step 0 (8 microsteps).
Run with: torchrun --standalone --nproc_per_node=1 diag_grad_norms.py
Compare output with CUDA train_gpt step-0 gradient norms.
"""
import os, sys, torch, time
import torch.distributed as dist

os.environ.setdefault("RANK", "0")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")

sys.path.insert(0, os.path.dirname(__file__))
exec(open("train_gpt.py").read().split("# start the clock")[0])

print("=== Gradient Norm Diagnostic ===", flush=True)

model.train()
train_loader2 = distributed_data_generator(
    args.train_files, TRAINING_STAGES[0].batch_size,
    TRAINING_STAGES[0].train_max_seq_len, grad_accum_steps=grad_accum_steps
)

send_args = training_manager.train_loader_send_args
fwd_args = training_manager.get_forward_args()

for idx in range(grad_accum_steps):
    inputs, targets, cum_seqlens, bigram_inputs, bigram_cpu = train_loader2.send(send_args)
    training_manager.sparse_index_update(0, bigram_cpu)
    fwd_args = training_manager.get_forward_args()
    loss = model(inputs, targets, cum_seqlens, bigram_inputs, fwd_args) * grad_scale
    training_manager.sparse_index_share(0)
    loss.backward()
    del loss

print("=== Step 0 gradient norms (after 8 microsteps) ===", flush=True)

param_map = {
    'attn_bank': 'attn_bank',
    'mlp_bank': 'mlp_bank',
    'embed': 'embed',
    'lm_head': 'lm_head',
    'bigram_embed': 'bigram_embed',
    'value_embeds': 'value_embeds',
    'attn_gate_bank': 'attn_gate_bank',
    've_gate_bank': 've_gate_bank',
    'smear_gate': 'smear_gate',
    'skip_gate': 'skip_gate',
    'scalars': 'scalars',
    'post_lambdas': 'post_lambdas',
    'x0_lambdas': 'x0_lambdas',
    'bigram_lambdas': 'bigram_lambdas',
    'resid_lambdas': 'resid_lambdas',
}

for pname, p in model.named_parameters():
    if pname in param_map:
        if p.grad is not None:
            gnorm = p.grad.float().norm().item()
            numel = p.grad.numel()
            print(f"  grad_norm {param_map[pname]:20s} = {gnorm:.6f}  (numel={numel})", flush=True)
        else:
            print(f"  grad_norm {param_map[pname]:20s} = NONE (no grad!)", flush=True)

print("=== Step 0 parameter norms (before optimizer step) ===", flush=True)
for pname, p in model.named_parameters():
    if pname in ['attn_bank', 'mlp_bank', 'embed', 'lm_head', 'scalars']:
        pnorm = p.data.float().norm().item()
        print(f"  param_norm {pname:20s} = {pnorm:.6f}  (numel={p.numel()})", flush=True)

print("=== End gradient norm diagnostic ===", flush=True)
dist.destroy_process_group()
