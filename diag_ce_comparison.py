"""
Diagnostic: Compare CE backward with FP8 vs BF16 logits.
Run with: torchrun --standalone --nproc_per_node=1 diag_ce_comparison.py
"""
import torch
import torch.nn.functional as F
import torch.distributed as dist
import os, sys

os.environ.setdefault("RANK", "0")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")

sys.path.insert(0, os.path.dirname(__file__))
exec(open("train_gpt.py").read().split("# start the clock")[0])

print("=== CE backward comparison: FP8 vs BF16 logits ===", flush=True)

train_loader2 = distributed_data_generator(
    args.train_files, TRAINING_STAGES[0].batch_size,
    TRAINING_STAGES[0].train_max_seq_len, grad_accum_steps=grad_accum_steps
)

model.train()

# Get one batch
send_args = training_manager.train_loader_send_args
inputs, targets, cum_seqlens, bigram_inputs, bigram_cpu = train_loader2.send(send_args)
training_manager.sparse_index_update(0, bigram_cpu)

fwd_args = training_manager.get_forward_args()
mtp_weights = fwd_args.mtp_weights

# Run forward to get the normed x (before lm_head)
with torch.no_grad():
    # Run partial forward to get x
    assert inputs.ndim == 1
    ws_short, ws_long = fwd_args.ws_short, fwd_args.ws_long
    skip_connections = []
    skip_in = [3]
    skip_out = [6]
    x_backout = None
    backout_layer = 7
    bm_sizes = [ws_short, ws_short, ws_short, ws_long, ws_short, ws_short, None, ws_short, ws_short, ws_short, ws_long]
    key_offset = [b==ws_long for b in bm_sizes]

    sa_lambdas = model.scalars[: 2 * model.num_layers].view(-1, 2)
    smear_lambda = model.scalars[2 * model.num_layers]
    backout_lambda = model.scalars[2 * model.num_layers + 1]
    skip_lambda = model.scalars[2 * model.num_layers + 2]
    resid_lambdas_attn = model.resid_lambdas[:, 0].bfloat16().unbind(0)
    resid_lambdas_mlp  = model.resid_lambdas[:, 1].bfloat16().unbind(0)
    post_lambdas_attn_ln0 = model.post_lambdas[:, 0, 0].bfloat16().unbind(0)
    post_lambdas_attn_ln1 = model.post_lambdas[:, 0, 1].bfloat16().unbind(0)
    post_lambdas_mlp_ln0  = model.post_lambdas[:, 1, 0].bfloat16().unbind(0)
    post_lambdas_mlp_ln1  = model.post_lambdas[:, 1, 1].bfloat16().unbind(0)
    x0_lambdas = model.x0_lambdas.bfloat16().unbind(0)
    bigram_lambdas = model.bigram_lambdas.bfloat16().unbind(0)
    ag = [w.bfloat16() for w in model.attn_gate_bank.unbind(0)]
    veg = [w.bfloat16() for w in model.ve_gate_bank.unbind(0)]
    attn_gates = ag[:6] + [None] + ag[6:]
    ve_gates = [None] + [veg[0], veg[1]] + [None] * (model.num_layers - 6) + [veg[2], veg[3], veg[4]]
    attn_weights = model.attn_bank.unbind(0)
    mlp_all = model.mlp_bank.flatten(0, 1).unbind(0)
    mlp_fcs = mlp_all[0::2]
    mlp_projs = mlp_all[1::2]

    x = model.embed(inputs)
    x0_bigram = model.bigram_embed(bigram_inputs)[None]
    ve = model.value_embeds.view(5, model.vocab_size, -1)[:, inputs]
    ve = [None, ve[0], ve[1]] + [None] * (model.num_layers - 6) + [ve[2], ve[3], ve[4]]

    smear_gate_out = smear_lambda * torch.sigmoid(model.smear_gate(x[1:, :model.smear_gate.weight.size(-1)]))
    x = torch.cat([x[:1], x[1:] + smear_gate_out * x[:-1]])
    x = x0 = norm(x[None])

    lane0 = x0 + x0_bigram * bigram_lambdas[0]
    lane1 = None
    x0_inject = (x0 * x0_lambdas[0],) + tuple(x0 * x0_lambdas[i] + x0_bigram * bigram_lambdas[i] for i in range(1, model.num_layers))

    for i in range(model.num_layers):
        yarn = model.yarn_paired_head if i in model.paired_head_layers else model.yarn
        attn_args = AttnArgs(
            ve=ve[i], sa_lambdas=sa_lambdas[i], seqlens=cum_seqlens,
            bm_size=bm_sizes[i], yarn=yarn, key_offset=key_offset[i],
            attn_gate_w=attn_gates[i], ve_gate_w=ve_gates[i],
            train_max_seq_len=fwd_args.train_max_seq_len
        )
        qkvo_w = attn_weights[i - (i > 6)] if i != 6 else None
        c_fc = mlp_fcs[i]
        c_proj = mlp_projs[i]

        if i == model.parallel_start:
            lane1 = lane0

        if i in skip_out:
            skip_gate_out = torch.sigmoid(skip_lambda) * 2 * torch.sigmoid(model.skip_gate(x0[..., :model.skip_gate.weight.size(-1)]))
            skip_val = skip_connections.pop()
            lane0 = lane0 + skip_gate_out * skip_val
            if lane1 is not None:
                lane1 = lane1 + skip_gate_out * skip_val

        attn = model.attn_paired if i in model.paired_head_layers else model.attn

        post_attn = None
        if i == 6:
            post_attn = lane0
            lane0 = resid_lambdas_mlp[i] * lane0 + post_lambdas_mlp_ln0[i] * ReLUSqrdMLP(norm(lane0), c_fc, c_proj)
        elif i < model.parallel_start:
            attn_out = attn(norm(lane0), attn_args, qkvo_w)
            lane0 = resid_lambdas_attn[i] * lane0 + attn_out + x0_inject[i]
            post_attn = lane0
            lane0 = resid_lambdas_mlp[i] * lane0 + post_lambdas_mlp_ln0[i] * ReLUSqrdMLP(norm(lane0), c_fc, c_proj)
        else:
            attn_out = attn(norm(lane0), attn_args, qkvo_w)
            lane0 = resid_lambdas_attn[i] * lane0 + post_lambdas_attn_ln0[i] * attn_out + x0_inject[i]
            lane1 = resid_lambdas_attn[i] * lane1 + post_lambdas_attn_ln1[i] * attn_out
            post_attn = lane0
            mlp_out = ReLUSqrdMLP(norm(lane1), c_fc, c_proj)
            lane0 = resid_lambdas_mlp[i] * lane0 + post_lambdas_mlp_ln0[i] * mlp_out
            lane1 = resid_lambdas_mlp[i] * lane1 + post_lambdas_mlp_ln1[i] * mlp_out

        if i in skip_in:
            skip_connections.append(post_attn)
        if i == backout_layer:
            x_backout = lane0

    x_final = (lane0 + lane1) * 0.5
    x_final -= backout_lambda * x_backout
    x_normed = norm(x_final)

# Now compare FP8 vs BF16 logits and their backward gradients
x_normed_flat = x_normed.view(-1, x_normed.size(-1)).detach().requires_grad_(True)
T = x_normed_flat.size(0)

# === Method 1: FP8 logits (matching Python's FusedSoftcappedCrossEntropy) ===
x_f8 = x_normed_flat.detach().div(model.lm_head.x_s).to(torch.float8_e4m3fn)
w_f8 = model.lm_head.weight.detach().div(model.lm_head.w_s).to(torch.float8_e4m3fn)
w_f8_col = w_f8.T.contiguous().T
logits_fp8 = torch._scaled_mm(
    x_f8, w_f8_col,
    out_dtype=torch.bfloat16,
    scale_a=torch.tensor(model.lm_head.x_s, dtype=torch.float32, device=x_normed_flat.device),
    scale_b=torch.tensor(model.lm_head.w_s, dtype=torch.float32, device=x_normed_flat.device),
    use_fast_accum=True,
).detach()

# === Method 2: BF16 logits (matching CUDA) ===
logits_bf16 = (x_normed_flat.detach().bfloat16() @ model.lm_head.weight.detach().bfloat16()).detach()

print(f"Logits FP8 norm: {logits_fp8.float().norm().item():.6f}", flush=True)
print(f"Logits BF16 norm: {logits_bf16.float().norm().item():.6f}", flush=True)
print(f"Logits diff norm: {(logits_fp8.float() - logits_bf16.float()).norm().item():.6f}", flush=True)

# Compute CE backward for both
# Use the Triton CE forward/backward kernels
from triton_kernels import fused_softcapped_entropy_fwd_kernel, fused_softcapped_entropy_bwd_kernel
import triton

A, B_val, C = 23.0, 5.0, 7.5
grad_s = model.lm_head.grad_s
n_predict = mtp_weights.shape[0] if mtp_weights is not None else 1
if mtp_weights is None:
    mtp_weights = torch.tensor([1.0], device=logits_fp8.device, dtype=torch.float32)

# CE forward + backward for FP8 logits
losses_fp8 = torch.empty(T, dtype=torch.float32, device=logits_fp8.device)
lse_fp8 = torch.empty(T, dtype=torch.float32, device=logits_fp8.device)
fused_softcapped_entropy_fwd_kernel[(T,)](
    logits_fp8.contiguous(), losses_fp8, lse_fp8, targets.contiguous(), mtp_weights.contiguous(),
    logits_fp8.stride(0), logits_fp8.stride(1),
    T, 50304, n_predict,
    A, B_val, C,
    BLOCK_SIZE=1024, num_warps=2
)

grad_output_fp8 = torch.full((T,), grad_scale, dtype=torch.float32, device=logits_fp8.device)
grad_logits_fp8 = torch.empty((T, 50304), dtype=torch.float8_e5m2, device=logits_fp8.device)
fused_softcapped_entropy_bwd_kernel[(T,)](
    grad_logits_fp8, grad_output_fp8, lse_fp8, logits_fp8.contiguous(), targets.contiguous(), mtp_weights.contiguous(),
    logits_fp8.stride(0), logits_fp8.stride(1), grad_logits_fp8.stride(0), grad_logits_fp8.stride(1),
    T, 50304, n_predict,
    A, B_val, C, grad_s,
    BLOCK_SIZE=1024, num_warps=2
)

# Dequantize grad_logits_fp8 and multiply by grad_s to get true gradient
grad_logits_fp8_true = grad_logits_fp8.float() * grad_s
print(f"CE grad_logits (FP8 logits, true scale): {grad_logits_fp8_true.norm().item():.6f}", flush=True)

# CE forward + backward for BF16 logits
losses_bf16 = torch.empty(T, dtype=torch.float32, device=logits_bf16.device)
lse_bf16 = torch.empty(T, dtype=torch.float32, device=logits_bf16.device)
fused_softcapped_entropy_fwd_kernel[(T,)](
    logits_bf16.contiguous(), losses_bf16, lse_bf16, targets.contiguous(), mtp_weights.contiguous(),
    logits_bf16.stride(0), logits_bf16.stride(1),
    T, 50304, n_predict,
    A, B_val, C,
    BLOCK_SIZE=1024, num_warps=2
)

# BF16 CE backward (no FP8 quantization)
grad_output_bf16 = torch.full((T,), grad_scale, dtype=torch.float32, device=logits_bf16.device)
grad_logits_bf16_out = torch.empty((T, 50304), dtype=torch.bfloat16, device=logits_bf16.device)
# Use the BF16 backward kernel (from the CUDA side)
# Actually, just compute manually in float32
import triton.language as tl

# Manual BF16 CE backward computation
logits_f = logits_bf16.float()
for_row = logits_f
inv_C = 1.0 / C
B_div_C = B_val * inv_C
inv_C_A = inv_C * A
u = for_row * inv_C + B_div_C
sigmoid_u = torch.sigmoid(u)
z = A * sigmoid_u
p = torch.exp(z - lse_bf16.unsqueeze(-1))

S_w_per_row = torch.zeros(T, device=logits_bf16.device)
for k in range(n_predict):
    mask = torch.arange(T, device=logits_bf16.device) + k < T
    S_w_per_row += mask.float() * mtp_weights[k]

term1 = S_w_per_row.unsqueeze(-1) * p
term2 = torch.zeros_like(logits_f)
for k in range(n_predict):
    valid = torch.arange(T, device=logits_bf16.device) + k < T
    tgt = targets[torch.arange(T, device=logits_bf16.device) + k].clamp(0, 50303)
    weight = mtp_weights[k]
    term2.scatter_add_(1, tgt.unsqueeze(-1), (valid.float() * weight).unsqueeze(-1).expand(-1, 1))

grad_z = grad_output_bf16.unsqueeze(-1) * (term1 - term2)
dz_dx = inv_C_A * sigmoid_u * (1.0 - sigmoid_u)
grad_logits_bf16_manual = grad_z * dz_dx  # true gradient (no grad_s division)

print(f"CE grad_logits (BF16 logits, true scale): {grad_logits_bf16_manual.norm().item():.6f}", flush=True)

# Compare
print(f"Ratio (BF16/FP8): {grad_logits_bf16_manual.norm().item() / grad_logits_fp8_true.norm().item():.4f}", flush=True)

# Also compute grad_x for both
w = model.lm_head.weight.detach()
# FP8 path: grad_x = _scaled_mm(grad_logits_fp8, w_f8.T, scale_a=grad_s, scale_b=w_s)
grad_x_fp8 = torch._scaled_mm(
    grad_logits_fp8, w_f8.T,
    out_dtype=torch.bfloat16,
    scale_a=torch.tensor(grad_s, dtype=torch.float32, device=logits_fp8.device),
    scale_b=torch.tensor(model.lm_head.w_s, dtype=torch.float32, device=logits_fp8.device),
    use_fast_accum=False,
)
# BF16 path: grad_x = grad_logits @ w.T
grad_x_bf16 = grad_logits_bf16_manual.bfloat16() @ w.T

print(f"grad_x (FP8 path): {grad_x_fp8.float().norm().item():.6f}", flush=True)
print(f"grad_x (BF16 path): {grad_x_bf16.float().norm().item():.6f}", flush=True)
print(f"Ratio grad_x (BF16/FP8): {grad_x_bf16.float().norm().item() / grad_x_fp8.float().norm().item():.4f}", flush=True)

# Loss comparison
print(f"Loss (FP8 logits): {losses_fp8.sum().item():.6f}", flush=True)
print(f"Loss (BF16 logits): {losses_bf16.sum().item():.6f}", flush=True)

print("=== End CE comparison ===", flush=True)
dist.destroy_process_group()
