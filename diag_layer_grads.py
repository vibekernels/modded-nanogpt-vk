"""
Diagnostic: Print per-layer backward gradient norms to compare with CUDA.
Run with: torchrun --standalone --nproc_per_node=1 diag_layer_grads.py
"""
import torch
import torch.distributed as dist
import os, sys

os.environ.setdefault("RANK", "0")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")

sys.path.insert(0, os.path.dirname(__file__))
exec(open("train_gpt.py").read().split("# start the clock")[0])

print("=== Running 1 microstep for per-layer gradient diagnostics ===", flush=True)

train_loader2 = distributed_data_generator(
    args.train_files, TRAINING_STAGES[0].batch_size,
    TRAINING_STAGES[0].train_max_seq_len, grad_accum_steps=grad_accum_steps
)

model.train()
model.zero_grad(set_to_none=True)

# Hook storage
layer_grad_norms = {}

def make_hook(name):
    def hook(grad):
        layer_grad_norms[name] = grad.float().norm().item()
    return hook

# We need to intercept the forward to add hooks
original_forward = model.forward

def instrumented_forward(input_seq, target_seq, seqlens, bigram_input_seq, schedule_cfg):
    assert input_seq.ndim == 1
    mtp_weights, train_max_seq_len = schedule_cfg.mtp_weights, schedule_cfg.train_max_seq_len
    ws_short, ws_long = schedule_cfg.ws_short, schedule_cfg.ws_long
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

    x = model.embed(input_seq)
    x0_bigram = model.bigram_embed(bigram_input_seq)[None]
    ve = model.value_embeds.view(5, model.vocab_size, -1)[:, input_seq]
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
            ve=ve[i], sa_lambdas=sa_lambdas[i], seqlens=seqlens,
            bm_size=bm_sizes[i], yarn=yarn, key_offset=key_offset[i],
            attn_gate_w=attn_gates[i], ve_gate_w=ve_gates[i],
            train_max_seq_len=train_max_seq_len
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

        # Register hook at end of each layer to capture grad_lane0
        if lane0.requires_grad:
            lane0.register_hook(make_hook(f"layer_{i}_lane0"))

        if i in skip_in:
            skip_connections.append(post_attn)
        if i == backout_layer:
            x_backout = lane0

    x = (lane0 + lane1) * 0.5
    x -= backout_lambda * x_backout
    x_pre_norm = x
    x = norm(x)

    # Hook on x (post-norm, pre-lm_head)
    x.register_hook(make_hook("grad_x_post_norm"))

    losses = FusedSoftcappedCrossEntropy.apply(x.view(-1, x.size(-1)), target_seq, mtp_weights, model.lm_head.weight, model.lm_head.x_s, model.lm_head.w_s, model.lm_head.grad_s)
    loss = losses.sum()
    return loss

# Run one microstep
send_args = training_manager.train_loader_send_args
inputs, targets, cum_seqlens, bigram_inputs, bigram_cpu = train_loader2.send(send_args)
training_manager.sparse_index_update(0, bigram_cpu)

loss = instrumented_forward(inputs, targets, cum_seqlens, bigram_inputs, training_manager.get_forward_args()) * grad_scale
loss.backward()

print("=== Per-layer gradient norms (first microstep) ===", flush=True)

# The hooks fire during backward, so the norms should be captured now
# Note: hook on layer_i captures grad flowing INTO layer_i (the grad of lane0 AFTER layer i's update)
# This matches the CUDA diagnostic which prints grad_lane0 AFTER each layer backward
for i in range(model.num_layers - 1, -1, -1):
    key = f"layer_{i}_lane0"
    if key in layer_grad_norms:
        print(f"  DIAG layer {i} grad_lane0={layer_grad_norms[key]:.4f}")
    else:
        print(f"  DIAG layer {i} grad_lane0=NOT_CAPTURED")

if "grad_x_post_norm" in layer_grad_norms:
    print(f"  DIAG grad_x_post_norm={layer_grad_norms['grad_x_post_norm']:.4f}")

print("=== End diagnostics ===", flush=True)
dist.destroy_process_group()
