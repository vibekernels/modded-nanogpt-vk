// kernels.h - Custom CUDA kernel launch wrappers
// Ports of triton_kernels.py: XXT, XTX, ba+cAA, fused ReLU², softcapped CE, transpose ops
// Plus additional kernels: RMS norm, RoPE, bigram hash, embedding, optimizer updates
#pragma once

#include "train_gpt.h"

// ============================================================================
// RMS Norm
// ============================================================================

// Forward: out[i] = x[i] / sqrt(mean(x[i]^2) + eps)
void rms_norm_fwd(bf16* out, const bf16* x, int rows, int cols, cudaStream_t stream);

// Backward: computes grad_x given grad_out and original x
void rms_norm_bwd(bf16* grad_x, const bf16* grad_out, const bf16* x, int rows, int cols, cudaStream_t stream);

// ============================================================================
// Symmetric matrix multiplication kernels (from triton_kernels.py)
// ============================================================================

// C = A @ A.T where A is [batch, M, K] and C is [batch, M, M]
// Exploits symmetry to compute ~half the blocks
void xxt_kernel_launch(const bf16* A, bf16* C,
                       int batch_size, int M, int K,
                       int a_stride_b, int a_stride_r, int a_stride_c,
                       int c_stride_b, int c_stride_r, int c_stride_c,
                       cudaStream_t stream);

// C = A.T @ A where A is [batch, M, K] and C is [batch, K, K]
void xtx_kernel_launch(const bf16* A, bf16* C,
                       int batch_size, int M, int K,
                       int a_stride_b, int a_stride_r, int a_stride_c,
                       int c_stride_b, int c_stride_r, int c_stride_c,
                       cudaStream_t stream);

// C = alpha * A @ A.T + beta * A where A is square [batch, M, M]
void ba_plus_caa_kernel_launch(const bf16* A, bf16* C,
                               int batch_size, int M,
                               int a_stride_b, int a_stride_r, int a_stride_c,
                               int c_stride_b, int c_stride_r, int c_stride_c,
                               float alpha, float beta,
                               cudaStream_t stream);

// ============================================================================
// Fused MLP: relu(x @ W1.T)^2
// ============================================================================

// Forward: computes pre = x @ W1.T, post = relu(pre)^2
// x: [M, K], W1: [N, K] -> pre: [M, N], post: [M, N]
// NOTE: Prefer using cuBLAS for matmul + relu_square_fwd() for activation
void linear_relu_square_fwd(const bf16* x, const bf16* W1, bf16* pre, bf16* post,
                            int M, int N, int K, cudaStream_t stream);

// Pure activation: post[i] = max(pre[i], 0)^2
void relu_square_fwd(const bf16* pre, bf16* post, int n, cudaStream_t stream);

// Backward: computes grad_x_W1 = 2 * grad_out * relu_mask(pre) * pre
// (the gradient through relu(x)^2 activation, before the W1 matmul)
// grad_out: [M, N], pre: [M, N] -> out: [M, N]
void linear_relu_square_bwd(const bf16* grad_out, const bf16* pre, bf16* out,
                            int M, int N, cudaStream_t stream);

// ============================================================================
// Softcapped Cross Entropy Loss
// ============================================================================

// Forward: logits = x @ lm_head.T (done via cuBLAS FP8 externally)
//          losses[i] = sum_k(mtp_weights[k] * (lse[i] - z_target[i+k]))
//          where z = A * sigmoid((logit + B) / C)
void softcapped_ce_fwd(const bf16* logits, float* losses, float* lse,
                       const int64_t* targets, const float* mtp_weights,
                       int n_rows, int n_cols, int n_predict,
                       float A, float B, float C,
                       cudaStream_t stream);

// Backward: computes FP8 e5m2 gradients
void softcapped_ce_bwd(fp8e5m2* grad_input, const float* grad_output, const float* lse,
                       const bf16* logits, const int64_t* targets, const float* mtp_weights,
                       int n_rows, int n_cols, int n_predict,
                       float A, float B, float C, float grad_s,
                       cudaStream_t stream);

// Backward: computes BF16 gradients (no FP8)
void softcapped_ce_bwd_bf16(bf16* grad_input, const float* grad_output, const float* lse,
                             const bf16* logits, const int64_t* targets, const float* mtp_weights,
                             int n_rows, int n_cols, int n_predict,
                             float A, float B, float C, float grad_s,
                             cudaStream_t stream);

// ============================================================================
// Transpose operations
// ============================================================================

// dst[N, M] = src[M, N].T  (tiled 32x32 coalesced transpose)
void transpose_copy(const bf16* src, bf16* dst, int M, int N, cudaStream_t stream);

// dst[N, M] += src[M, N].T  (tiled 32x32 coalesced transpose-add)
void transpose_add(const bf16* src, bf16* dst, int M, int N, cudaStream_t stream);

// ============================================================================
// RoPE (Rotary Position Embeddings)
// ============================================================================

// Apply rotary embedding: x_out = cos * x + sin * flip(x)
// x: [T, num_heads, head_dim], cos/sin: [T, head_dim]
void rope_apply(bf16* x, const bf16* cos_table, const bf16* sin_table,
                int T, int num_heads, int head_dim, cudaStream_t stream);

// Compute cos/sin tables for YaRN
void yarn_compute_tables(bf16* cos_out, bf16* sin_out,
                         const float* angular_freq, int head_dim,
                         int max_seq_len, int paired, cudaStream_t stream);

// ============================================================================
// Embedding and data processing kernels
// ============================================================================

// Embedding lookup: out[i] = embed[indices[i]]
void gather_embed(bf16* out, const bf16* embed, const int32_t* indices,
                  int num_tokens, int embed_dim, cudaStream_t stream);

// Scatter-add for embedding gradient: embed_grad[indices[i]] += grad[i]
void scatter_add_embed(bf16* embed_grad, const bf16* grad, const int32_t* indices,
                       int num_tokens, int embed_dim, cudaStream_t stream);

// Bigram hash: out[0] = BIGRAM_VOCAB_SIZE-1, out[i] = XOR(rand1*x[i], rand2*x[i-1]) % (BIGRAM_VOCAB_SIZE-1)
void bigram_hash(int32_t* out, const int32_t* input, int num_tokens, cudaStream_t stream);

// ============================================================================
// Gating kernels
// ============================================================================

// Sigmoid gate: out = scale * sigmoid(linear(x_slice, gate_w))
// x_slice: first `in_dim` features of x, gate_w: [out_dim, in_dim]
void sigmoid_gate(bf16* out, const bf16* x, const bf16* gate_w,
                  int num_tokens, int in_dim, int out_dim, float scale,
                  cudaStream_t stream);

// ============================================================================
// Elementwise kernels
// ============================================================================

// x_out = a * x + y (bf16)
void fused_add_scale(bf16* out, const bf16* x, const bf16* y,
                     float scale_x, float scale_y, int n, cudaStream_t stream);

// out = a * x + b * y + c * z (3-way fused add)
void fused_add3(bf16* out, const bf16* x, const bf16* y, const bf16* z,
                float a, float b, float c, int n, cudaStream_t stream);

// Smear: out[0] = x[0], out[i] = x[i] + gate * x[i-1] for i > 0
void smear_forward(bf16* out, const bf16* x, const bf16* smear_gate_w,
                   float smear_lambda, int num_tokens, int model_dim,
                   cudaStream_t stream);

// ============================================================================
// Optimizer kernels
// ============================================================================

// Fused Adam step with cautious weight decay
void adam_update(bf16* param, const bf16* grad,
                float* exp_avg, float* exp_avg_sq,
                float beta1, float beta2, float eps,
                float step_size, float eff_wd,
                int numel, cudaStream_t stream);

// Fused NorMuon update: cautious WD + mantissa-tracked BF16 update
void muon_cautious_update(uint16_t* param_u16, uint16_t* mantissa,
                          const bf16* grad, float eff_wd, float eff_lr,
                          int numel, cudaStream_t stream);

// NorMuon variance reduction
void normuon_variance_reduction(bf16* v_chunk, float* second_momentum_buffer,
                                float beta2, int red_dim,
                                int outer_size, int dim0, int dim1,
                                cudaStream_t stream);

// Scale tensor by scalar: x *= scale
void scale_tensor(bf16* x, float scale, int n, cudaStream_t stream);

// Compute tensor norm (for Polar Express spectral norm scaling)
void tensor_norm(const bf16* x, float* out_norm, int batch, int rows, int cols,
                 cudaStream_t stream);

// ============================================================================
// FP8 quantization
// ============================================================================

// Convert FP32 to BF16: out[i] = bf16(x[i])
void f32_to_bf16(bf16* out, const float* x, int n, cudaStream_t stream);

// Convert BF16 to FP8 e4m3 with scaling: out[i] = fp8(x[i] * scale)
void bf16_to_fp8_e4m3(fp8e4m3* out, const bf16* x, float scale, int n, cudaStream_t stream);

// Convert BF16 to FP8 e5m2 with scaling: out[i] = fp8(x[i] * scale)
void bf16_to_fp8_e5m2(fp8e5m2* out, const bf16* x, float scale, int n, cudaStream_t stream);

// ============================================================================
// Nesterov momentum (for Polar Express)
// ============================================================================

// Fused Nesterov: buf = momentum*buf + (1-momentum)*grad_fp32
//                 grad_out = momentum*buf + (1-momentum)*grad_fp32  (= Nesterov lookahead)
// Then casts to BF16
void nesterov_momentum(bf16* grad_out, float* momentum_buffer, const bf16* grad_in,
                       float momentum, int n, cudaStream_t stream);

// Normalize each batch element: x[b] /= (norm[b] * (1+safety) + eps)
void norm_divide(bf16* x, const float* norms, int batch, int elems_per_batch,
                 float safety, float eps, cudaStream_t stream);

// ============================================================================
// GPU reduction
// ============================================================================

// Sum all elements of a float array, result stored at out[0]
void gpu_reduce_sum(float* out, const float* x, int n, cudaStream_t stream);

// Fill float array with a constant value
void fill_constant_f32(float* out, float val, int n, cudaStream_t stream);

// Dot product of two bf16 tensors, result added to a float output: *out += sum(a[i]*b[i])
// Uses block reduction; out must be pre-initialized (e.g., to 0)
void bf16_dot_product(float* out, const bf16* a, const bf16* b, int n, cudaStream_t stream);

// ============================================================================
// Elementwise multiply (gating)
// ============================================================================

// out[i] = x[i] * gate[i % gate_stride], broadcasting gate across tokens
// Used for per-head gating: x [T, H, HD], gate [H] -> out [T, H, HD]
void elementwise_mul_broadcast(bf16* out, const bf16* x, const bf16* gate,
                               int total, int inner_dim, int gate_stride,
                               cudaStream_t stream);

// out[i] = x[i] + gate[i % gate_stride] * y[i], fused gate-add
void fused_gate_add(bf16* out, const bf16* x, const bf16* gate, const bf16* y,
                    int total, int inner_dim, int gate_stride, cudaStream_t stream);

// ============================================================================
// Naive varlen flash attention (placeholder for cuDNN)
// ============================================================================

// Naive multi-head attention with variable-length sequences, causal mask, sliding window
// Q, K, V: [T, H, HD], out: [T, H, HD]
// cu_seqlens: [num_seqs + 1] cumulative sequence lengths
// window_left: left window size in tokens (0 = full causal, no window)
void naive_varlen_attention(bf16* out,
                            const bf16* Q, const bf16* K, const bf16* V,
                            const int32_t* cu_seqlens, int num_seqs,
                            int total_T, int H, int HD,
                            float scale, int window_left,
                            cudaStream_t stream);

// Naive attention backward: computes dQ, dK, dV from d_out
// Q, K, V, out: [T, H, HD], d_out: [T, H, HD], dQ, dK, dV: [T, H, HD]
// Recomputes attention weights internally (no saved softmax)
// dQ_f32, dK_f32, dV_f32: pre-allocated float scratch buffers [T, H, HD] each
void naive_varlen_attention_backward(
    bf16* dQ, bf16* dK, bf16* dV,
    float* dQ_f32, float* dK_f32, float* dV_f32,
    const bf16* d_out, const bf16* Q, const bf16* K, const bf16* V,
    const int32_t* cu_seqlens, int num_seqs,
    int total_T, int H, int HD,
    float scale, int window_left,
    cudaStream_t stream);

// Key offset: shift second half of key dimensions forward by 1 position per sequence
// k: [T, H, HD], shifts k[t, :, HD/2:] = k[t-1, :, HD/2:] for t > seq_start
void key_offset_shift(bf16* k, const int32_t* cu_seqlens, int num_seqs,
                      int total_T, int H, int HD, cudaStream_t stream);

// ============================================================================
// Scalar gradient accumulation (GPU-side)
// ============================================================================

// Accumulate float scalar gradient accumulators into bf16 gradient buffers on GPU
// acc layout: [0..10] resid_attn, [11..21] resid_mlp, [22..32] x0_lambda,
//   [33..43] bigram_lambda, [44..87] post_lambdas, [88] backout_lambda
// negate_backout: if true, subtract acc[88] instead of add (forward was subtraction)
void accumulate_scalar_grads(bf16* resid_grads, bf16* x0_grads, bf16* bigram_grads,
                             bf16* post_lambda_grads, bf16* scalar_grads,
                             const float* acc, int num_layers, cudaStream_t stream);
