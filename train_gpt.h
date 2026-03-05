// train_gpt.h - Shared declarations for pure CUDA C GPT training
// Port of modded-nanogpt-vk (train_gpt.py + triton_kernels.py)
#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <curand.h>
#include <cudnn.h>
#include <cudnn_frontend.h>
#include <stdint.h>
#include <unordered_map>
#include <memory>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>

// ============================================================================
// Constants
// ============================================================================

#define VOCAB_SIZE_RAW  50257
#define VOCAB_SIZE      50304   // next_multiple_of_n(50257, n=128)
#define NUM_LAYERS      11
#define NUM_HEADS       6
#define HEAD_DIM        128
#define MODEL_DIM       768
#define MLP_HDIM        3072    // 4 * MODEL_DIM
#define NUM_ATTN_LAYERS 10      // Layer 6 has no attention
#define NUM_MLP_LAYERS  11
#define MLP_BANK_SIZE   12      // 11 layers + 1 padding for even 8-GPU distribution
#define BIGRAM_VOCAB_SIZE (VOCAB_SIZE * 5)  // 50304 * 5 = 251520

#define BOS_ID          50256
#define GRAD_ACCUM_STEPS 8      // Single GPU: always 8

// FP8 scales for lm_head
#define LM_HEAD_X_S     (100.0f / 448.0f)
#define LM_HEAD_W_S     (1.6f / 448.0f)
#define LM_HEAD_GRAD_S  ((1.0f / 8.0f) * 0.75f / 448.0f)  // grad_scale * 0.75/448

// Softcap constants
#define SOFTCAP_A       23.0f
#define SOFTCAP_B       5.0f
#define SOFTCAP_C       7.5f

// Optimizer defaults
#define ADAM_LR         0.008f
#define ADAM_EPS        1e-10f
#define ADAM_WD         0.005f
#define MUON_LR        0.023f
#define MUON_MOMENTUM   0.95f
#define MUON_BETA2      0.95f
#define MUON_WD         1.2f

// Polar Express coefficients (5 iterations)
#define POLAR_NUM_ITERS 5
static const float POLAR_COEFFS[POLAR_NUM_ITERS][3] = {
    {8.156554524902461f, -22.48329292557795f, 15.878769915207462f},
    {4.042929935166739f, -2.808917465908714f, 0.5000178451051316f},
    {3.8916678022926607f, -2.772484153217685f, 0.5060648178503393f},
    {3.285753657755655f, -2.3681294933425376f, 0.46449024233003106f},
    {2.3465413258596377f, -1.7097828382687081f, 0.42323551169305323f},
};

// Schedule constants
#define NUM_SCHEDULED_ITERS  1450
#define NUM_EXTENSION_ITERS  40
#define TOTAL_STEPS          (NUM_SCHEDULED_ITERS + NUM_EXTENSION_ITERS)
#define COOLDOWN_FRAC        0.60f
#define VAL_TOKENS           10485760
#define VAL_BATCH_SIZE       (4 * 64 * 1024 * 8)  // 2097152
#define VAL_LOSS_EVERY       250
#define BLOCK_SIZE_SCHED     128   // block_size for window size units

// Bigram hash constants
#define BIGRAM_RAND_INT_1    36313
#define BIGRAM_RAND_INT_2    27191

// Parallel layers
#define PARALLEL_START       7
#define BACKOUT_LAYER        7
#define SKIP_IN_LAYER        3
#define SKIP_OUT_LAYER       6
#define ATTN_SKIP_LAYER      6

// ============================================================================
// CUDA error checking
// ============================================================================

#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                   \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

#define CUBLAS_CHECK(call)                                                     \
    do {                                                                        \
        cublasStatus_t err = (call);                                           \
        if (err != CUBLAS_STATUS_SUCCESS) {                                    \
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__,          \
                    __LINE__, (int)err);                                        \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

#define CUDNN_CHECK(call)                                                      \
    do {                                                                        \
        cudnnStatus_t err = (call);                                            \
        if (err != CUDNN_STATUS_SUCCESS) {                                     \
            fprintf(stderr, "cuDNN error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudnnGetErrorString(err));                                  \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

#define CURAND_CHECK(call)                                                     \
    do {                                                                        \
        curandStatus_t err = (call);                                           \
        if (err != CURAND_STATUS_SUCCESS) {                                    \
            fprintf(stderr, "cuRAND error at %s:%d: %d\n", __FILE__,          \
                    __LINE__, (int)err);                                        \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

// ============================================================================
// Type aliases
// ============================================================================

typedef __nv_bfloat16 bf16;
typedef __nv_fp8_e4m3 fp8e4m3;
typedef __nv_fp8_e5m2 fp8e5m2;

// ============================================================================
// Structs
// ============================================================================

// Model parameters - all GPU pointers
typedef struct {
    // Embeddings
    bf16* embed;                    // [VOCAB_SIZE, MODEL_DIM]
    bf16* bigram_embed;             // [BIGRAM_VOCAB_SIZE, MODEL_DIM]
    bf16* value_embeds;             // [5 * VOCAB_SIZE, MODEL_DIM]

    // Transformer banks
    bf16* attn_bank;                // [NUM_ATTN_LAYERS, 4*MODEL_DIM, MODEL_DIM] = [10, 3072, 768]
    bf16* mlp_bank;                 // [MLP_BANK_SIZE, 2, MLP_HDIM, MODEL_DIM] = [12, 2, 3072, 768]

    // lm_head (transposed layout: in_features, out_features)
    bf16* lm_head;                  // [MODEL_DIM, VOCAB_SIZE]

    // Gates and scalars
    bf16* attn_gate_bank;           // [10, NUM_HEADS, 12]
    bf16* ve_gate_bank;             // [5, NUM_HEADS, 12]
    bf16* smear_gate;               // [1, 12] (Linear weight, no bias)
    bf16* skip_gate;                // [1, 12]

    // Lambda/scaling parameters (bf16 storage, but initialized as float)
    bf16* post_lambdas;             // [NUM_LAYERS, 2, 2]
    bf16* x0_lambdas;              // [NUM_LAYERS]
    bf16* bigram_lambdas;          // [NUM_LAYERS]
    bf16* resid_lambdas;           // [NUM_LAYERS, 2]

    // Scalars (compact 1D tensor)
    bf16* scalars;                  // [2*NUM_LAYERS + 3 + pad] = [25+pad]
    int scalars_size;               // actual size including padding
} ModelParams;

// Gradients - mirrors ModelParams but some in float32
typedef struct {
    bf16* embed;
    bf16* bigram_embed;
    bf16* value_embeds;
    bf16* attn_bank;
    bf16* mlp_bank;
    bf16* lm_head;                  // BF16 for lm_head grad (using BF16 backward)
    bf16* attn_gate_bank;
    bf16* ve_gate_bank;
    bf16* smear_gate;
    bf16* skip_gate;
    bf16* post_lambdas;
    bf16* x0_lambdas;
    bf16* bigram_lambdas;
    bf16* resid_lambdas;
    bf16* scalars;
} ModelGrads;

// Per-parameter Adam state
typedef struct {
    float* exp_avg;
    float* exp_avg_sq;
    int step;
    int numel;  // number of elements
} AdamState;

// Per-parameter NorMuon state
typedef struct {
    float* momentum_buffer;         // FP32 momentum
    float* second_momentum_buffer;  // reduced-dim second moment
    uint16_t* mantissa;             // precision tracking
    int chunk_shape[4];             // reshaped dimensions
    int chunk_ndim;
    int numel;
} NorMuonState;

// Complete optimizer state
typedef struct {
    // Adam states
    AdamState adam_scalars;
    AdamState adam_smear_gate;
    AdamState adam_skip_gate;
    AdamState adam_attn_gate_bank;
    AdamState adam_ve_gate_bank;
    AdamState adam_lm_head;
    AdamState adam_embed;
    AdamState adam_bigram_embed;
    AdamState adam_value_embeds;
    AdamState adam_post_lambdas;
    AdamState adam_x0_lambdas;
    AdamState adam_bigram_lambdas;
    AdamState adam_resid_lambdas;

    // NorMuon states
    NorMuonState muon_attn_bank;
    NorMuonState muon_mlp_bank;

    // Embed/lm_head tying
    int split_embed;  // 0 = tied, 1 = split
    int split_step;

    // Polar Express scratch (pre-allocated)
    float* polar_norms;     // [max_batch]
    bf16* polar_A;          // [max_batch, small_dim, small_dim]
    bf16* polar_B;          // [max_batch, small_dim, small_dim]
    bf16* polar_C;          // [max_batch, rows, cols]
} OptimizerState;

// Activation scratch buffers
typedef struct {
    // Per-layer intermediates
    bf16* x;                    // current hidden state [1, T, MODEL_DIM]
    bf16* x0;                   // initial normed embedding [1, T, MODEL_DIM]
    bf16* x0_bigram;            // bigram embedding [1, T, MODEL_DIM]
    bf16* lane0;                // residual lane 0 [1, T, MODEL_DIM]
    bf16* lane1;                // residual lane 1 [1, T, MODEL_DIM]

    // Attention intermediates
    bf16* qkv;                  // [1, T, 3*NUM_HEADS, HEAD_DIM]
    bf16* q;                    // [T, NUM_HEADS, HEAD_DIM]
    bf16* k;                    // [T, NUM_HEADS, HEAD_DIM]
    bf16* v;                    // [T, NUM_HEADS, HEAD_DIM]
    bf16* attn_out;             // [T, NUM_HEADS, HEAD_DIM]
    bf16* normed;               // [1, T, MODEL_DIM] - RMS norm output

    // MLP intermediates
    bf16* mlp_pre;              // pre-activation [T, MLP_HDIM] (from linear_relu_square)
    bf16* mlp_post;             // post-activation [T, MLP_HDIM] (relu²)
    bf16* mlp_out;              // after W2 [T, MODEL_DIM]

    // FP8 intermediates for lm_head
    fp8e4m3* x_f8;             // [T, MODEL_DIM]
    fp8e4m3* w_f8;             // [MODEL_DIM, VOCAB_SIZE]
    bf16* logits;               // [T, VOCAB_SIZE]

    // Loss computation
    float* losses;              // [T]
    float* lse;                 // [T] log-sum-exp
    fp8e5m2* grad_logits;       // [T, VOCAB_SIZE] backward gradient in FP8

    // Skip connection storage
    bf16* skip_save;            // [1, T, MODEL_DIM]
    bf16* x_backout;            // [1, T, MODEL_DIM]

    // Bulk-allocated per-layer saved activations (single cudaMalloc, pointer arithmetic)
    bf16* saved_bulk;           // single large allocation for all per-layer buffers

    // Per-layer pointers (set up by alloc_activations, point into saved_bulk)
    bf16* saved_normed[NUM_LAYERS];     // normed input for attention (each layer)
    bf16* saved_normed_mlp[NUM_LAYERS]; // normed input for MLP (recomputed or saved)
    bf16* saved_post_attn[NUM_LAYERS];  // post-attention residual
    bf16* saved_lane0[NUM_LAYERS];      // lane0 before each layer
    bf16* saved_lane1[NUM_LAYERS];      // lane1 before each layer (parallel layers)
    bf16* saved_lane1_post_attn[NUM_LAYERS]; // lane1 after attn update (parallel layers, for MLP backward)
    bf16* saved_mlp_pre[NUM_LAYERS];    // MLP pre-activation for backward
    bf16* saved_mlp_post[NUM_LAYERS];   // MLP post-activation (relu²) for backward
    bf16* saved_mlp_out[NUM_LAYERS];    // MLP output for scalar gradient computation
    bf16* saved_attn_proj[NUM_LAYERS];  // Attention projection output (for parallel layer gradients)

    // Saved attention intermediates for backward (per attention layer)
    bf16* saved_qkv[NUM_ATTN_LAYERS];     // [T, 3*H, HD] QKV after RMS norm + RoPE
    bf16* saved_attn_out[NUM_ATTN_LAYERS]; // [T, H, HD] output of attention before gating
    bf16* saved_attn_gate[NUM_ATTN_LAYERS];// [T, H] attention output gate values
    bf16* saved_ve_gate[5];               // [T, H] or [T, H/2] VE gate values (5 VE layers)

    // Gradient scratch
    bf16* grad_x;               // gradient flowing backward [1, T, MODEL_DIM]
    bf16* grad_lane0;           // gradient for lane0
    bf16* grad_lane1;           // gradient for lane1
    bf16* grad_x0;              // accumulated gradient for x0 embedding
    bf16* grad_x0_bigram;       // accumulated gradient for bigram embedding

    // cuDNN flash attention stats (log-sum-exp from forward, needed by backward)
    // Shape: [num_seqs, attn_H, s_max, 1] per attention layer
    float* attn_stats[NUM_ATTN_LAYERS];

    // Per-layer saved info for backward (set during forward)
    int saved_is_paired[NUM_ATTN_LAYERS];
    int saved_num_seqs;

    // cuDNN ragged offset scratch buffers
    int32_t* cu_seqlens_paired;   // ragged offsets (element offsets, not token counts)
    int32_t* seq_len_scratch;     // per-sequence lengths for padding mask [num_seqs]

    // cuDNN attention workspace
    void* attn_workspace;
    size_t attn_workspace_size;

    // cuBLAS workspace
    void* cublas_workspace;
    size_t cublas_workspace_size;

    // YaRN cos/sin tables
    bf16* yarn_cos;             // [2*max_seq_len, HEAD_DIM]
    bf16* yarn_sin;             // [2*max_seq_len, HEAD_DIM]
    bf16* yarn_paired_cos;      // [2*max_seq_len, 2*HEAD_DIM] for paired heads
    bf16* yarn_paired_sin;

    // General-purpose scratch
    bf16* scratch1;
    bf16* scratch2;
    float* scratch_f32;
} Activations;

// Training stage definition
typedef struct {
    float lr_mul;
    int batch_size;
    int ws_short;               // window size (short) in block units
    int ws_long;                // window size (long) in block units
    int train_max_seq_len;
    float mtp_weights_start[3]; // up to 3 MTP heads
    float mtp_weights_end[3];
    int n_mtp_weights;          // how many MTP weights active
    float duration;             // fraction of scheduled iterations
} TrainingStageConfig;

// cuDNN Frontend graph cache for flash attention
namespace fe = cudnn_frontend;

struct CudnnGraphCache {
    struct Key {
        int num_seqs, attn_H, s_max, window_left;
        float attn_scale;
        bool is_backward;
        bool operator==(const Key& o) const {
            return num_seqs == o.num_seqs && attn_H == o.attn_H &&
                   s_max == o.s_max && window_left == o.window_left &&
                   attn_scale == o.attn_scale && is_backward == o.is_backward;
        }
    };
    struct KeyHash {
        size_t operator()(const Key& k) const {
            size_t h = 0;
            h ^= std::hash<int>()(k.num_seqs) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<int>()(k.attn_H) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<int>()(k.s_max) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<int>()(k.window_left) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<float>()(k.attn_scale) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<bool>()(k.is_backward) + 0x9e3779b9 + (h << 6) + (h >> 2);
            return h;
        }
    };
    std::unordered_map<Key, std::shared_ptr<fe::graph::Graph>, KeyHash> cache;
};

// Tensor UIDs for cuDNN variant pack
enum CudnnTensorUID {
    UID_Q = 1, UID_K, UID_V, UID_O, UID_DO, UID_STATS,
    UID_DQ, UID_DK, UID_DV, UID_SEQ_Q, UID_SEQ_KV,
    UID_SEQ_LEN_Q, UID_SEQ_LEN_KV
};

// Cached host-side copies of scalar parameters (constant across microsteps)
struct CachedScalars {
    bf16 scalars[2 * NUM_LAYERS + 3];
    bf16 resid_lambdas[NUM_LAYERS * 2];
    bf16 post_lambdas[NUM_LAYERS * 4];
    bf16 x0_lambdas[NUM_LAYERS];
    bf16 bigram_lambdas[NUM_LAYERS];
};

// Overall training state
struct TrainingContext {
    // Handles
    cublasHandle_t cublas_handle;
    cublasLtHandle_t cublaslt_handle;
    cudnnHandle_t cudnn_handle;
    curandGenerator_t curand_gen;
    cudaStream_t stream;

    // Model
    ModelParams params;
    ModelGrads grads;
    OptimizerState opt;
    Activations act;

    // Schedule state
    int current_step;
    int ws_short;
    int ws_long;
    int batch_size;
    int train_max_seq_len;
    float* mtp_weights;         // [TOTAL_STEPS+1][3] precomputed

    // Data
    uint16_t* train_tokens;     // mmap'd training data
    uint16_t* val_tokens;       // mmap'd validation data
    int64_t train_num_tokens;
    int64_t val_num_tokens;

    // Timing
    double training_time_ms;

    // cuDNN flash attention graph cache
    CudnnGraphCache cudnn_graph_cache;
};

// ============================================================================
// Forward declarations (implemented in train_gpt.cu)
// ============================================================================

void init_training_context(TrainingContext* ctx);
void free_training_context(TrainingContext* ctx);
void init_model_params(TrainingContext* ctx);
void forward_pass(TrainingContext* ctx, int32_t* inputs, int64_t* targets,
                  int32_t* cum_seqlens, int32_t* bigram_inputs,
                  int num_tokens, float* mtp_weights, int n_mtp,
                  int ws_short, int ws_long, int train_max_seq_len,
                  int is_training, float* loss_out);
void backward_pass(TrainingContext* ctx, int32_t* inputs, int64_t* targets,
                   int32_t* cum_seqlens, int32_t* bigram_inputs,
                   int num_tokens, float* mtp_weights, int n_mtp,
                   int ws_short, int ws_long, int train_max_seq_len);
void optimizer_step(TrainingContext* ctx, int step, int do_adam);

// ============================================================================
// Utility inline functions
// ============================================================================

static inline int cdiv(int a, int b) { return (a + b - 1) / b; }
static inline int next_multiple_of(int v, int n) {
    return ((v + n - 1) / n) * n;
}

// BF16 conversion helpers
static inline float bf16_to_float(__nv_bfloat16 x) {
    return __bfloat162float(x);
}
static inline __nv_bfloat16 float_to_bf16(float x) {
    return __float2bfloat16(x);
}
