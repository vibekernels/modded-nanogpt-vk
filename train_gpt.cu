// train_gpt.cu - Pure CUDA C GPT training implementation
// Port of modded-nanogpt-vk (train_gpt.py)
// Single GPU (1xH100), no distributed training
#include "train_gpt.h"
#include "kernels.h"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <curand.h>
#include <cudnn.h>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <glob.h>
#include <time.h>
#include <pthread.h>
#include <string.h>
#include <math.h>
#include <malloc.h>

// ============================================================================
// Section A: Infrastructure
// ============================================================================

// ---- Memory Arena ----
typedef struct {
    void* base;
    size_t size;
    size_t offset;
} MemoryArena;

static MemoryArena g_arena;

void arena_init(size_t total_bytes) {
    CUDA_CHECK(cudaMalloc(&g_arena.base, total_bytes));
    g_arena.size = total_bytes;
    g_arena.offset = 0;
    CUDA_CHECK(cudaMemset(g_arena.base, 0, total_bytes));
}

void* arena_alloc(size_t bytes) {
    // Align to 256 bytes
    size_t aligned = (bytes + 255) & ~255;
    if (g_arena.offset + aligned > g_arena.size) {
        fprintf(stderr, "Arena OOM: requested %zu, used %zu/%zu\n", aligned, g_arena.offset, g_arena.size);
        exit(1);
    }
    void* ptr = (char*)g_arena.base + g_arena.offset;
    g_arena.offset += aligned;
    return ptr;
}

void arena_reset() {
    g_arena.offset = 0;
}

void arena_free() {
    if (g_arena.base) cudaFree(g_arena.base);
    g_arena.base = NULL;
    g_arena.size = 0;
    g_arena.offset = 0;
}

// ---- cuBLAS GEMM wrappers ----

// BF16 GEMM: C = alpha * A @ B + beta * C
// A: [M, K], B: [K, N], C: [M, N]
static void gemm_bf16(cublasHandle_t handle,
                      const bf16* A, const bf16* B, bf16* C,
                      int M, int N, int K,
                      float alpha = 1.0f, float beta = 0.0f) {
    // cuBLAS is column-major, so we compute C.T = B.T @ A.T
    // which gives C = A @ B in row-major
    CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, CUDA_R_16BF, N,
        A, CUDA_R_16BF, K,
        &beta,
        C, CUDA_R_16BF, N,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));
}

// BF16 GEMM: C = alpha * A @ B.T (B transposed)
static void gemm_bf16_Bt(cublasHandle_t handle,
                         const bf16* A, const bf16* B, bf16* C,
                         int M, int N, int K,
                         float alpha = 1.0f, float beta = 0.0f) {
    // Row-major: C[M,N] = A[M,K] @ B[N,K].T
    // cuBLAS col-major: C.T = B @ A.T
    CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, CUDA_R_16BF, K,
        A, CUDA_R_16BF, K,
        &beta,
        C, CUDA_R_16BF, N,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));
}

// BF16 GEMM: C = alpha * A.T @ B (A transposed)
static void gemm_bf16_At(cublasHandle_t handle,
                         const bf16* A, const bf16* B, bf16* C,
                         int M, int N, int K,
                         float alpha = 1.0f, float beta = 0.0f) {
    // Row-major: C[M,N] = A[K,M].T @ B[K,N]
    // cuBLAS col-major: C.T = B.T @ A
    CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        N, M, K,
        &alpha,
        B, CUDA_R_16BF, N,
        A, CUDA_R_16BF, M,
        &beta,
        C, CUDA_R_16BF, N,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));
}

// Batched BF16 GEMM: C[b] = A[b] @ B[b] for b in [0, batch)
static void gemm_bf16_batched(cublasHandle_t handle,
                              const bf16* A, const bf16* B, bf16* C,
                              int M, int N, int K, int batch,
                              int strideA, int strideB, int strideC,
                              float alpha = 1.0f, float beta = 0.0f) {
    CUBLAS_CHECK(cublasGemmStridedBatchedEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, CUDA_R_16BF, N, strideB,
        A, CUDA_R_16BF, K, strideA,
        &beta,
        C, CUDA_R_16BF, N, strideC,
        batch,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));
}

// FP8 GEMM via cuBLAS LT: out[M,N] = x_f8[M,K] @ w_f8[K,N]
// x_f8, w_f8 stored row-major. cuBLAS uses column-major: swap operands.
// Row-major [R,C] ≡ col-major [C,R] with ld=C.
static void gemm_fp8_forward(cublasLtHandle_t handle,
                             const fp8e4m3* x_f8, const fp8e4m3* w_f8,
                             bf16* out,
                             int M, int N, int K,
                             float x_scale, float w_scale,
                             cudaStream_t stream) {
    cublasLtMatmulDesc_t matmulDesc;
    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc;

    CUBLAS_CHECK(cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

    // Standard row-major trick: out_cm[N,M] = w_cm[N,K] @ x_cm[K,M]
    // w_f8 row-major [K,N] → col-major [N,K,ld=N]
    // x_f8 row-major [M,K] → col-major [K,M,ld=K]
    // out   row-major [M,N] → col-major [N,M,ld=N]
    cublasOperation_t transA = CUBLAS_OP_N;
    cublasOperation_t transB = CUBLAS_OP_N;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB)));

    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_8F_E4M3, N, K, N));   // w_f8 cm: [N,K]
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_8F_E4M3, K, M, K));   // x_f8 cm: [K,M]
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16BF, N, M, N));       // out cm:  [N,M]

    float alpha = x_scale * w_scale;
    float beta_val = 0.0f;

    CUBLAS_CHECK(cublasLtMatmul(
        handle, matmulDesc,
        &alpha,
        w_f8, Adesc,
        x_f8, Bdesc,
        &beta_val,
        out, Cdesc,
        out, Cdesc,
        NULL, NULL, 0, stream
    ));

    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Cdesc);
    cublasLtMatmulDescDestroy(matmulDesc);
}

// FP8 backward: grad_x[M,K] = grad_f8[M,N] @ w_f8[K,N].T
// grad_f8 row-major [M,N], w_f8 row-major [K,N], grad_x row-major [M,K]
static void gemm_fp8_backward_grad_x(cublasLtHandle_t handle,
                                      const fp8e5m2* grad_f8, const fp8e4m3* w_f8,
                                      bf16* grad_x,
                                      int M, int N, int K,
                                      float grad_scale, float w_scale,
                                      cudaStream_t stream) {
    cublasLtMatmulDesc_t matmulDesc;
    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc;

    CUBLAS_CHECK(cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

    // Row-major trick: grad_x_cm[K,M] = w_cm.T[K,N] @ grad_cm[N,M]
    // w_f8 rm [K,N] → cm [N,K,ld=N], need transpose → op(A) = [K,N]
    // grad_f8 rm [M,N] → cm [N,M,ld=N], no transpose → op(B) = [N,M]
    // grad_x rm [M,K] → cm [K,M,ld=K]
    cublasOperation_t transA = CUBLAS_OP_T;
    cublasOperation_t transB = CUBLAS_OP_N;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB)));

    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_8F_E4M3, N, K, N));   // w_f8 cm: [N,K]
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_8F_E5M2, N, M, N));   // grad cm: [N,M]
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16BF, K, M, K));       // grad_x cm: [K,M]

    float alpha = grad_scale * w_scale;
    float beta_val = 0.0f;

    CUBLAS_CHECK(cublasLtMatmul(
        handle, matmulDesc,
        &alpha,
        w_f8, Adesc,
        grad_f8, Bdesc,
        &beta_val,
        grad_x, Cdesc,
        grad_x, Cdesc,
        NULL, NULL, 0, stream
    ));

    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Cdesc);
    cublasLtMatmulDescDestroy(matmulDesc);
}

// FP8 backward: grad_w[K,N] = x_f8[M,K].T @ grad_f8[M,N]
// x_f8 row-major [M,K], grad_f8 row-major [M,N], grad_w row-major [K,N]
static void gemm_fp8_backward_grad_w(cublasLtHandle_t handle,
                                      const fp8e4m3* x_f8, const fp8e5m2* grad_f8,
                                      float* grad_w,
                                      int M, int N, int K,
                                      float x_scale, float grad_scale,
                                      float beta,
                                      cudaStream_t stream) {
    cublasLtMatmulDesc_t matmulDesc;
    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc;

    CUBLAS_CHECK(cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

    // Row-major trick: grad_w_cm[N,K] = grad_cm[N,M] @ x_cm.T[M,K]
    // grad_f8 rm [M,N] → cm [N,M,ld=N], no transpose
    // x_f8 rm [M,K] → cm [K,M,ld=K], need transpose → op(B) = [M,K]
    // grad_w rm [K,N] → cm [N,K,ld=N]
    cublasOperation_t transA = CUBLAS_OP_N;
    cublasOperation_t transB = CUBLAS_OP_T;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB)));

    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_8F_E5M2, N, M, N));   // grad cm: [N,M]
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_8F_E4M3, K, M, K));   // x_f8 cm: [K,M]
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, N, K, N));        // grad_w cm: [N,K]

    float alpha = x_scale * grad_scale;

    CUBLAS_CHECK(cublasLtMatmul(
        handle, matmulDesc,
        &alpha,
        grad_f8, Adesc,
        x_f8, Bdesc,
        &beta,
        grad_w, Cdesc,
        grad_w, Cdesc,
        NULL, NULL, 0, stream
    ));

    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Cdesc);
    cublasLtMatmulDescDestroy(matmulDesc);
}

// ============================================================================
// Section A2: cuDNN Flash Attention (SDPA)
// ============================================================================

// Helper kernel: compute ragged offsets from cu_seqlens
// out[i] = cu_seqlens[i] * stride_mul  (converts token indices to element offsets)
__global__ void compute_ragged_offsets_kernel(int32_t* out, const int32_t* in, int stride_mul, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] * stride_mul;
}

static void compute_ragged_offsets(int32_t* out, const int32_t* in, int stride_mul, int n, cudaStream_t stream) {
    int block = 256;
    int grid = (n + block - 1) / block;
    compute_ragged_offsets_kernel<<<grid, block, 0, stream>>>(out, in, stride_mul, n);
}

// Helper kernel: compute per-sequence lengths from cu_seqlens
// seq_len[i] = cu_seqlens[i+1] - cu_seqlens[i], optionally scaled
__global__ void compute_seq_lens_kernel(int32_t* out, const int32_t* cu_seqlens, int scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (cu_seqlens[i + 1] - cu_seqlens[i]) * scale;
}

static void compute_seq_lens(int32_t* out, const int32_t* cu_seqlens, int scale, int n, cudaStream_t stream) {
    int block = 256;
    int grid = (n + block - 1) / block;
    compute_seq_lens_kernel<<<grid, block, 0, stream>>>(out, cu_seqlens, scale, n);
}

// Compute max sequence length from cu_seqlens (on host)
static int compute_s_max(const int32_t* h_cu_seqlens, int num_seqs) {
    int s_max = 0;
    for (int i = 0; i < num_seqs; i++) {
        int sl = h_cu_seqlens[i + 1] - h_cu_seqlens[i];
        if (sl > s_max) s_max = sl;
    }
    return s_max;
}

// Build cuDNN SDPA forward graph
static std::shared_ptr<fe::graph::Graph> build_cudnn_sdpa_forward(
    cudnnHandle_t handle,
    int num_seqs, int attn_H, int s_max, int HD,
    float attn_scale, int window_left)
{
    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(fe::DataType_t::BFLOAT16)
          .set_intermediate_data_type(fe::DataType_t::FLOAT)
          .set_compute_data_type(fe::DataType_t::FLOAT);

    // Q, K, V: [num_seqs, attn_H, s_max, HD] with ragged offsets
    auto Q = graph->tensor(fe::graph::Tensor_attributes()
        .set_name("Q")
        .set_uid(UID_Q)
        .set_dim({num_seqs, attn_H, s_max, HD})
        .set_stride({attn_H * s_max * HD, HD, attn_H * HD, 1})
        .set_data_type(fe::DataType_t::BFLOAT16)
        .set_ragged_offset(graph->tensor(fe::graph::Tensor_attributes()
            .set_name("seq_q")
            .set_uid(UID_SEQ_Q)
            .set_dim({num_seqs + 1, 1, 1, 1})
            .set_stride({1, 1, 1, 1})
            .set_data_type(fe::DataType_t::INT32))));


    auto K = graph->tensor(fe::graph::Tensor_attributes()
        .set_name("K")
        .set_uid(UID_K)
        .set_dim({num_seqs, attn_H, s_max, HD})
        .set_stride({attn_H * s_max * HD, HD, attn_H * HD, 1})
        .set_data_type(fe::DataType_t::BFLOAT16)
        .set_ragged_offset(graph->tensor(fe::graph::Tensor_attributes()
            .set_name("seq_kv")
            .set_uid(UID_SEQ_KV)
            .set_dim({num_seqs + 1, 1, 1, 1})
            .set_stride({1, 1, 1, 1})
            .set_data_type(fe::DataType_t::INT32))));


    auto V = graph->tensor(fe::graph::Tensor_attributes()
        .set_name("V")
        .set_uid(UID_V)
        .set_dim({num_seqs, attn_H, s_max, HD})
        .set_stride({attn_H * s_max * HD, HD, attn_H * HD, 1})
        .set_data_type(fe::DataType_t::BFLOAT16)
        .set_ragged_offset(K->get_ragged_offset()));


    // Per-sequence length tensors (required for padding mask with ragged offsets)
    auto Seq_len_q = graph->tensor(fe::graph::Tensor_attributes()
        .set_name("seq_len_q")
        .set_uid(UID_SEQ_LEN_Q)
        .set_dim({num_seqs, 1, 1, 1})
        .set_stride({1, 1, 1, 1})
        .set_data_type(fe::DataType_t::INT32));

    auto Seq_len_kv = graph->tensor(fe::graph::Tensor_attributes()
        .set_name("seq_len_kv")
        .set_uid(UID_SEQ_LEN_KV)
        .set_dim({num_seqs, 1, 1, 1})
        .set_stride({1, 1, 1, 1})
        .set_data_type(fe::DataType_t::INT32));

    // SDPA options — padding mask required for ragged offsets
    auto sdpa_opts = fe::graph::SDPA_attributes()
        .set_name("sdpa_forward")
        .set_is_inference(false)
        .set_causal_mask(true)
        .set_padding_mask(true)
        .set_seq_len_q(Seq_len_q)
        .set_seq_len_kv(Seq_len_kv)
        .set_attn_scale(attn_scale);

    if (window_left > 0) {
        sdpa_opts.set_sliding_window_length(window_left);
    }

    auto [O, Stats] = graph->sdpa(Q, K, V, sdpa_opts);


    O->set_output(true)
      .set_uid(UID_O)
      .set_dim({num_seqs, attn_H, s_max, HD})
      .set_stride({attn_H * s_max * HD, HD, attn_H * HD, 1})
      .set_data_type(fe::DataType_t::BFLOAT16);

    Stats->set_output(true)
          .set_uid(UID_STATS)
          .set_data_type(fe::DataType_t::FLOAT);

    auto status = graph->validate();
    if (!status.is_good()) {
        fprintf(stderr, "cuDNN SDPA fwd validate failed: %s\n", status.get_message().c_str());
        exit(1);
    }
    status = graph->build_operation_graph(handle);
    if (!status.is_good()) {
        fprintf(stderr, "cuDNN SDPA fwd build failed: %s\n", status.get_message().c_str());
        exit(1);
    }
    status = graph->create_execution_plans({fe::HeurMode_t::A});
    if (!status.is_good()) {
        fprintf(stderr, "cuDNN SDPA fwd plan failed: %s\n", status.get_message().c_str());
        exit(1);
    }
    status = graph->check_support(handle);
    if (!status.is_good()) {
        fprintf(stderr, "cuDNN SDPA fwd support check failed: %s\n", status.get_message().c_str());
        exit(1);
    }
    status = graph->build_plans(handle, fe::BuildPlanPolicy_t::HEURISTICS_CHOICE);
    if (!status.is_good()) {
        fprintf(stderr, "cuDNN SDPA fwd build_plans failed: %s\n", status.get_message().c_str());
        exit(1);
    }


    return graph;
}

// Build cuDNN SDPA backward graph
static std::shared_ptr<fe::graph::Graph> build_cudnn_sdpa_backward(
    cudnnHandle_t handle,
    int num_seqs, int attn_H, int s_max, int HD,
    float attn_scale, int window_left)
{
    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(fe::DataType_t::BFLOAT16)
          .set_intermediate_data_type(fe::DataType_t::FLOAT)
          .set_compute_data_type(fe::DataType_t::FLOAT);

    // Ragged offset tensors
    auto seq_q = graph->tensor(fe::graph::Tensor_attributes()
        .set_name("seq_q")
        .set_uid(UID_SEQ_Q)
        .set_dim({num_seqs + 1, 1, 1, 1})
        .set_stride({1, 1, 1, 1})
        .set_data_type(fe::DataType_t::INT32));

    auto seq_kv = graph->tensor(fe::graph::Tensor_attributes()
        .set_name("seq_kv")
        .set_uid(UID_SEQ_KV)
        .set_dim({num_seqs + 1, 1, 1, 1})
        .set_stride({1, 1, 1, 1})
        .set_data_type(fe::DataType_t::INT32));

    auto make_ragged_tensor = [&](const char* name, int uid) {
        return graph->tensor(fe::graph::Tensor_attributes()
            .set_name(name)
            .set_uid(uid)
            .set_dim({num_seqs, attn_H, s_max, HD})
            .set_stride({attn_H * s_max * HD, HD, attn_H * HD, 1})
            .set_data_type(fe::DataType_t::BFLOAT16)
            .set_ragged_offset(seq_q));
    };

    auto Q  = make_ragged_tensor("Q", UID_Q);
    auto O  = make_ragged_tensor("O", UID_O);
    auto dO = make_ragged_tensor("dO", UID_DO);

    auto K = graph->tensor(fe::graph::Tensor_attributes()
        .set_name("K")
        .set_uid(UID_K)
        .set_dim({num_seqs, attn_H, s_max, HD})
        .set_stride({attn_H * s_max * HD, HD, attn_H * HD, 1})
        .set_data_type(fe::DataType_t::BFLOAT16)
        .set_ragged_offset(seq_kv));

    auto V = graph->tensor(fe::graph::Tensor_attributes()
        .set_name("V")
        .set_uid(UID_V)
        .set_dim({num_seqs, attn_H, s_max, HD})
        .set_stride({attn_H * s_max * HD, HD, attn_H * HD, 1})
        .set_data_type(fe::DataType_t::BFLOAT16)
        .set_ragged_offset(seq_kv));

    auto Stats = graph->tensor(fe::graph::Tensor_attributes()
        .set_name("Stats")
        .set_uid(UID_STATS)
        .set_dim({num_seqs, attn_H, s_max, 1})
        .set_stride({attn_H * s_max, s_max, 1, 1})
        .set_data_type(fe::DataType_t::FLOAT));

    // Per-sequence length tensors (required for padding mask with ragged offsets)
    auto Seq_len_q = graph->tensor(fe::graph::Tensor_attributes()
        .set_name("seq_len_q")
        .set_uid(UID_SEQ_LEN_Q)
        .set_dim({num_seqs, 1, 1, 1})
        .set_stride({1, 1, 1, 1})
        .set_data_type(fe::DataType_t::INT32));

    auto Seq_len_kv = graph->tensor(fe::graph::Tensor_attributes()
        .set_name("seq_len_kv")
        .set_uid(UID_SEQ_LEN_KV)
        .set_dim({num_seqs, 1, 1, 1})
        .set_stride({1, 1, 1, 1})
        .set_data_type(fe::DataType_t::INT32));

    // SDPA backward options — padding mask required for ragged offsets
    auto sdpa_bwd_opts = fe::graph::SDPA_backward_attributes()
        .set_name("sdpa_backward")
        .set_causal_mask(true)
        .set_padding_mask(true)
        .set_seq_len_q(Seq_len_q)
        .set_seq_len_kv(Seq_len_kv)
        .set_attn_scale(attn_scale);

    if (window_left > 0) {
        sdpa_bwd_opts.set_sliding_window_length(window_left);
    }

    auto [dQ, dK, dV] = graph->sdpa_backward(Q, K, V, O, dO, Stats, sdpa_bwd_opts);

    dQ->set_output(true)
       .set_uid(UID_DQ)
       .set_dim({num_seqs, attn_H, s_max, HD})
       .set_stride({attn_H * s_max * HD, HD, attn_H * HD, 1})
       .set_data_type(fe::DataType_t::BFLOAT16);

    dK->set_output(true)
       .set_uid(UID_DK)
       .set_dim({num_seqs, attn_H, s_max, HD})
       .set_stride({attn_H * s_max * HD, HD, attn_H * HD, 1})
       .set_data_type(fe::DataType_t::BFLOAT16);

    dV->set_output(true)
       .set_uid(UID_DV)
       .set_dim({num_seqs, attn_H, s_max, HD})
       .set_stride({attn_H * s_max * HD, HD, attn_H * HD, 1})
       .set_data_type(fe::DataType_t::BFLOAT16);

    auto status = graph->validate();
    if (!status.is_good()) {
        fprintf(stderr, "cuDNN SDPA bwd validate failed: %s\n", status.get_message().c_str());
        exit(1);
    }
    status = graph->build_operation_graph(handle);
    if (!status.is_good()) {
        fprintf(stderr, "cuDNN SDPA bwd build failed: %s\n", status.get_message().c_str());
        exit(1);
    }
    status = graph->create_execution_plans({fe::HeurMode_t::A});
    if (!status.is_good()) {
        fprintf(stderr, "cuDNN SDPA bwd plan failed: %s\n", status.get_message().c_str());
        exit(1);
    }
    status = graph->check_support(handle);
    if (!status.is_good()) {
        fprintf(stderr, "cuDNN SDPA bwd support check failed: %s\n", status.get_message().c_str());
        exit(1);
    }
    status = graph->build_plans(handle, fe::BuildPlanPolicy_t::HEURISTICS_CHOICE);
    if (!status.is_good()) {
        fprintf(stderr, "cuDNN SDPA bwd build_plans failed: %s\n", status.get_message().c_str());
        exit(1);
    }

    return graph;
}

// Round up to next power of 2 (for graph caching)
static int bucket_num_seqs(int n) {
    if (n <= 1) return 1;
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

// Get or build a cached cuDNN SDPA graph
// Uses bucketed num_seqs to minimize graph rebuilds
static std::shared_ptr<fe::graph::Graph> get_or_build_graph(
    TrainingContext* ctx, int num_seqs, int attn_H, int s_max,
    int window_left, int HD, float attn_scale, bool is_backward)
{
    int bucketed = bucket_num_seqs(num_seqs);
    CudnnGraphCache::Key key = {bucketed, attn_H, s_max, window_left, attn_scale, is_backward};
    auto it = ctx->cudnn_graph_cache.cache.find(key);
    if (it != ctx->cudnn_graph_cache.cache.end()) {
        return it->second;
    }

    auto graph = is_backward
        ? build_cudnn_sdpa_backward(ctx->cudnn_handle, bucketed, attn_H, s_max, HD, attn_scale, window_left)
        : build_cudnn_sdpa_forward(ctx->cudnn_handle, bucketed, attn_H, s_max, HD, attn_scale, window_left);

    // Check workspace requirement and grow if needed
    size_t needed = graph->get_workspace_size();
    if (needed > ctx->act.attn_workspace_size) {
        fprintf(stderr, "  cuDNN workspace: growing from %zu to %zu bytes\n",
                ctx->act.attn_workspace_size, needed * 2);
        cudaFree(ctx->act.attn_workspace);
        ctx->act.attn_workspace_size = needed * 2;
        CUDA_CHECK(cudaMalloc(&ctx->act.attn_workspace, ctx->act.attn_workspace_size));
    }

    ctx->cudnn_graph_cache.cache[key] = graph;
    return graph;
}

// Execute cuDNN SDPA forward
static void execute_cudnn_sdpa_forward(
    TrainingContext* ctx,
    std::shared_ptr<fe::graph::Graph>& graph,
    bf16* Q, bf16* K, bf16* V,
    bf16* O, float* stats,
    int32_t* ragged_offsets, int32_t* seq_lens,
    int num_seqs, int attn_H, int s_max, int HD)
{
    std::unordered_map<int64_t, void*> variant_pack = {
        {UID_Q, Q}, {UID_K, K}, {UID_V, V},
        {UID_O, O}, {UID_STATS, stats},
        {UID_SEQ_Q, ragged_offsets}, {UID_SEQ_KV, ragged_offsets},
        {UID_SEQ_LEN_Q, seq_lens}, {UID_SEQ_LEN_KV, seq_lens}
    };

    auto status = graph->execute(ctx->cudnn_handle, variant_pack, ctx->act.attn_workspace);
    if (!status.is_good()) {
        fprintf(stderr, "cuDNN SDPA fwd execute failed: %s\n", status.get_message().c_str());
        exit(1);
    }
}

// Execute cuDNN SDPA backward
static void execute_cudnn_sdpa_backward(
    TrainingContext* ctx,
    std::shared_ptr<fe::graph::Graph>& graph,
    bf16* dQ, bf16* dK, bf16* dV,
    bf16* dO, bf16* Q, bf16* K, bf16* V,
    bf16* O, float* stats,
    int32_t* ragged_offsets, int32_t* seq_lens,
    int num_seqs, int attn_H, int s_max, int HD)
{
    std::unordered_map<int64_t, void*> variant_pack = {
        {UID_Q, Q}, {UID_K, K}, {UID_V, V},
        {UID_O, O}, {UID_DO, dO}, {UID_STATS, stats},
        {UID_DQ, dQ}, {UID_DK, dK}, {UID_DV, dV},
        {UID_SEQ_Q, ragged_offsets}, {UID_SEQ_KV, ragged_offsets},
        {UID_SEQ_LEN_Q, seq_lens}, {UID_SEQ_LEN_KV, seq_lens}
    };

    auto status = graph->execute(ctx->cudnn_handle, variant_pack, ctx->act.attn_workspace);
    if (!status.is_good()) {
        fprintf(stderr, "cuDNN SDPA bwd execute failed: %s\n", status.get_message().c_str());
        exit(1);
    }
}

// ============================================================================
// Section B: Data Loading
// ============================================================================

typedef struct {
    uint16_t* tokens;
    int64_t num_tokens;
    int fd;
    void* mmap_ptr;
    size_t mmap_size;
} DataShard;

static DataShard load_data_shard(const char* filename) {
    DataShard shard = {};

    int fd = open(filename, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "Failed to open %s\n", filename);
        exit(1);
    }

    struct stat st;
    fstat(fd, &st);
    size_t file_size = st.st_size;

    void* mapped = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapped == MAP_FAILED) {
        fprintf(stderr, "mmap failed for %s\n", filename);
        exit(1);
    }

    int32_t* header = (int32_t*)mapped;
    if (header[0] != 20240520) {
        fprintf(stderr, "Magic number mismatch in %s: got %d\n", filename, header[0]);
        exit(1);
    }
    if (header[1] != 1) {
        fprintf(stderr, "Unsupported version %d in %s\n", header[1], filename);
        exit(1);
    }

    int64_t num_tokens = header[2];
    uint16_t* tokens = (uint16_t*)((char*)mapped + 256 * sizeof(int32_t));

    shard.tokens = tokens;
    shard.num_tokens = num_tokens;
    shard.fd = fd;
    shard.mmap_ptr = mapped;
    shard.mmap_size = file_size;
    return shard;
}

static void unload_data_shard(DataShard* shard) {
    if (shard->mmap_ptr) munmap(shard->mmap_ptr, shard->mmap_size);
    if (shard->fd >= 0) close(shard->fd);
    memset(shard, 0, sizeof(*shard));
}

// Find BOS positions for aligned batching
static int find_bos_positions(const uint16_t* tokens, int64_t num_tokens,
                              int64_t* bos_positions, int max_bos) {
    int count = 0;
    for (int64_t i = 0; i < num_tokens && count < max_bos; i++) {
        if (tokens[i] == BOS_ID) {
            bos_positions[count++] = i;
        }
    }
    return count;
}

// Construct a BOS-aligned batch
// Returns total tokens in batch (including +1 for targets offset)
static int construct_aligned_batch(const uint16_t* shard_tokens, int64_t shard_size,
                                   const int64_t* bos_pos, int num_bos, int* bos_idx_ptr,
                                   int num_tokens_local, int max_seq_len,
                                   uint16_t* buf_out, int32_t* cum_lengths_out, int max_docs,
                                   int* num_seqs_out) {
    int bos_idx = *bos_idx_ptr;
    int cur_len = 0;
    int doc_count = 0;

    cum_lengths_out[0] = 0;

    while (cur_len <= num_tokens_local) {
        if (bos_idx >= num_bos) return -1;  // exhausted

        int64_t start = bos_pos[bos_idx];
        int64_t end;
        if (bos_idx + 1 < num_bos) {
            end = bos_pos[bos_idx + 1];
        } else {
            end = shard_size;
        }
        // Clamp to max_seq_len
        if (end - start > max_seq_len) end = start + max_seq_len;
        // Clamp to remaining budget
        if ((int)(end - start) > num_tokens_local - cur_len + 1)
            end = start + num_tokens_local - cur_len + 1;

        // Copy tokens
        int len = (int)(end - start);
        memcpy(buf_out + cur_len, shard_tokens + start, len * sizeof(uint16_t));
        cur_len += len;
        doc_count++;
        if (doc_count < max_docs)
            cum_lengths_out[doc_count] = cur_len;
        bos_idx++;
    }

    // Adjust last doc
    cum_lengths_out[doc_count] = num_tokens_local;

    *bos_idx_ptr = bos_idx;
    if (num_seqs_out) *num_seqs_out = doc_count;
    return cur_len;
}

// ============================================================================
// Section C: Parameter Initialization
// ============================================================================

// Helper: fill buffer with uniform random [-bound, bound]
static void init_uniform(bf16* ptr, int n, float bound, curandGenerator_t gen) {
    // Host-side initialization with rand() — only called at model init, not hot path
    float* host = (float*)malloc((size_t)n * sizeof(float));
    for (int i = 0; i < n; i++) {
        host[i] = ((float)rand() / RAND_MAX) * 2.0f * bound - bound;
    }
    bf16* host_bf16 = (bf16*)malloc((size_t)n * sizeof(bf16));
    for (int i = 0; i < n; i++) {
        host_bf16[i] = __float2bfloat16(host[i]);
    }
    CUDA_CHECK(cudaMemcpy(ptr, host_bf16, (size_t)n * sizeof(bf16), cudaMemcpyHostToDevice));
    free(host);
    free(host_bf16);
}

static void init_normal(bf16* ptr, int n, float mean, float std_dev) {
    float* host = (float*)malloc(n * sizeof(float));
    // Box-Muller transform
    for (int i = 0; i < n; i += 2) {
        float u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
        float u2 = ((float)rand()) / ((float)RAND_MAX);
        float z0 = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
        float z1 = sqrtf(-2.0f * logf(u1)) * sinf(2.0f * M_PI * u2);
        host[i] = mean + std_dev * z0;
        if (i + 1 < n) host[i + 1] = mean + std_dev * z1;
    }
    bf16* host_bf16 = (bf16*)malloc(n * sizeof(bf16));
    for (int i = 0; i < n; i++) {
        host_bf16[i] = __float2bfloat16(host[i]);
    }
    CUDA_CHECK(cudaMemcpy(ptr, host_bf16, n * sizeof(bf16), cudaMemcpyHostToDevice));
    free(host);
    free(host_bf16);
}

static void init_zeros(bf16* ptr, int n) {
    CUDA_CHECK(cudaMemset(ptr, 0, n * sizeof(bf16)));
}

static void init_zeros_f32(float* ptr, int n) {
    CUDA_CHECK(cudaMemset(ptr, 0, n * sizeof(float)));
}

static void init_constant(bf16* ptr, int n, float val) {
    bf16* host = (bf16*)malloc(n * sizeof(bf16));
    bf16 v = __float2bfloat16(val);
    for (int i = 0; i < n; i++) host[i] = v;
    CUDA_CHECK(cudaMemcpy(ptr, host, n * sizeof(bf16), cudaMemcpyHostToDevice));
    free(host);
}


void init_model_params(TrainingContext* ctx) {
    ModelParams* p = &ctx->params;
    curandGenerator_t gen = ctx->curand_gen;

    srand(42);  // Deterministic initialization

    // Compute sizes
    int embed_size = VOCAB_SIZE * MODEL_DIM;
    int bigram_embed_size = BIGRAM_VOCAB_SIZE * MODEL_DIM;
    int value_embeds_size = 5 * VOCAB_SIZE * MODEL_DIM;
    int attn_bank_size = NUM_ATTN_LAYERS * 4 * MODEL_DIM * MODEL_DIM;  // [10, 3072, 768]
    int mlp_bank_size = MLP_BANK_SIZE * 2 * MLP_HDIM * MODEL_DIM;      // [12, 2, 3072, 768]
    int lm_head_size = MODEL_DIM * VOCAB_SIZE;
    int attn_gate_size = 10 * NUM_HEADS * 12;
    int ve_gate_size = 5 * NUM_HEADS * 12;
    int smear_gate_size = 1 * 12;
    int skip_gate_size = 1 * 12;
    int post_lambdas_size = NUM_LAYERS * 2 * 2;
    int x0_lambdas_size = NUM_LAYERS;
    int bigram_lambdas_size = NUM_LAYERS;
    int resid_lambdas_size = NUM_LAYERS * 2;
    int scalars_pad = (-(NUM_LAYERS * 2 + 3)) % 1;  // world_size=1, so pad=0
    if (scalars_pad < 0) scalars_pad += 1;
    int scalars_size_val = NUM_LAYERS * 2 + 3 + scalars_pad;
    p->scalars_size = scalars_size_val;

    // Allocate parameters
    CUDA_CHECK(cudaMalloc(&p->embed, embed_size * sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&p->bigram_embed, bigram_embed_size * sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&p->value_embeds, value_embeds_size * sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&p->attn_bank, attn_bank_size * sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&p->mlp_bank, mlp_bank_size * sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&p->lm_head, lm_head_size * sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&p->attn_gate_bank, attn_gate_size * sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&p->ve_gate_bank, ve_gate_size * sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&p->smear_gate, smear_gate_size * sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&p->skip_gate, skip_gate_size * sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&p->post_lambdas, post_lambdas_size * sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&p->x0_lambdas, x0_lambdas_size * sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&p->bigram_lambdas, bigram_lambdas_size * sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&p->resid_lambdas, resid_lambdas_size * sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&p->scalars, scalars_size_val * sizeof(bf16)));

    // Initialize parameters matching PyTorch init
    float std_val = 0.5f * powf((float)MODEL_DIM, -0.5f);
    float bound = sqrtf(3.0f) * std_val;

    // attn_bank: uniform(-bound, bound) for all [10, 3072, 768]
    init_uniform(p->attn_bank, attn_bank_size, bound, gen);

    // mlp_bank: c_fc uniform(-bound, bound), c_proj zeros
    // mlp_bank is [12, 2, 3072, 768]: [i,0,:,:] = c_fc, [i,1,:,:] = c_proj
    {
        int layer_size = 2 * MLP_HDIM * MODEL_DIM;
        int fc_size = MLP_HDIM * MODEL_DIM;
        for (int i = 0; i < MLP_BANK_SIZE; i++) {
            init_uniform(p->mlp_bank + i * layer_size, fc_size, bound, gen);  // c_fc
            init_zeros(p->mlp_bank + i * layer_size + fc_size, fc_size);       // c_proj
        }
    }

    // lm_head: normal(0, 0.005), transposed layout [MODEL_DIM, VOCAB_SIZE]
    init_normal(p->lm_head, lm_head_size, 0.0f, 0.005f);

    // embed: copy from lm_head.T -> embed[v, d] = lm_head[d, v]
    transpose_copy(p->lm_head, p->embed, MODEL_DIM, VOCAB_SIZE, ctx->stream);
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));

    // value_embeds: 0.01 * randn
    init_normal(p->value_embeds, value_embeds_size, 0.0f, 0.01f);

    // bigram_embed: zeros
    init_zeros(p->bigram_embed, bigram_embed_size);

    // Gates: zeros
    init_zeros(p->attn_gate_bank, attn_gate_size);
    init_zeros(p->ve_gate_bank, ve_gate_size);
    init_zeros(p->smear_gate, smear_gate_size);
    init_zeros(p->skip_gate, skip_gate_size);

    // post_lambdas: ones, except [7-10, 0, 1] = 1.5
    {
        float* host = (float*)malloc(post_lambdas_size * sizeof(float));
        for (int i = 0; i < post_lambdas_size; i++) host[i] = 1.0f;
        for (int layer = PARALLEL_START; layer < NUM_LAYERS; layer++) {
            host[layer * 4 + 0 * 2 + 1] = 1.5f;
        }
        bf16* host_bf16 = (bf16*)malloc(post_lambdas_size * sizeof(bf16));
        for (int i = 0; i < post_lambdas_size; i++)
            host_bf16[i] = __float2bfloat16(host[i]);
        CUDA_CHECK(cudaMemcpy(p->post_lambdas, host_bf16, post_lambdas_size * sizeof(bf16), cudaMemcpyHostToDevice));
        free(host);
        free(host_bf16);
    }

    // x0_lambdas: zeros
    init_zeros(p->x0_lambdas, x0_lambdas_size);

    // bigram_lambdas: 0.05
    init_constant(p->bigram_lambdas, bigram_lambdas_size, 0.05f);

    // resid_lambdas: sqrt(1.1) for all
    init_constant(p->resid_lambdas, resid_lambdas_size, sqrtf(1.1f));

    // scalars: [sa_lambda_0, sa_lambda_1, ...x11..., smear_lambda, backout_lambda, skip_lambda]
    {
        float* host = (float*)malloc(scalars_size_val * sizeof(float));
        memset(host, 0, scalars_size_val * sizeof(float));
        for (int i = 0; i < NUM_LAYERS; i++) {
            host[2 * i] = 0.5f;      // sa_lambdas[i][0]
            host[2 * i + 1] = 1.0f;  // sa_lambdas[i][1]
        }
        host[2 * NUM_LAYERS] = 0.0f;     // smear_lambda
        host[2 * NUM_LAYERS + 1] = 0.5f; // backout_lambda
        host[2 * NUM_LAYERS + 2] = -1.5f; // skip_lambda -> sigmoid(-1.5) ≈ 0.18
        bf16* host_bf16 = (bf16*)malloc(scalars_size_val * sizeof(bf16));
        for (int i = 0; i < scalars_size_val; i++)
            host_bf16[i] = __float2bfloat16(host[i]);
        CUDA_CHECK(cudaMemcpy(p->scalars, host_bf16, scalars_size_val * sizeof(bf16), cudaMemcpyHostToDevice));
        free(host);
        free(host_bf16);
    }
}

// ============================================================================
// Section C2: Allocate gradients
// ============================================================================

static void alloc_grads(ModelGrads* g) {
    CUDA_CHECK(cudaMalloc(&g->embed, VOCAB_SIZE * MODEL_DIM * sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&g->bigram_embed, BIGRAM_VOCAB_SIZE * MODEL_DIM * sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&g->value_embeds, 5 * VOCAB_SIZE * MODEL_DIM * sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&g->attn_bank, NUM_ATTN_LAYERS * 4 * MODEL_DIM * MODEL_DIM * sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&g->mlp_bank, MLP_BANK_SIZE * 2 * MLP_HDIM * MODEL_DIM * sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&g->lm_head, MODEL_DIM * VOCAB_SIZE * sizeof(bf16)));    // BF16
    CUDA_CHECK(cudaMalloc(&g->attn_gate_bank, 10 * NUM_HEADS * 12 * sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&g->ve_gate_bank, 5 * NUM_HEADS * 12 * sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&g->smear_gate, 1 * 12 * sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&g->skip_gate, 1 * 12 * sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&g->post_lambdas, NUM_LAYERS * 2 * 2 * sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&g->x0_lambdas, NUM_LAYERS * sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&g->bigram_lambdas, NUM_LAYERS * sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&g->resid_lambdas, NUM_LAYERS * 2 * sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&g->scalars, (NUM_LAYERS * 2 + 3 + 1) * sizeof(bf16)));
}

static void zero_normuon_grads(ModelGrads* g, cudaStream_t stream) {
    // Only zero NorMuon parameter grads (attn_bank, mlp_bank)
    CUDA_CHECK(cudaMemsetAsync(g->attn_bank, 0, NUM_ATTN_LAYERS * 4 * MODEL_DIM * MODEL_DIM * sizeof(bf16), stream));
    CUDA_CHECK(cudaMemsetAsync(g->mlp_bank, 0, MLP_BANK_SIZE * 2 * MLP_HDIM * MODEL_DIM * sizeof(bf16), stream));
}

static void zero_adam_grads(ModelGrads* g, cudaStream_t stream) {
    // Zero Adam parameter grads (everything except attn_bank, mlp_bank)
    CUDA_CHECK(cudaMemsetAsync(g->embed, 0, VOCAB_SIZE * MODEL_DIM * sizeof(bf16), stream));
    CUDA_CHECK(cudaMemsetAsync(g->bigram_embed, 0, BIGRAM_VOCAB_SIZE * MODEL_DIM * sizeof(bf16), stream));
    CUDA_CHECK(cudaMemsetAsync(g->value_embeds, 0, 5 * VOCAB_SIZE * MODEL_DIM * sizeof(bf16), stream));
    CUDA_CHECK(cudaMemsetAsync(g->lm_head, 0, MODEL_DIM * VOCAB_SIZE * sizeof(bf16), stream));
    CUDA_CHECK(cudaMemsetAsync(g->attn_gate_bank, 0, 10 * NUM_HEADS * 12 * sizeof(bf16), stream));
    CUDA_CHECK(cudaMemsetAsync(g->ve_gate_bank, 0, 5 * NUM_HEADS * 12 * sizeof(bf16), stream));
    CUDA_CHECK(cudaMemsetAsync(g->smear_gate, 0, 1 * 12 * sizeof(bf16), stream));
    CUDA_CHECK(cudaMemsetAsync(g->skip_gate, 0, 1 * 12 * sizeof(bf16), stream));
    CUDA_CHECK(cudaMemsetAsync(g->post_lambdas, 0, NUM_LAYERS * 2 * 2 * sizeof(bf16), stream));
    CUDA_CHECK(cudaMemsetAsync(g->x0_lambdas, 0, NUM_LAYERS * sizeof(bf16), stream));
    CUDA_CHECK(cudaMemsetAsync(g->bigram_lambdas, 0, NUM_LAYERS * sizeof(bf16), stream));
    CUDA_CHECK(cudaMemsetAsync(g->resid_lambdas, 0, NUM_LAYERS * 2 * sizeof(bf16), stream));
    CUDA_CHECK(cudaMemsetAsync(g->scalars, 0, (NUM_LAYERS * 2 + 3 + 1) * sizeof(bf16), stream));
}

static void zero_grads(ModelGrads* g, cudaStream_t stream) {
    zero_normuon_grads(g, stream);
    zero_adam_grads(g, stream);
}

// ============================================================================
// Section C3: Optimizer state allocation
// ============================================================================

static void init_adam_state(AdamState* s, int numel) {
    s->numel = numel;
    s->step = 0;
    CUDA_CHECK(cudaMalloc(&s->exp_avg, numel * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s->exp_avg_sq, numel * sizeof(float)));
    init_zeros_f32(s->exp_avg, numel);
    init_zeros_f32(s->exp_avg_sq, numel);
}

static void init_muon_state(NorMuonState* s, int numel, const int* shape, int ndim,
                            int d0, int d1) {
    s->numel = numel;
    s->chunk_ndim = ndim;
    for (int i = 0; i < ndim && i < 4; i++) s->chunk_shape[i] = shape[i];

    CUDA_CHECK(cudaMalloc(&s->momentum_buffer, numel * sizeof(float)));
    init_zeros_f32(s->momentum_buffer, numel);

    // Second momentum: reduced along one dim
    int second_mom_size;
    if (d0 >= d1) {
        second_mom_size = (numel / d1); // reduce along d1 -> [outer, d0, 1]
    } else {
        second_mom_size = (numel / d0); // reduce along d0 -> [outer, 1, d1]
    }
    CUDA_CHECK(cudaMalloc(&s->second_momentum_buffer, second_mom_size * sizeof(float)));
    init_zeros_f32(s->second_momentum_buffer, second_mom_size);

    CUDA_CHECK(cudaMalloc(&s->mantissa, numel * sizeof(uint16_t)));
    CUDA_CHECK(cudaMemset(s->mantissa, 0, numel * sizeof(uint16_t)));
}

static void alloc_optimizer_state(OptimizerState* opt) {
    // Adam states
    init_adam_state(&opt->adam_scalars, NUM_LAYERS * 2 + 3);
    init_adam_state(&opt->adam_smear_gate, 12);
    init_adam_state(&opt->adam_skip_gate, 12);
    init_adam_state(&opt->adam_attn_gate_bank, 10 * NUM_HEADS * 12);
    init_adam_state(&opt->adam_ve_gate_bank, 5 * NUM_HEADS * 12);
    init_adam_state(&opt->adam_lm_head, MODEL_DIM * VOCAB_SIZE);
    init_adam_state(&opt->adam_embed, VOCAB_SIZE * MODEL_DIM);
    init_adam_state(&opt->adam_bigram_embed, BIGRAM_VOCAB_SIZE * MODEL_DIM);
    init_adam_state(&opt->adam_value_embeds, 5 * VOCAB_SIZE * MODEL_DIM);
    init_adam_state(&opt->adam_post_lambdas, NUM_LAYERS * 2 * 2);
    init_adam_state(&opt->adam_x0_lambdas, NUM_LAYERS);
    init_adam_state(&opt->adam_bigram_lambdas, NUM_LAYERS);
    init_adam_state(&opt->adam_resid_lambdas, NUM_LAYERS * 2);

    // NorMuon states
    // attn_bank reshaped: (40, 768, 768) -> single GPU chunk is all of it
    {
        int shape[] = {NUM_ATTN_LAYERS * 4, MODEL_DIM, MODEL_DIM};
        int numel = shape[0] * shape[1] * shape[2];
        init_muon_state(&opt->muon_attn_bank, numel, shape, 3, shape[1], shape[2]);
    }
    // mlp_bank reshaped: (24, 3072, 768)
    {
        int shape[] = {24, MLP_HDIM, MODEL_DIM};
        int numel = shape[0] * shape[1] * shape[2];
        init_muon_state(&opt->muon_mlp_bank, numel, shape, 3, shape[1], shape[2]);
    }

    opt->split_embed = 0;
    // split_step at 2/3 of training, on odd step
    int split_step = (int)roundf(NUM_SCHEDULED_ITERS * 2.0f / 3.0f);
    if (split_step % 2 == 0) split_step |= 1;
    opt->split_step = split_step;

    // Polar Express scratch buffers (sized for largest usage: mlp_bank)
    // attn: batch=40, small_dim=768 (square), elems=40*768*768
    // mlp: batch=24, small_dim=768 (tall, cols=768), elems=24*3072*768
    int max_batch = 40;
    int max_small_dim = MODEL_DIM;  // 768
    int max_elems = 24 * MLP_HDIM * MODEL_DIM;  // 24*3072*768 > 40*768*768
    CUDA_CHECK(cudaMalloc(&opt->polar_norms, max_batch * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&opt->polar_A, max_batch * max_small_dim * max_small_dim * sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&opt->polar_B, max_batch * max_small_dim * max_small_dim * sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&opt->polar_C, max_elems * sizeof(bf16)));
}

// ============================================================================
// Section C4: Activation allocation
// ============================================================================

// Helper to bump-allocate from a byte pointer with alignment
static inline char* bump_alloc(char*& ptr, size_t bytes) {
    char* result = ptr;
    ptr += bytes;
    // Align to 256 bytes for GPU coalescing
    ptr = (char*)(((uintptr_t)ptr + 255) & ~(uintptr_t)255);
    return result;
}

static void alloc_activations(Activations* act, int max_tokens) {
    int T = max_tokens;
    int D = MODEL_DIM;
    int H = NUM_HEADS;
    int HD = HEAD_DIM;

    // Compute total size needed for all activation buffers (except workspaces)
    size_t TD2 = (size_t)T * D * sizeof(bf16);
    size_t THD2 = (size_t)T * H * HD * sizeof(bf16);
    size_t TMLP2 = (size_t)T * MLP_HDIM * sizeof(bf16);
    size_t TV2 = (size_t)T * VOCAB_SIZE * sizeof(bf16);
    size_t TV1 = (size_t)T * VOCAB_SIZE;  // FP8
    size_t ALIGN = 256;  // alignment padding per buffer

    // Count all buffers and compute total (with generous alignment padding)
    size_t total = 0;
    int n_buffers = 0;

    // Basic bf16 buffers: x, x0, x0_bigram, lane0, lane1, normed, mlp_out, skip_save, x_backout
    total += 9 * (TD2 + ALIGN); n_buffers += 9;
    // Attention: qkv, q, k, v, attn_out (all T*H*HD bf16, qkv is 3x)
    total += (size_t)T * 3 * H * HD * sizeof(bf16) + ALIGN; n_buffers++; // qkv
    total += 4 * (THD2 + ALIGN); n_buffers += 4; // q,k,v,attn_out
    // MLP: mlp_pre, mlp_post
    total += 2 * (TMLP2 + ALIGN); n_buffers += 2;
    // FP8: x_f8, w_f8
    total += (size_t)T * D + ALIGN; n_buffers++; // x_f8
    total += (size_t)D * VOCAB_SIZE + ALIGN; n_buffers++; // w_f8
    // Logits + grad_logits
    total += TV2 + ALIGN; n_buffers++; // logits
    total += TV1 + ALIGN; n_buffers++; // grad_logits (fp8e5m2)
    // Loss
    total += 2 * ((size_t)T * sizeof(float) + ALIGN); n_buffers += 2; // losses, lse
    // Per-layer saved (11 layers * 10 buffers)
    total += NUM_LAYERS * (8 * (TD2 + ALIGN) + 2 * (TMLP2 + ALIGN)); n_buffers += NUM_LAYERS * 10;
    // Per-attn-layer saved (10 layers * 3 buffers)
    total += NUM_ATTN_LAYERS * ((size_t)T * 3 * H * HD * sizeof(bf16) + ALIGN + THD2 + ALIGN + (size_t)T * H * sizeof(bf16) + ALIGN);
    n_buffers += NUM_ATTN_LAYERS * 3;
    // Saved VE gate values: 5 VE layers * [T, H] bf16
    total += 5 * ((size_t)T * H * sizeof(bf16) + ALIGN);
    n_buffers += 5;
    // Gradient scratch: 5 * T*D bf16
    total += 5 * (TD2 + ALIGN); n_buffers += 5;
    // cuDNN attn stats: [num_seqs, attn_H, s_max, 1] per attention layer
    // s_max is at most 2*2048=4096 (paired heads), attn_H * s_max ≤ 12288
    // num_seqs_max ≈ T / min_seq_len (BOS-aligned, min ~128)
    int num_seqs_max_est = T / 128 + 16;
    int stats_per_seq = H * 2048;  // max(attn_H * s_max) = 6*2048 = 3*4096 = 12288
    total += NUM_ATTN_LAYERS * ((size_t)num_seqs_max_est * stats_per_seq * sizeof(float) + ALIGN);
    n_buffers += NUM_ATTN_LAYERS;
    // cu_seqlens_paired scratch (max 4096 entries)
    total += 4096 * sizeof(int32_t) + ALIGN; n_buffers += 1;
    // YaRN tables
    int max_sl = VAL_BATCH_SIZE / GRAD_ACCUM_STEPS;
    total += 2 * (2 * (size_t)max_sl * HD * sizeof(bf16) + ALIGN); // cos, sin
    total += 2 * (2 * (size_t)max_sl * 2 * HD * sizeof(bf16) + ALIGN); // paired cos, sin
    n_buffers += 4;
    // Scratch buffers
    size_t scratch1_size = (size_t)T * D * sizeof(bf16);
    size_t lm_head_conv_size = (size_t)D * VOCAB_SIZE * sizeof(bf16);
    if (lm_head_conv_size > scratch1_size) scratch1_size = lm_head_conv_size;
    total += scratch1_size + ALIGN; // scratch1
    total += TD2 + ALIGN; // scratch2
    total += (size_t)T * D * sizeof(float) + ALIGN; // scratch_f32
    n_buffers += 3;

    fprintf(stderr, "  alloc_activations: %d buffers, total %.1f GB\n", n_buffers, total / 1e9);

    // Single massive allocation
    char* base;
    CUDA_CHECK(cudaMalloc(&base, total));
    char* p = base;

    // Macro for bump allocation
    #define BUMP(type, field, nbytes) act->field = (type*)bump_alloc(p, nbytes)

    // Basic buffers
    BUMP(bf16, x, TD2);
    BUMP(bf16, x0, TD2);
    BUMP(bf16, x0_bigram, TD2);
    BUMP(bf16, lane0, TD2);
    BUMP(bf16, lane1, TD2);

    // Attention buffers
    BUMP(bf16, qkv, (size_t)T * 3 * H * HD * sizeof(bf16));
    BUMP(bf16, q, THD2);
    BUMP(bf16, k, THD2);
    BUMP(bf16, v, THD2);
    BUMP(bf16, attn_out, THD2);
    BUMP(bf16, normed, TD2);

    // MLP buffers
    BUMP(bf16, mlp_pre, TMLP2);
    BUMP(bf16, mlp_post, TMLP2);
    BUMP(bf16, mlp_out, TD2);

    // FP8 buffers
    BUMP(fp8e4m3, x_f8, (size_t)T * D);
    BUMP(fp8e4m3, w_f8, (size_t)D * VOCAB_SIZE);
    BUMP(bf16, logits, TV2);

    // Loss buffers
    BUMP(float, losses, (size_t)T * sizeof(float));
    BUMP(float, lse, (size_t)T * sizeof(float));
    BUMP(fp8e5m2, grad_logits, TV1);

    // Skip buffers
    BUMP(bf16, skip_save, TD2);
    BUMP(bf16, x_backout, TD2);

    // Per-layer saved activations
    act->saved_bulk = (bf16*)base;  // track for cleanup
    for (int i = 0; i < NUM_LAYERS; i++) {
        BUMP(bf16, saved_normed[i], TD2);
        BUMP(bf16, saved_normed_mlp[i], TD2);
        BUMP(bf16, saved_post_attn[i], TD2);
        BUMP(bf16, saved_lane0[i], TD2);
        BUMP(bf16, saved_lane1[i], TD2);
        BUMP(bf16, saved_lane1_post_attn[i], TD2);
        BUMP(bf16, saved_mlp_pre[i], TMLP2);
        BUMP(bf16, saved_mlp_post[i], TMLP2);
        BUMP(bf16, saved_mlp_out[i], TD2);
        BUMP(bf16, saved_attn_proj[i], TD2);
    }
    for (int i = 0; i < NUM_ATTN_LAYERS; i++) {
        BUMP(bf16, saved_qkv[i], (size_t)T * 3 * H * HD * sizeof(bf16));
        BUMP(bf16, saved_attn_out[i], THD2);
        BUMP(bf16, saved_attn_gate[i], (size_t)T * H * sizeof(bf16));
    }
    for (int i = 0; i < 5; i++) {
        BUMP(bf16, saved_ve_gate[i], (size_t)T * H * sizeof(bf16));
    }

    // Gradient scratch
    BUMP(bf16, grad_x, TD2);
    BUMP(bf16, grad_lane0, TD2);
    BUMP(bf16, grad_lane1, TD2);
    BUMP(bf16, grad_x0, TD2);
    BUMP(bf16, grad_x0_bigram, TD2);

    // cuDNN flash attention stats (log-sum-exp) for backward
    for (int i = 0; i < NUM_ATTN_LAYERS; i++) {
        BUMP(float, attn_stats[i], (size_t)num_seqs_max_est * stats_per_seq * sizeof(float));
    }
    // cuDNN ragged offset scratch buffers
    BUMP(int32_t, cu_seqlens_paired, 4096 * sizeof(int32_t));  // ragged offsets
    BUMP(int32_t, seq_len_scratch, 4096 * sizeof(int32_t));    // per-seq lengths

    // YaRN tables
    BUMP(bf16, yarn_cos, 2 * (size_t)max_sl * HD * sizeof(bf16));
    BUMP(bf16, yarn_sin, 2 * (size_t)max_sl * HD * sizeof(bf16));
    BUMP(bf16, yarn_paired_cos, 2 * (size_t)max_sl * 2 * HD * sizeof(bf16));
    BUMP(bf16, yarn_paired_sin, 2 * (size_t)max_sl * 2 * HD * sizeof(bf16));

    // Scratch buffers
    BUMP(bf16, scratch1, scratch1_size);
    BUMP(bf16, scratch2, TD2);
    BUMP(float, scratch_f32, (size_t)T * D * sizeof(float));

    #undef BUMP

    size_t used = p - base;
    fprintf(stderr, "  activation arena: used %.1f GB / %.1f GB allocated\n", used / 1e9, total / 1e9);
    if (used > total) {
        fprintf(stderr, "FATAL: activation arena overflow: used %zu > total %zu\n", used, total);
        exit(1);
    }

    // Workspaces need separate allocations (different types/alignment requirements)
    act->attn_workspace_size = 256 * 1024 * 1024;  // 256 MB
    CUDA_CHECK(cudaMalloc(&act->attn_workspace, act->attn_workspace_size));

    act->cublas_workspace_size = 64 * 1024 * 1024;  // 64 MB
    CUDA_CHECK(cudaMalloc(&act->cublas_workspace, act->cublas_workspace_size));
}

// ============================================================================
// YarnState forward declaration (full implementation in Section F)
// ============================================================================

typedef struct {
    float angular_freq[HEAD_DIM];
    float attn_scale;
    int head_dim;
    int max_seq_len;
    int paired;
} YarnState;

static YarnState g_yarn;
static YarnState g_yarn_paired;

// ============================================================================
// Section D: Forward Pass
// ============================================================================

// Helper to read a scalar from device bf16
static float read_bf16_scalar(const bf16* ptr) {
    bf16 val;
    CUDA_CHECK(cudaMemcpy(&val, ptr, sizeof(bf16), cudaMemcpyDeviceToHost));
    return __bfloat162float(val);
}

// Attention forward pass for a single layer
// Implements CausalSelfAttention.forward() from Python
static void attention_forward(TrainingContext* ctx, int layer, int attn_idx,
                              bf16* x_in, bf16* out,
                              const bf16* qkvo_w,  // [4*D, D] row-major
                              float sa_lambda0, float sa_lambda1,
                              const bf16* attn_gate_w,  // [H, 12] or NULL
                              const bf16* ve_gate_w,    // [H, 12] or NULL
                              const bf16* ve,           // [T, H, HD] or NULL (already looked up + reshaped)
                              int ve_gate_idx,          // index into saved_ve_gate (-1 if no VE)
                              int32_t* cu_seqlens, int num_seqs, int num_tokens,
                              int max_seq_len, int window_size,
                              int is_paired, int key_offset_flag,
                              bf16* yarn_cos, bf16* yarn_sin,
                              float attn_scale,
                              int is_training) {
    Activations* act = &ctx->act;
    int T = num_tokens;
    int D = MODEL_DIM;
    int H = NUM_HEADS;
    int HD = HEAD_DIM;

    // Step 1: QKV projection with sa_lambda0 scaling
    // qkvo_w is [4*D, D]: rows 0..3D-1 are QKV, rows 3D..4D-1 are output proj
    // qkv = x @ (sa_lambda0 * qkvo_w[:3D]).T -> [T, 3D] = [T, 3H*HD]
    gemm_bf16_Bt(ctx->cublas_handle, x_in, qkvo_w, act->qkv,
                 T, 3 * D, D, sa_lambda0);

    // Step 2: Split QKV -> Q[T,H,HD], K[T,H,HD], V[T,H,HD]
    // Python: .view(B,T,3*H,HD).chunk(3,dim=-2)
    bf16* Q = act->qkv;                  // [T, H, HD]
    bf16* K = act->qkv + T * H * HD;     // [T, H, HD]
    bf16* V = act->qkv + 2 * T * H * HD; // [T, H, HD]

    // Step 3: QK RMS normalization (per head-vector)
    rms_norm_fwd(Q, Q, T * H, HD, ctx->stream);
    rms_norm_fwd(K, K, T * H, HD, ctx->stream);

    int attn_T = T;     // effective T for attention (doubled for paired)
    int attn_H = H;     // effective H for attention (halved for paired)

    if (!is_paired) {
        // Step 4a: RoPE (regular)
        rope_apply(Q, yarn_cos, yarn_sin, T, H, HD, ctx->stream);
        rope_apply(K, yarn_cos, yarn_sin, T, H, HD, ctx->stream);

        // Step 4b: Key offset (shift second half of key dims forward by 1)
        if (key_offset_flag && T > 1) {
            key_offset_shift(K, cu_seqlens, num_seqs, T, H, HD, ctx->stream);
        }

        // Step 4c: Value embedding gating
        if (ve != NULL && ve_gate_w != NULL) {
            // gate = 2 * sigmoid(cat(x[:6], ve[:6]) @ ve_gate_w.T) -> [T, H]
            // v = v + gate.unsqueeze(-1) * ve.view(T, H, HD)
            // ve is [T, D] (already gathered), use first 6 dims from x and ve
            sigmoid_gate_2src(act->scratch2, x_in, ve, ve_gate_w,
                        T, 6, H, D, D, 2.0f, ctx->stream);
            // Save VE gate values for backward
            if (is_training && ve_gate_idx >= 0) {
                CUDA_CHECK(cudaMemcpyAsync(act->saved_ve_gate[ve_gate_idx], act->scratch2,
                           T * H * sizeof(bf16), cudaMemcpyDeviceToDevice, ctx->stream));
            }
            // scratch2 is [T, H], broadcast to [T, H, HD]: gate_idx = (idx/HD) % (T*H) = t*H+h
            fused_gate_add(V, V, act->scratch2, ve,
                          T * H * HD, HD, T * H, ctx->stream);
        }
    } else {
        // Paired heads: reshape Q,K to [T, H/2, 2*HD], apply RoPE, reshape to [2T, H/2, HD]
        // V reshaped to [2T, H/2, HD] by interleaving
        attn_H = H / 2;
        attn_T = T * 2;

        // Apply RoPE on paired shape [T, H/2, 2*HD]
        rope_apply(Q, yarn_cos, yarn_sin, T, H / 2, 2 * HD, ctx->stream);
        rope_apply(K, yarn_cos, yarn_sin, T, H / 2, 2 * HD, ctx->stream);

        // After RoPE, reshape Q,K from [T, H/2, 2*HD] to [2T, H/2, HD]
        // This is a view change - the data layout is already correct:
        // [T, H/2, 2*HD] and [2T, H/2, HD] have the same memory layout
        // V is [T, H, HD] = [T, H/2, 2, HD], needs reshape to [2T, H/2, HD]
        // V's memory layout [T, H, HD] = [T * H/2 * 2, HD] = [2T * H/2, HD] is already correct

        // VE gating for paired path
        if (ve != NULL && ve_gate_w != NULL) {
            // gate = 2 * sigmoid(x[:12] @ ve_gate_w.T) -> [T, H]
            // Python: [B,T,H] then .view(B, 2T, H/2, 1) for paired V broadcast
            // In flat memory [T, H] = [2T, H/2], matching paired V layout [T, H, HD] = [2T, H/2, HD]
            sigmoid_gate(act->scratch2, x_in, ve_gate_w,
                        T, 12, H, D, 2.0f, ctx->stream);
            if (is_training && ve_gate_idx >= 0) {
                CUDA_CHECK(cudaMemcpyAsync(act->saved_ve_gate[ve_gate_idx], act->scratch2,
                           T * H * sizeof(bf16), cudaMemcpyDeviceToDevice, ctx->stream));
            }
            // Gate is [T, H] = [2T, H/2], data is [T, H, HD] = [2T, H/2, HD]
            fused_gate_add(V, V, act->scratch2, ve,
                          T * H * HD, HD, T * H, ctx->stream);
        }
    }

    // Save QKV for backward (after RMS norm, RoPE, key_offset, VE gating)
    if (is_training && attn_idx >= 0) {
        CUDA_CHECK(cudaMemcpyAsync(act->saved_qkv[attn_idx], act->qkv,
                   T * 3 * H * HD * sizeof(bf16), cudaMemcpyDeviceToDevice, ctx->stream));
    }

    // Step 5: cuDNN Flash Attention
    int s_max = is_paired ? 2 * max_seq_len : max_seq_len;

    // Compute element-offset ragged offsets for cuDNN
    // cuDNN ragged offsets are element offsets, not token counts
    // For both paired and non-paired: offset[i] = cu_seqlens[i] * H * HD
    // (paired: 2*cu_seqlens[i] * (H/2)*HD = cu_seqlens[i]*H*HD, same formula)
    int bucketed = bucket_num_seqs(num_seqs);
    // cu_seqlens is pre-padded on host: entries [num_seqs+1..bucketed] = T
    compute_ragged_offsets(act->cu_seqlens_paired, cu_seqlens,
                          NUM_HEADS * HD, bucketed + 1, ctx->stream);
    int32_t* ragged_offsets = act->cu_seqlens_paired;

    // Compute per-sequence lengths in attention view
    int seq_len_scale = is_paired ? 2 : 1;
    compute_seq_lens(act->seq_len_scratch, cu_seqlens, seq_len_scale, bucketed, ctx->stream);

    auto graph = get_or_build_graph(ctx, num_seqs, attn_H, s_max,
                                     window_size, HD, attn_scale, /*is_backward=*/false);

    execute_cudnn_sdpa_forward(ctx, graph,
        Q, K, V, act->attn_out,
        (is_training && attn_idx >= 0) ? act->attn_stats[attn_idx] : act->attn_stats[0],
        ragged_offsets, act->seq_len_scratch,
        bucketed, attn_H, s_max, HD);

    // Save per-layer metadata for backward
    if (is_training && attn_idx >= 0) {
        act->saved_is_paired[attn_idx] = is_paired;
    }

    // Step 6: Reshape output back to [T, H, HD] (for paired: [2T, H/2, HD] -> [T, H, HD])
    // The memory layouts are equivalent, so no actual reshape needed.
    // attn_out is [attn_T, attn_H, HD] which is [T*H*HD] elements total.

    // Save raw attention output (before gating) for backward
    if (is_training && attn_idx >= 0) {
        CUDA_CHECK(cudaMemcpyAsync(act->saved_attn_out[attn_idx], act->attn_out,
                   T * H * HD * sizeof(bf16), cudaMemcpyDeviceToDevice, ctx->stream));
    }

    // Step 7: Attention output gating
    // y = y * sigmoid(linear(x[:12], attn_gate_w)) where attn_gate_w is [H, 12]
    // sigmoid_gate computes gate values [T, H], then elementwise multiply with attn_out
    if (attn_gate_w != NULL) {
        sigmoid_gate(act->scratch2, x_in, attn_gate_w,
                    T, 12, H, D, 1.0f, ctx->stream);
        // Save gate values for backward
        if (is_training && attn_idx >= 0) {
            CUDA_CHECK(cudaMemcpyAsync(act->saved_attn_gate[attn_idx], act->scratch2,
                       T * H * sizeof(bf16), cudaMemcpyDeviceToDevice, ctx->stream));
        }
        // Multiply attn_out by gate: attn_out[t,h,:] *= gate[t*H+h]
        // gate is [T, H], data is [T, H, HD]: gate_idx = (idx/HD) % (T*H) = t*H+h
        elementwise_mul_broadcast(act->attn_out, act->attn_out, act->scratch2,
                                  T * H * HD, HD, T * H, ctx->stream);
    }

    // Step 8: Output projection with sa_lambda1 scaling
    // y is [T, D] (already contiguous after reshape)
    // out = y @ (sa_lambda1 * qkvo_w[3D:]).T -> [T, D]
    const bf16* out_w = qkvo_w + 3 * D * D;  // rows 3D..4D-1 of [4D, D]
    gemm_bf16_Bt(ctx->cublas_handle, act->attn_out, out_w, out,
                 T, D, D, sa_lambda1);
}

// MLP forward: relu(x @ c_fc.T)² @ c_proj.T
// Saves mlp_pre and mlp_post for backward pass
static void mlp_forward(TrainingContext* ctx, const bf16* x_in, bf16* out,
                        const bf16* c_fc, const bf16* c_proj,
                        int num_tokens, int layer) {
    Activations* act = &ctx->act;
    int T = num_tokens;

    // Step 1: pre = x @ c_fc.T via cuBLAS
    gemm_bf16_Bt(ctx->cublas_handle, x_in, c_fc, act->mlp_pre,
                 T, MLP_HDIM, MODEL_DIM);

    // Step 2: post = relu(pre)²
    relu_square_fwd(act->mlp_pre, act->mlp_post, T * MLP_HDIM, ctx->stream);

    // Save intermediates for backward
    CUDA_CHECK(cudaMemcpyAsync(act->saved_mlp_pre[layer], act->mlp_pre,
               T * MLP_HDIM * sizeof(bf16), cudaMemcpyDeviceToDevice, ctx->stream));
    CUDA_CHECK(cudaMemcpyAsync(act->saved_mlp_post[layer], act->mlp_post,
               T * MLP_HDIM * sizeof(bf16), cudaMemcpyDeviceToDevice, ctx->stream));

    // Step 3: out = post @ c_proj
    gemm_bf16(ctx->cublas_handle, act->mlp_post, c_proj, out,
              T, MODEL_DIM, MLP_HDIM);

    // Save mlp_out for scalar gradient computation in backward
    CUDA_CHECK(cudaMemcpyAsync(act->saved_mlp_out[layer], out,
               T * MODEL_DIM * sizeof(bf16), cudaMemcpyDeviceToDevice, ctx->stream));
}

// Attention backward: computes grad for output projection + QKV projection weights,
// runs naive attention backward for dQ/dK/dV, and returns grad_normed.
// The attention backward is O(T²) — for production, replace with cuDNN flash attn backward.
static void attention_backward(TrainingContext* ctx, int attn_idx,
                                bf16* grad_out,     // [T, D] incoming gradient
                                bf16* grad_normed,  // [T, D] output: gradient w.r.t. normed input
                                const bf16* qkvo_w, // [4*D, D] attn bank weights
                                float sa_lambda0, float sa_lambda1,
                                bf16* g_attn_bank,  // [4*D, D] gradient accumulator
                                const bf16* saved_normed, // [T, D] saved input to attention
                                int32_t* cu_seqlens, int num_seqs,
                                int num_tokens, int window_size,
                                int max_seq_len, int is_paired,
                                float attn_scale,
                                bf16* yarn_cos, bf16* yarn_sin,
                                const bf16* attn_gate_w,  // [H, 12] gate weights (NULL if no gate)
                                bf16* g_attn_gate_w,      // [H, 12] gate gradient accumulator
                                float* gate_grad_scratch,  // [T*H] scratch for gate sigmoid grad
                                // VE backward parameters (all NULL/0 if no VE for this layer)
                                const bf16* ve_gate_w,     // [H, 12] VE gate weights
                                bf16* g_ve_gate_w,         // VE gate weight gradient accumulator
                                bf16* g_value_embeds,      // [VOCAB_SIZE, D] VE gradient (for scatter-add)
                                const int32_t* inputs,     // [T] input token IDs (for VE gather/scatter)
                                int ve_gate_idx,           // index into saved_ve_gate (-1 if no VE)
                                int ve_idx,                // which value_embed table
                                const bf16* value_embeds,  // full value_embeds pointer
                                int layer_idx,             // layer index for sa_lambda scalar grad
                                float* scalar_grad_acc)    // scalar gradient accumulators
{
    Activations* act = &ctx->act;
    int T = num_tokens;
    int D = MODEL_DIM;
    int H = NUM_HEADS;
    int HD = HEAD_DIM;

    const bf16* out_w = qkvo_w + 3 * D * D;

    // Step 1: Reconstruct gated attention output for weight gradient
    // Forward: gated_out = attn_out * gate, then out = gated_out @ (sa_lambda1 * out_w).T
    // We saved attn_out (pre-gating) and gate values; reconstruct gated version
    bf16* gated_out = act->mlp_pre;  // scratch [T, MLP_HDIM] >= [T, D]
    CUDA_CHECK(cudaMemcpyAsync(gated_out, act->saved_attn_out[attn_idx],
               T * D * sizeof(bf16), cudaMemcpyDeviceToDevice, ctx->stream));
    elementwise_mul_broadcast(gated_out, gated_out, act->saved_attn_gate[attn_idx],
                              T * H * HD, HD, T * H, ctx->stream);

    // Output projection weight gradient: g_out_w += sa_lambda1 * gated_out.T @ grad_out
    bf16* g_out_w = g_attn_bank + 3 * D * D;
    gemm_bf16_At(ctx->cublas_handle, gated_out, grad_out, g_out_w,
                D, D, T, sa_lambda1, 1.0f);

    // Step 2: Compute gradient w.r.t. gated output
    // grad_gated = grad_out @ out_w * sa_lambda1
    bf16* grad_gated = act->attn_out;  // reuse as scratch [T, D]
    gemm_bf16(ctx->cublas_handle, grad_out, out_w, grad_gated, T, D, D);

    // sa_lambda1 scalar gradient: dot(unscaled_grad_gated, gated_out)
    bf16_dot_product(scalar_grad_acc + 89 + 2 * layer_idx + 1,
                     grad_gated, gated_out, T * D, ctx->stream);

    scale_tensor(grad_gated, sa_lambda1, T * D, ctx->stream);

    // Step 2b: Gate weight gradient (before masking grad_gated by gate)
    // grad_gate = sum(grad_gated * saved_attn_out, HD_dim) * sig * (1-sig)
    if (attn_gate_w != NULL) {
        gate_sigmoid_grad(gate_grad_scratch, grad_gated,
                          act->saved_attn_out[attn_idx],
                          act->saved_attn_gate[attn_idx],
                          T, H, HD, 1.0f, ctx->stream);
        gate_weight_grad(g_attn_gate_w, gate_grad_scratch,
                         saved_normed, T, H, D, 12, ctx->stream);
    }

    // Step 3: Gate backward - multiply gradient by gate values
    // grad_raw_attn = grad_gated * gate (broadcast gate [T,H] over HD)
    elementwise_mul_broadcast(grad_gated, grad_gated, act->saved_attn_gate[attn_idx],
                              T * H * HD, HD, T * H, ctx->stream);

    // Step 4: Attention backward — compute dQ, dK, dV via cuDNN
    int attn_H = is_paired ? H / 2 : H;
    int s_max = is_paired ? 2 * max_seq_len : max_seq_len;

    const bf16* saved_Q = act->saved_qkv[attn_idx];
    const bf16* saved_K = act->saved_qkv[attn_idx] + T * H * HD;
    const bf16* saved_V = act->saved_qkv[attn_idx] + 2 * T * H * HD;

    // dQ, dK, dV -> reuse q, k, v buffers as scratch
    bf16* dQ = act->q;   // [T, H, HD]
    bf16* dK = act->k;   // [T, H, HD]
    bf16* dV = act->v;   // [T, H, HD]

    // Compute element-offset ragged offsets for cuDNN
    int bucketed = bucket_num_seqs(num_seqs);
    compute_ragged_offsets(act->cu_seqlens_paired, cu_seqlens,
                          NUM_HEADS * HD, bucketed + 1, ctx->stream);
    int32_t* ragged_offsets = act->cu_seqlens_paired;

    // Per-sequence lengths in attention view
    int seq_len_scale = is_paired ? 2 : 1;
    compute_seq_lens(act->seq_len_scratch, cu_seqlens, seq_len_scale, bucketed, ctx->stream);

    auto graph = get_or_build_graph(ctx, num_seqs, attn_H, s_max,
                                     window_size, HD, attn_scale, /*is_backward=*/true);

    execute_cudnn_sdpa_backward(ctx, graph,
        dQ, dK, dV,
        grad_gated,                          // dO
        (bf16*)saved_Q, (bf16*)saved_K, (bf16*)saved_V,
        (bf16*)act->saved_attn_out[attn_idx], // O (saved attention output)
        act->attn_stats[attn_idx],            // stats from forward
        ragged_offsets, act->seq_len_scratch,
        bucketed, attn_H, s_max, HD);

    // Step 3b: VE gate backward (value embeddings gating: V += gate * ve)
    // Step 3b: VE gate backward (value embeddings gating: V += gate * ve)
    // dV from cuDNN is gradient w.r.t. modified V. Propagate through VE gating.
    // Uses gate_grad_scratch[T*H..] to avoid conflict with attn gate grad at [0..T*H-1]
    if (ve_gate_idx >= 0 && ve_gate_w != NULL) {
        float* ve_gate_grad = gate_grad_scratch + T * H;

        // Re-gather VE data into scratch (act->mlp_pre is free after step 1)
        bf16* scratch_ve = act->mlp_pre;  // [T, D]
        gather_embed(scratch_ve, value_embeds + (size_t)ve_idx * VOCAB_SIZE * D,
                     inputs, T, D, ctx->stream);

        // VE gate sigmoid grad: d_z = sum(dV * ve, HD_dim) * scale * sig * (1-sig)
        // For both paired and non-paired: gate is [T, H], dV and ve are [T, H, HD]
        gate_sigmoid_grad(ve_gate_grad, dV, scratch_ve,
                          act->saved_ve_gate[ve_gate_idx],
                          T, H, HD, 2.0f, ctx->stream);

        if (!is_paired) {
            // Non-paired: gate input is cat(x[:6], ve[:6]), weight is [H, 12]
            gate_weight_grad_2src(g_ve_gate_w, ve_gate_grad,
                                  saved_normed, scratch_ve,
                                  T, H, D, D, 6, ctx->stream);
        } else {
            // Paired: gate input is x[:12] only, weight is [H, 12]
            gate_weight_grad(g_ve_gate_w, ve_gate_grad,
                             saved_normed, T, H, D, 12, ctx->stream);
        }

        // d_ve = gate * dV (gradient flowing to value embeddings through multiplication)
        // Use act->mlp_post as scratch for d_ve [T, D]
        bf16* d_ve = act->mlp_post;
        elementwise_mul_broadcast(d_ve, dV, act->saved_ve_gate[ve_gate_idx],
                                  T * H * HD, HD, T * H, ctx->stream);

        // Scatter-add d_ve to g_value_embeds
        scatter_add_embed(g_value_embeds + (size_t)ve_idx * VOCAB_SIZE * D,
                          d_ve, inputs, T, D, ctx->stream);
    }

    // Step 4: Backward through RoPE (apply transpose of rotation matrix)
    // dQ and dK are in attention layout:
    //   non-paired: [T, H, HD]
    //   paired: [2T, H/2, HD] = [T, H/2, 2*HD] in memory (same layout)
    if (!is_paired) {
        rope_backward(dQ, dQ, yarn_cos, yarn_sin, T, H, HD, ctx->stream);
        rope_backward(dK, dK, yarn_cos, yarn_sin, T, H, HD, ctx->stream);
    } else {
        // Paired: data is [T, H/2, 2*HD] — same memory as [2T, H/2, HD]
        rope_backward(dQ, dQ, yarn_cos, yarn_sin, T, H / 2, 2 * HD, ctx->stream);
        rope_backward(dK, dK, yarn_cos, yarn_sin, T, H / 2, 2 * HD, ctx->stream);
    }

    // Step 5: Backward through QK RMS normalization
    // Need pre-norm Q,K — recompute by re-doing QKV projection
    // qkv_raw = saved_normed @ (sa_lambda0 * qkvo_w[:3D]).T
    gemm_bf16_Bt(ctx->cublas_handle, saved_normed, qkvo_w, act->qkv,
                 T, 3 * D, D, sa_lambda0);
    // act->qkv now has pre-norm Q and K (and V which doesn't need norm backward)
    bf16* Q_pre = act->qkv;                  // [T, H, HD] pre-norm
    bf16* K_pre = act->qkv + T * H * HD;     // [T, H, HD] pre-norm

    // rms_norm_bwd: dQ_raw = rms_norm_bwd(dQ_post_rope, Q_pre)
    rms_norm_bwd(dQ, dQ, Q_pre, T * H, HD, ctx->stream);
    rms_norm_bwd(dK, dK, K_pre, T * H, HD, ctx->stream);

    // dV passes through unchanged (no norm/RoPE on V)

    // Concatenate dQ, dK, dV back into qkv layout [T, 3*H*HD]
    CUDA_CHECK(cudaMemcpyAsync(act->qkv, dQ, T * H * HD * sizeof(bf16),
               cudaMemcpyDeviceToDevice, ctx->stream));
    CUDA_CHECK(cudaMemcpyAsync(act->qkv + T * H * HD, dK, T * H * HD * sizeof(bf16),
               cudaMemcpyDeviceToDevice, ctx->stream));
    CUDA_CHECK(cudaMemcpyAsync(act->qkv + 2 * T * H * HD, dV, T * H * HD * sizeof(bf16),
               cudaMemcpyDeviceToDevice, ctx->stream));

    // Step 6: QKV projection backward
    // Forward: qkv = x_normed @ (sa_lambda0 * qkvo_w[:3D]).T
    // grad_qkvo[:3D] += sa_lambda0 * grad_qkv.T @ x_normed  (= [3D,T] @ [T,D] = [3D,D])
    gemm_bf16_At(ctx->cublas_handle, act->qkv, saved_normed, g_attn_bank,
                3 * D, D, T, sa_lambda0, 1.0f);

    // grad_normed = grad_qkv @ qkvo_w[:3D] * sa_lambda0
    gemm_bf16(ctx->cublas_handle, act->qkv, qkvo_w, grad_normed,
              T, D, 3 * D);

    // sa_lambda0 scalar gradient: dot(unscaled_grad_normed, saved_normed)
    bf16_dot_product(scalar_grad_acc + 89 + 2 * layer_idx,
                     grad_normed, saved_normed, T * D, ctx->stream);

    scale_tensor(grad_normed, sa_lambda0, T * D, ctx->stream);

    // Step 7: Gate input gradient — add gate's contribution to grad_normed[:, :12]
    // attn_gate_grad is at gate_grad_scratch[0..T*H-1] (set in step 2b)
    if (attn_gate_w != NULL) {
        gate_input_grad(grad_normed, gate_grad_scratch, attn_gate_w,
                        T, H, D, 12, ctx->stream);
    }

    // Step 7b: VE gate input gradient (deferred from step 3b)
    // ve_gate_grad is at gate_grad_scratch[T*H..] (set in step 3b)
    if (ve_gate_idx >= 0 && ve_gate_w != NULL) {
        float* ve_gate_grad = gate_grad_scratch + T * H;
        if (!is_paired) {
            // Non-paired: gate input was cat(x[:6], ve[:6])
            // Add grad to grad_normed[:,:6] and a throwaway buffer for ve[:,:6]
            gate_input_grad_2src(grad_normed, act->mlp_pre, ve_gate_grad,
                                  ve_gate_w, T, H, D, D, 6, ctx->stream);
        } else {
            // Paired: gate input was x[:12] only
            gate_input_grad(grad_normed, ve_gate_grad, ve_gate_w,
                            T, H, D, 12, ctx->stream);
        }
    }
}

void forward_pass(TrainingContext* ctx, int32_t* inputs, int64_t* targets,
                  int32_t* cum_seqlens, int32_t* bigram_inputs,
                  int num_tokens, int num_seqs, const CachedScalars* cs,
                  float* mtp_weights, int n_mtp,
                  int ws_short, int ws_long, int train_max_seq_len,
                  int is_training, float* loss_out) {
    ModelParams* p = &ctx->params;
    Activations* act = &ctx->act;
    int T = num_tokens;
    int D = MODEL_DIM;

    // ---- Embeddings ----
    // x = embed(input_seq)
    gather_embed(act->x, p->embed, inputs, T, D, ctx->stream);

    // x0_bigram = bigram_embed(bigram_inputs)
    gather_embed(act->x0_bigram, p->bigram_embed, bigram_inputs, T, D, ctx->stream);

    // Smear token embedding forward 1 position
    float smear_lambda = __bfloat162float(cs->scalars[2 * NUM_LAYERS]);
    smear_forward(act->x, act->x, p->smear_gate, smear_lambda, T, D, ctx->stream);

    // x0 = norm(x) -- per-token RMS normalization
    rms_norm_fwd(act->x0, act->x, T, D, ctx->stream);

    // lane0 = x0 + bigram_lambdas[0] * x0_bigram
    float bigram_lambda0 = __bfloat162float(cs->bigram_lambdas[0]);
    fused_add_scale(act->lane0, act->x0, act->x0_bigram, 1.0f, bigram_lambda0, T * D, ctx->stream);

    // Use cached scalars (no D2H copies needed)
    const bf16* h_scalars = cs->scalars;
    float backout_lambda = __bfloat162float(h_scalars[2 * NUM_LAYERS + 1]);
    float skip_lambda_raw = __bfloat162float(h_scalars[2 * NUM_LAYERS + 2]);

    const bf16* h_resid = cs->resid_lambdas;
    const bf16* h_post_lambdas = cs->post_lambdas;
    const bf16* h_x0_lambdas = cs->x0_lambdas;
    const bf16* h_bigram_lambdas = cs->bigram_lambdas;

    // Window sizes per layer (in tokens)
    int bm_sizes[NUM_LAYERS] = {ws_short, ws_short, ws_short, ws_long, ws_short, ws_short, 0, ws_short, ws_short, ws_short, ws_long};
    int key_offsets[NUM_LAYERS];
    for (int i = 0; i < NUM_LAYERS; i++) key_offsets[i] = (bm_sizes[i] == ws_long) ? 1 : 0;
    int paired_layers[] = {0, 2, 5, 9};
    int is_paired_layer[NUM_LAYERS] = {};
    for (int i = 0; i < 4; i++) is_paired_layer[paired_layers[i]] = 1;

    // Attention gate mapping: layers 0-5 use ag[0-5], layer 6 has none, layers 7-10 use ag[6-9]
    // VE gate mapping: layer 1->veg[0], layer 2->veg[1], layers 8->veg[2], 9->veg[3], 10->veg[4]
    int attn_gate_map[NUM_LAYERS] = {0, 1, 2, 3, 4, 5, -1, 6, 7, 8, 9};
    int ve_gate_map[NUM_LAYERS] = {-1, 0, 1, -1, -1, -1, -1, -1, 2, 3, 4};
    int ve_embed_map[NUM_LAYERS]; // which of the 5 value embeds to use
    // Value embed usage: layers that have ve gates use ve_embed[ve_gate_map[layer]]
    // ve_gate_map gives the index into ve_gate_bank AND value_embeds

    // num_seqs passed in from caller (construct_aligned_batch)
    act->saved_num_seqs = num_seqs;

    // Get YaRN attention scale
    float attn_scale_val = g_yarn.attn_scale;

    // ---- Transformer layers ----
    int attn_bank_idx = 0;
    for (int i = 0; i < NUM_LAYERS; i++) {
        float resid_attn = __bfloat162float(h_resid[i * 2]);
        float resid_mlp = __bfloat162float(h_resid[i * 2 + 1]);
        float pl_attn_ln0 = __bfloat162float(h_post_lambdas[i * 4 + 0]);
        float pl_attn_ln1 = __bfloat162float(h_post_lambdas[i * 4 + 1]);
        float pl_mlp_ln0 = __bfloat162float(h_post_lambdas[i * 4 + 2]);
        float pl_mlp_ln1 = __bfloat162float(h_post_lambdas[i * 4 + 3]);
        float x0_lambda = __bfloat162float(h_x0_lambdas[i]);
        float bigram_lambda = __bfloat162float(h_bigram_lambdas[i]);
        float sa_lambda0 = __bfloat162float(h_scalars[2 * i]);
        float sa_lambda1 = __bfloat162float(h_scalars[2 * i + 1]);

        // Save lane0 for backward
        CUDA_CHECK(cudaMemcpyAsync(act->saved_lane0[i], act->lane0, T * D * sizeof(bf16), cudaMemcpyDeviceToDevice, ctx->stream));

        // Introduce lane1 at parallel_start
        if (i == PARALLEL_START) {
            CUDA_CHECK(cudaMemcpyAsync(act->lane1, act->lane0, T * D * sizeof(bf16), cudaMemcpyDeviceToDevice, ctx->stream));
        }
        if (i >= PARALLEL_START) {
            CUDA_CHECK(cudaMemcpyAsync(act->saved_lane1[i], act->lane1, T * D * sizeof(bf16), cudaMemcpyDeviceToDevice, ctx->stream));
        }

        // Skip connection injection at layer 6
        if (i == SKIP_OUT_LAYER) {
            // skip_out = sigmoid(skip_lambda_raw) * 2 * sigmoid(skip_gate(x0[:, :12]))
            // Compute sigmoid(skip_gate(x0[:,:12])) -> [T, 1]
            // Then scale by sigmoid(skip_lambda_raw) * 2
            float skip_scale = 2.0f / (1.0f + expf(-skip_lambda_raw));
            sigmoid_gate(act->scratch2, act->x0, p->skip_gate,
                        T, 12, 1, D, skip_scale, ctx->stream);
            // lane0 += skip_gate_out * skip_save (gate [T,1], data [T,D])
            // gate_idx = (idx/D) % T = t
            fused_gate_add(act->lane0, act->lane0, act->scratch2, act->skip_save,
                          T * D, D, T, ctx->stream);
            // Re-save lane0 after skip injection so backward has the correct input
            // (saved_lane0[6] must reflect lane0 AFTER skip injection for RMS norm backward)
            CUDA_CHECK(cudaMemcpyAsync(act->saved_lane0[i], act->lane0, T * D * sizeof(bf16), cudaMemcpyDeviceToDevice, ctx->stream));
        }

        // Get weight pointers
        const bf16* qkvo_w = (i != ATTN_SKIP_LAYER) ? p->attn_bank + attn_bank_idx * 4 * D * D : NULL;
        const bf16* c_fc = p->mlp_bank + i * 2 * MLP_HDIM * D;
        const bf16* c_proj = p->mlp_bank + i * 2 * MLP_HDIM * D + MLP_HDIM * D;

        // Gate weight pointers
        const bf16* attn_gw = (attn_gate_map[i] >= 0) ?
            p->attn_gate_bank + attn_gate_map[i] * NUM_HEADS * 12 : NULL;
        const bf16* ve_gw = (ve_gate_map[i] >= 0) ?
            p->ve_gate_bank + ve_gate_map[i] * NUM_HEADS * 12 : NULL;
        // VE: look up value_embeds for inputs and reshape
        // value_embeds[ve_gate_map[i]] gives [VOCAB_SIZE, D]
        // ve = value_embeds[ve_idx][inputs] -> [T, D]
        // This requires an embedding lookup per layer that has VE
        const bf16* ve_ptr = NULL;
        if (ve_gate_map[i] >= 0) {
            int ve_idx = ve_gate_map[i];
            // Look up value embeds: ve = value_embeds[ve_idx * VOCAB_SIZE * D + inputs[t] * D]
            gather_embed(act->scratch1, p->value_embeds + ve_idx * VOCAB_SIZE * D,
                        inputs, T, D, ctx->stream);
            ve_ptr = act->scratch1;
        }

        bf16* y_cos = is_paired_layer[i] ? act->yarn_paired_cos : act->yarn_cos;
        bf16* y_sin = is_paired_layer[i] ? act->yarn_paired_sin : act->yarn_sin;

        if (i == ATTN_SKIP_LAYER) {
            // MLP-only layer (no attention)
            rms_norm_fwd(act->normed, act->lane0, T, D, ctx->stream);
            CUDA_CHECK(cudaMemcpyAsync(act->saved_normed[i], act->normed, T * D * sizeof(bf16), cudaMemcpyDeviceToDevice, ctx->stream));
            // For ATTN_SKIP_LAYER, MLP norm = attention norm (same input)
            CUDA_CHECK(cudaMemcpyAsync(act->saved_normed_mlp[i], act->normed, T * D * sizeof(bf16), cudaMemcpyDeviceToDevice, ctx->stream));

            mlp_forward(ctx, act->normed, act->mlp_out, c_fc, c_proj, T, i);

            fused_add_scale(act->lane0, act->lane0, act->mlp_out, resid_mlp, pl_mlp_ln0, T * D, ctx->stream);
            CUDA_CHECK(cudaMemcpyAsync(act->saved_post_attn[i], act->lane0, T * D * sizeof(bf16), cudaMemcpyDeviceToDevice, ctx->stream));
        } else if (i < PARALLEL_START) {
            // Single-stream layer
            rms_norm_fwd(act->normed, act->lane0, T, D, ctx->stream);
            CUDA_CHECK(cudaMemcpyAsync(act->saved_normed[i], act->normed, T * D * sizeof(bf16), cudaMemcpyDeviceToDevice, ctx->stream));

            attention_forward(ctx, i, attn_bank_idx, act->normed, act->scratch1,
                            qkvo_w, sa_lambda0, sa_lambda1,
                            attn_gw, ve_gw, ve_ptr, ve_gate_map[i],
                            cum_seqlens, num_seqs, T, train_max_seq_len,
                            bm_sizes[i], is_paired_layer[i], key_offsets[i],
                            y_cos, y_sin, attn_scale_val, is_training);

            // lane0 = resid_attn * lane0 + attn_out + x0_lambda * x0
            fused_add3(act->lane0, act->lane0, act->scratch1, act->x0,
                       resid_attn, 1.0f, x0_lambda, T * D, ctx->stream);
            if (i > 0) {
                fused_add_scale(act->lane0, act->lane0, act->x0_bigram, 1.0f, bigram_lambda, T * D, ctx->stream);
            }

            CUDA_CHECK(cudaMemcpyAsync(act->saved_post_attn[i], act->lane0, T * D * sizeof(bf16), cudaMemcpyDeviceToDevice, ctx->stream));

            // MLP
            rms_norm_fwd(act->normed, act->lane0, T, D, ctx->stream);
            if (is_training) {
                CUDA_CHECK(cudaMemcpyAsync(act->saved_normed_mlp[i], act->normed, T * D * sizeof(bf16), cudaMemcpyDeviceToDevice, ctx->stream));
            }
            mlp_forward(ctx, act->normed, act->mlp_out, c_fc, c_proj, T, i);
            fused_add_scale(act->lane0, act->lane0, act->mlp_out, resid_mlp, pl_mlp_ln0, T * D, ctx->stream);

            attn_bank_idx++;
        } else {
            // Parallel layers (7-10): attn reads lane0, MLP reads lane1
            rms_norm_fwd(act->normed, act->lane0, T, D, ctx->stream);
            CUDA_CHECK(cudaMemcpyAsync(act->saved_normed[i], act->normed, T * D * sizeof(bf16), cudaMemcpyDeviceToDevice, ctx->stream));

            attention_forward(ctx, i, attn_bank_idx, act->normed, act->scratch1,
                            qkvo_w, sa_lambda0, sa_lambda1,
                            attn_gw, ve_gw, ve_ptr, ve_gate_map[i],
                            cum_seqlens, num_seqs, T, train_max_seq_len,
                            bm_sizes[i], is_paired_layer[i], key_offsets[i],
                            y_cos, y_sin, attn_scale_val, is_training);

            // Save attention projection output for scalar gradient computation
            if (is_training) {
                CUDA_CHECK(cudaMemcpyAsync(act->saved_attn_proj[i], act->scratch1, T * D * sizeof(bf16), cudaMemcpyDeviceToDevice, ctx->stream));
            }

            // Update lane0: resid_attn * lane0 + pl_attn_ln0 * attn_out + x0_inject
            fused_add3(act->lane0, act->lane0, act->scratch1, act->x0,
                       resid_attn, pl_attn_ln0, x0_lambda, T * D, ctx->stream);
            if (i > 0) {
                fused_add_scale(act->lane0, act->lane0, act->x0_bigram, 1.0f, bigram_lambda, T * D, ctx->stream);
            }

            // Update lane1: resid_attn * lane1 + pl_attn_ln1 * attn_out
            fused_add_scale(act->lane1, act->lane1, act->scratch1, resid_attn, pl_attn_ln1, T * D, ctx->stream);

            CUDA_CHECK(cudaMemcpyAsync(act->saved_post_attn[i], act->lane0, T * D * sizeof(bf16), cudaMemcpyDeviceToDevice, ctx->stream));

            // Save lane1 after attn update (needed for MLP backward RMS norm)
            if (is_training) {
                CUDA_CHECK(cudaMemcpyAsync(act->saved_lane1_post_attn[i], act->lane1, T * D * sizeof(bf16), cudaMemcpyDeviceToDevice, ctx->stream));
            }

            // MLP on lane1
            rms_norm_fwd(act->scratch2, act->lane1, T, D, ctx->stream);
            if (is_training) {
                CUDA_CHECK(cudaMemcpyAsync(act->saved_normed_mlp[i], act->scratch2, T * D * sizeof(bf16), cudaMemcpyDeviceToDevice, ctx->stream));
            }
            mlp_forward(ctx, act->scratch2, act->mlp_out, c_fc, c_proj, T, i);

            // Update both lanes with MLP output
            fused_add_scale(act->lane0, act->lane0, act->mlp_out, resid_mlp, pl_mlp_ln0, T * D, ctx->stream);
            fused_add_scale(act->lane1, act->lane1, act->mlp_out, resid_mlp, pl_mlp_ln1, T * D, ctx->stream);

            attn_bank_idx++;
        }

        // Skip connection save at layer 3
        if (i == SKIP_IN_LAYER) {
            CUDA_CHECK(cudaMemcpyAsync(act->skip_save, act->saved_post_attn[i], T * D * sizeof(bf16), cudaMemcpyDeviceToDevice, ctx->stream));
        }

        // Backout save at layer 7
        if (i == BACKOUT_LAYER) {
            CUDA_CHECK(cudaMemcpyAsync(act->x_backout, act->lane0, T * D * sizeof(bf16), cudaMemcpyDeviceToDevice, ctx->stream));
        }
    }

    // ---- Output and loss ----
    // x = (lane0 + lane1) * 0.5
    fused_add_scale(act->x, act->lane0, act->lane1, 0.5f, 0.5f, T * D, ctx->stream);

    // Subtract backout
    fused_add_scale(act->x, act->x, act->x_backout, 1.0f, -backout_lambda, T * D, ctx->stream);

    // Final RMS norm
    rms_norm_fwd(act->x, act->x, T, D, ctx->stream);

    if (is_training) {
        // TODO: Use FP8 matmul once cuBLAS FP8 support is verified on this platform
        // For now, use BF16 lm_head matmul (same as eval mode)
        gemm_bf16(ctx->cublas_handle, act->x, p->lm_head, act->logits, T, VOCAB_SIZE, D);

        // Copy MTP weights to scratch (reuse scratch_f32 to avoid alloc in hot path)
        CUDA_CHECK(cudaMemcpyAsync(act->scratch_f32, mtp_weights, n_mtp * sizeof(float),
                                    cudaMemcpyHostToDevice, ctx->stream));

        // Softcapped CE forward
        softcapped_ce_fwd(act->logits, act->losses, act->lse,
                          targets, act->scratch_f32,
                          T, VOCAB_SIZE, n_mtp,
                          SOFTCAP_A, SOFTCAP_B, SOFTCAP_C, ctx->stream);

        // Sum losses on GPU, copy to host if loss_out is non-null
        if (loss_out) {
            gpu_reduce_sum(act->scratch_f32, act->losses, T, ctx->stream);
            CUDA_CHECK(cudaMemcpy(loss_out, act->scratch_f32, sizeof(float), cudaMemcpyDeviceToHost));
            *loss_out /= T;
        }
    } else {
        // Eval mode: BF16 lm_head matmul + softcapped CE
        // lm_head is [D, VOCAB_SIZE] (transposed), so x[T,D] @ lm_head[D,V] = logits[T,V]
        gemm_bf16(ctx->cublas_handle, act->x, p->lm_head, act->logits, T, VOCAB_SIZE, D);

        // Use softcapped CE with weight=1.0 for single-token prediction
        float eval_weight = 1.0f;
        CUDA_CHECK(cudaMemcpyAsync(act->scratch_f32, &eval_weight, sizeof(float),
                                    cudaMemcpyHostToDevice, ctx->stream));

        softcapped_ce_fwd(act->logits, act->losses, act->lse,
                          targets, act->scratch_f32,
                          T, VOCAB_SIZE, 1,
                          SOFTCAP_A, SOFTCAP_B, SOFTCAP_C, ctx->stream);

        gpu_reduce_sum(act->scratch_f32, act->losses, T, ctx->stream);
        CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
        CUDA_CHECK(cudaMemcpyAsync(loss_out, act->scratch_f32, sizeof(float), cudaMemcpyDeviceToHost, ctx->stream));
        CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
        *loss_out /= T;
    }
}

// ============================================================================
// Section E: Backward Pass (stub - to be fully implemented)
// ============================================================================

// Global counters for backward section timing (accumulated across microsteps)
static float g_bwd_ce_ms = 0, g_bwd_lmhead_ms = 0, g_bwd_layers_ms = 0, g_bwd_embed_ms = 0, g_bwd_scalgrad_ms = 0;
static float g_bwd_attn_ms = 0, g_bwd_mlp_ms = 0, g_bwd_elemwise_ms = 0;
static int g_bwd_profile_step = -1;

void backward_pass(TrainingContext* ctx, int32_t* inputs, int64_t* targets,
                   int32_t* cum_seqlens, int32_t* bigram_inputs,
                   int num_tokens, int num_seqs, const CachedScalars* cs,
                   float* mtp_weights, int n_mtp,
                   int ws_short, int ws_long, int train_max_seq_len) {
    // Manual backward pass - reverse order of forward
    // This is a large implementation that mirrors forward_pass in reverse

    int T = num_tokens;
    int D = MODEL_DIM;
    Activations* act = &ctx->act;
    ModelParams* p = &ctx->params;
    ModelGrads* g = &ctx->grads;

    // Section timing disabled (was adding cudaEventSynchronize pipeline stalls)
    #define BWD_TIMER_START() do {} while(0)
    #define BWD_TIMER_END(var) do {} while(0)

    BWD_TIMER_START();

    // 1. Softcapped CE backward -> BF16 gradients
    // Scale by 1/GRAD_ACCUM_STEPS to match Python: loss = losses.sum() * grad_scale
    // where grad_scale = 1/grad_accum_steps. Each token's upstream gradient = grad_scale.
    fill_constant_f32(act->scratch_f32, 1.0f / GRAD_ACCUM_STEPS, T, ctx->stream);
    CUDA_CHECK(cudaMemcpyAsync(act->scratch_f32 + T, mtp_weights, n_mtp * sizeof(float),
                                cudaMemcpyHostToDevice, ctx->stream));

    // Use logits buffer as BF16 grad output (logits no longer needed after CE backward)
    bf16* grad_logits_bf16 = act->logits;
    softcapped_ce_bwd_bf16(grad_logits_bf16, act->scratch_f32, act->lse,
                           act->logits, targets, act->scratch_f32 + T,
                           T, VOCAB_SIZE, n_mtp,
                           SOFTCAP_A, SOFTCAP_B, SOFTCAP_C, 1.0f,
                           ctx->stream);

    BWD_TIMER_END(g_bwd_ce_ms);
    BWD_TIMER_START();

    // 2. lm_head backward via BF16 matmul
    // grad_x = grad_logits @ lm_head.T  (= [T,V] @ [V,D] = [T,D])
    // lm_head is stored as [D,V], so grad_logits[T,V] @ lm_head.T[V,D] = gemm_bf16_Bt
    gemm_bf16_Bt(ctx->cublas_handle, grad_logits_bf16, p->lm_head,
                 act->grad_x, T, D, VOCAB_SIZE);

    // grad_w = grad_logits.T @ x  (= [V,T] @ [T,D] = [V,D])
    // But lm_head is stored as [D,V], so we need [D,V] += x.T @ grad_logits
    gemm_bf16_At(ctx->cublas_handle, act->x, grad_logits_bf16, g->lm_head,
                 D, VOCAB_SIZE, T, 1.0f, 1.0f);  // beta=1: accumulate

    // Read scalars early (needed for pre-norm recomputation in step 3)
    const bf16* h_scalars = cs->scalars;
    float backout_lambda = __bfloat162float(h_scalars[2 * NUM_LAYERS + 1]);

    // 3. Final RMS norm backward
    // act->x was overwritten by in-place rms_norm_fwd in the forward pass.
    // Recompute the pre-norm value: pre_norm = 0.5*(lane0+lane1) - backout_lambda*x_backout
    // lane0/lane1 still contain final forward values (backward uses grad_lane0/grad_lane1).
    {
        bf16* pre_norm_final = act->scratch1;
        fused_add_scale(pre_norm_final, act->lane0, act->lane1, 0.5f, 0.5f, T * D, ctx->stream);
        fused_add_scale(pre_norm_final, pre_norm_final, act->x_backout, 1.0f, -backout_lambda, T * D, ctx->stream);
        rms_norm_bwd(act->grad_x, act->grad_x, pre_norm_final, T, D, ctx->stream);
    }

    // 4. Backout/lane-merge backward
    // Forward: x = 0.5*(lane0+lane1) - backout_lambda*x_backout
    // Backward: grad_lane0 += 0.5*grad_x, grad_lane1 += 0.5*grad_x
    // grad_x_backout -= backout_lambda * grad_x
    CUDA_CHECK(cudaMemcpyAsync(act->grad_lane0, act->grad_x, T * D * sizeof(bf16), cudaMemcpyDeviceToDevice, ctx->stream));
    CUDA_CHECK(cudaMemcpyAsync(act->grad_lane1, act->grad_x, T * D * sizeof(bf16), cudaMemcpyDeviceToDevice, ctx->stream));
    scale_tensor(act->grad_lane0, 0.5f, T * D, ctx->stream);
    scale_tensor(act->grad_lane1, 0.5f, T * D, ctx->stream);

    // Per-layer window sizes and attention scale (mirrors forward_pass setup)
    int bm_sizes[NUM_LAYERS] = {ws_short, ws_short, ws_short, ws_long, ws_short, ws_short, 0,
                                 ws_short, ws_short, ws_short, ws_long};
    int paired_layers_bk[] = {0, 2, 5, 9};
    int is_paired_layer[NUM_LAYERS] = {};
    for (int j = 0; j < 4; j++) is_paired_layer[paired_layers_bk[j]] = 1;
    float attn_scale_val = g_yarn.attn_scale;

    // Gate mappings (same as forward)
    int attn_gate_map[NUM_LAYERS] = {0, 1, 2, 3, 4, 5, -1, 6, 7, 8, 9};
    int ve_gate_map[NUM_LAYERS] = {-1, 0, 1, -1, -1, -1, -1, -1, 2, 3, 4};

    // Float scratch for gate sigmoid gradient [T*H floats]
    // Located past scalar_grad_acc in scratch_f32
    float* gate_grad_scratch = act->scratch_f32 + T + 256;

    const bf16* h_resid = cs->resid_lambdas;
    const bf16* h_post_lambdas = cs->post_lambdas;
    const bf16* h_x0_lambdas = cs->x0_lambdas;
    const bf16* h_bigram_lambdas = cs->bigram_lambdas;

    // Backout backward: grad_x_backout contribution
    // Forward: x -= backout_lambda * x_backout
    // Backward: grad for x_backout (not accumulated to params, but contributes to grad_lane0 at layer 7)
    // We track this as a separate gradient that gets added to grad_lane0 at BACKOUT_LAYER
    // grad_backout = -backout_lambda * grad_x (already computed, add to grad_lane0 at layer 7 later)

    // Initialize grad_x0 and grad_x0_bigram accumulators (pre-allocated)
    bf16* grad_x0 = act->grad_x0;
    bf16* grad_x0_bigram = act->grad_x0_bigram;
    CUDA_CHECK(cudaMemsetAsync(grad_x0, 0, T * D * sizeof(bf16), ctx->stream));
    CUDA_CHECK(cudaMemsetAsync(grad_x0_bigram, 0, T * D * sizeof(bf16), ctx->stream));

    // Float accumulators for scalar/lambda gradients
    // Layout in scratch_f32 starting at offset T:
    //   [0..10]  resid_attn gradients (per layer)
    //   [11..21] resid_mlp gradients (per layer)
    //   [22..32] x0_lambda gradients (per layer)
    //   [33..43] bigram_lambda gradients (per layer)
    //   [44..87] post_lambda gradients (per layer * 4)
    //   [88]     backout_lambda gradient
    //   [89..110] sa_lambda gradients (per layer * 2: [89+2*i]=sa_lambda0, [89+2*i+1]=sa_lambda1)
    //   [111]    skip_lambda raw dot product (multiplied by (1-σ(sl)) in accumulation)
    //   Total: 112 floats
    float* scalar_grad_acc = act->scratch_f32 + T;  // safe: scratch_f32 is T*D floats
    CUDA_CHECK(cudaMemsetAsync(scalar_grad_acc, 0, 112 * sizeof(float), ctx->stream));

    // backout_lambda gradient: forward was x -= backout_lambda * x_backout
    // grad_backout_lambda = -dot(grad_x, x_backout)
    // Use a negative trick: compute dot(grad_x, x_backout) and negate at accumulation
    bf16_dot_product(scalar_grad_acc + 88, act->grad_x, act->x_backout, T * D, ctx->stream);



    BWD_TIMER_END(g_bwd_lmhead_ms);
    BWD_TIMER_START();

    // 5. Per-layer backward (reverse order 10→0)
    int attn_bank_idx = NUM_ATTN_LAYERS - 1;  // starts at 9 (layer 10 is last attn layer)
    for (int i = NUM_LAYERS - 1; i >= 0; i--) {
        float resid_attn = __bfloat162float(h_resid[i * 2]);
        float resid_mlp = __bfloat162float(h_resid[i * 2 + 1]);
        float pl_attn_ln0 = __bfloat162float(h_post_lambdas[i * 4 + 0]);
        float pl_attn_ln1 = __bfloat162float(h_post_lambdas[i * 4 + 1]);
        float pl_mlp_ln0 = __bfloat162float(h_post_lambdas[i * 4 + 2]);
        float pl_mlp_ln1 = __bfloat162float(h_post_lambdas[i * 4 + 3]);
        float x0_lambda = __bfloat162float(h_x0_lambdas[i]);
        float bigram_lambda = __bfloat162float(h_bigram_lambdas[i]);

        // Backout gradient injection at layer 7
        if (i == BACKOUT_LAYER) {
            // grad_lane0 -= backout_lambda * grad_x (from the lane merge)
            fused_add_scale(act->grad_lane0, act->grad_lane0, act->grad_x,
                           1.0f, -backout_lambda, T * D, ctx->stream);
        }

        // Weight pointers
        const bf16* c_fc = p->mlp_bank + i * 2 * MLP_HDIM * D;
        const bf16* c_proj = p->mlp_bank + i * 2 * MLP_HDIM * D + MLP_HDIM * D;

        if (i == ATTN_SKIP_LAYER) {
            // MLP-only layer backward
            // Forward: lane0 = resid_mlp * saved_lane0[i] + pl_mlp_ln0 * mlp_out
            //          mlp_out = relu(normed @ c_fc.T)² @ c_proj
            //          normed = rms_norm(saved_lane0[i])

            // Scalar gradients (before MLP backward modifies grad_lane0)
            bf16_dot_product(scalar_grad_acc + 11 + i, act->grad_lane0, act->saved_lane0[i], T * D, ctx->stream);  // resid_mlp
            bf16_dot_product(scalar_grad_acc + 44 + i * 4 + 2, act->grad_lane0, act->saved_mlp_out[i], T * D, ctx->stream);  // pl_mlp_ln0

            // grad_mlp_out = pl_mlp_ln0 * grad_lane0
            scale_tensor(act->mlp_out, 0.0f, T * D, ctx->stream);  // reuse as scratch
            fused_add_scale(act->mlp_out, act->grad_lane0, act->mlp_out, pl_mlp_ln0, 0.0f, T * D, ctx->stream);

            // MLP backward: grad_normed, grad_c_fc, grad_c_proj
            // Forward: pre = normed @ c_fc.T, post = relu(pre)², out = post @ c_proj
            bf16* grad_mlp_out_skip = act->mlp_out;  // reuse, already zeroed above

            // grad_c_proj += saved_post.T @ grad_mlp_out  (accumulate weight gradient)
            bf16* g_c_proj = g->mlp_bank + i * 2 * MLP_HDIM * D + MLP_HDIM * D;
            gemm_bf16_At(ctx->cublas_handle, act->saved_mlp_post[i], grad_mlp_out_skip, g_c_proj,
                        MLP_HDIM, D, T, 1.0f, 1.0f);

            // grad_post = grad_mlp_out @ c_proj.T -> [T, MLP_HDIM]
            gemm_bf16_Bt(ctx->cublas_handle, grad_mlp_out_skip, c_proj, act->mlp_post,
                        T, MLP_HDIM, D);

            // grad through relu²: grad_pre = 2 * relu(pre) * grad_post
            linear_relu_square_bwd(act->mlp_post, act->saved_mlp_pre[i], act->mlp_post,
                                  T, MLP_HDIM, ctx->stream);

            // grad_normed = grad_pre @ c_fc -> [T, D]
            bf16* grad_normed_skip = act->scratch2;
            gemm_bf16(ctx->cublas_handle, act->mlp_post, c_fc, grad_normed_skip,
                     T, D, MLP_HDIM);
            // grad_c_fc += grad_pre.T @ normed_mlp_input
            bf16* g_c_fc = g->mlp_bank + i * 2 * MLP_HDIM * D;
            gemm_bf16_At(ctx->cublas_handle, act->mlp_post, act->saved_normed_mlp[i], g_c_fc,
                        MLP_HDIM, D, T, 1.0f, 1.0f);

            // RMS norm backward (original x = saved_lane0 for ATTN_SKIP_LAYER)
            rms_norm_bwd(grad_normed_skip, grad_normed_skip, act->saved_lane0[i], T, D, ctx->stream);

            // grad_lane0 = resid_mlp * grad_lane0 + grad_normed
            fused_add_scale(act->grad_lane0, act->grad_lane0, grad_normed_skip, resid_mlp, 1.0f, T * D, ctx->stream);

            // Skip connection backward at layer 6 (after MLP backward, since skip injection
            // happens before MLP in forward). Now grad_lane0 = gradient w.r.t. lane0 after skip.
            // Forward: lane0 += skip_gate_out * skip_save
            //   where skip_gate_out = 2*σ(sl)*σ(skip_gate@x0[:,:12])
            // Backward: grad for skip_gate weight, skip_lambda, and skip_save
            {
                float skip_lambda_raw = __bfloat162float(h_scalars[2 * NUM_LAYERS + 2]);
                float sl_sig = 1.0f / (1.0f + expf(-skip_lambda_raw));
                float skip_scale = 2.0f * sl_sig;
                sigmoid_gate(act->scratch2, act->x0, p->skip_gate,
                            T, 12, 1, D, skip_scale, ctx->stream);

                // Skip gate weight gradient: d(loss)/d(skip_gate) via sigmoid backward
                // gate_sigmoid_grad: for each t, computes dot(grad_lane0[t,:], skip_save[t,:]) * scale * σ(z)*(1-σ(z))
                gate_sigmoid_grad(gate_grad_scratch, act->grad_lane0, act->skip_save,
                                  act->scratch2, T, 1, D, skip_scale, ctx->stream);
                gate_weight_grad(g->skip_gate, gate_grad_scratch,
                                 act->x0, T, 1, D, 12, ctx->stream);

                // Skip gate input gradient: grad_x0[:,:12] += gate_grad * skip_gate_weight
                gate_input_grad(grad_x0, gate_grad_scratch, p->skip_gate,
                                T, 1, D, 12, ctx->stream);

                // Skip lambda gradient: (1-σ(sl)) * Σ_t gate_full[t] * Σ_d grad_lane0[t,d] * skip_save[t,d]
                // Compute temp = grad_lane0 * gate_broadcast, then dot(temp, skip_save)
                elementwise_mul_broadcast(act->mlp_pre, act->grad_lane0, act->scratch2,
                                          T * D, D, T, ctx->stream);
                bf16_dot_product(scalar_grad_acc + 111, act->mlp_pre, act->skip_save,
                                 T * D, ctx->stream);

                // grad_skip_save = gate * grad_lane0 (for layer 3)
                elementwise_mul_broadcast(act->skip_save, act->grad_lane0, act->scratch2,
                                          T * D, D, T, ctx->stream);
            }

        } else if (i < PARALLEL_START) {
            // Single-stream layer backward
            // Forward: lane0_out = resid_mlp * lane0_post_attn + pl_mlp_ln0 * mlp_out
            //          lane0_post_attn = resid_attn * saved_lane0[i] + attn_out + x0_inject
            //          attn_out = attention(normed, ...)
            //          normed = rms_norm(saved_lane0[i])

            // Scalar gradients (before MLP backward modifies grad_lane0)
            bf16_dot_product(scalar_grad_acc + 11 + i, act->grad_lane0, act->saved_post_attn[i], T * D, ctx->stream);  // resid_mlp
            bf16_dot_product(scalar_grad_acc + 44 + i * 4 + 2, act->grad_lane0, act->saved_mlp_out[i], T * D, ctx->stream);  // pl_mlp_ln0

            // MLP backward
            // grad_mlp_out = pl_mlp_ln0 * grad_lane0
            bf16* grad_mlp_out = act->scratch1;
            fused_add_scale(grad_mlp_out, act->grad_lane0, act->grad_lane0, pl_mlp_ln0, 0.0f, T * D, ctx->stream);

            // MLP backward: grad through c_proj
            bf16* g_c_proj = g->mlp_bank + i * 2 * MLP_HDIM * D + MLP_HDIM * D;
            gemm_bf16_At(ctx->cublas_handle, act->saved_mlp_post[i], grad_mlp_out, g_c_proj,
                        MLP_HDIM, D, T, 1.0f, 1.0f);

            // grad_post = grad_mlp_out @ c_proj.T -> [T, MLP_HDIM]
            gemm_bf16_Bt(ctx->cublas_handle, grad_mlp_out, c_proj, act->mlp_post,
                        T, MLP_HDIM, D);

            // grad through relu²: grad_pre = 2 * relu(pre) * grad_post
            linear_relu_square_bwd(act->mlp_post, act->saved_mlp_pre[i], act->mlp_post,
                                  T, MLP_HDIM, ctx->stream);

            // grad_normed = grad_pre @ c_fc -> [T, D]
            bf16* grad_normed = act->scratch2;
            gemm_bf16(ctx->cublas_handle, act->mlp_post, c_fc, grad_normed,
                     T, D, MLP_HDIM);
            // grad_c_fc += grad_pre.T @ saved_normed_mlp (MLP input norm)
            bf16* g_c_fc = g->mlp_bank + i * 2 * MLP_HDIM * D;
            gemm_bf16_At(ctx->cublas_handle, act->mlp_post, act->saved_normed_mlp[i], g_c_fc,
                        MLP_HDIM, D, T, 1.0f, 1.0f);

            // RMS norm backward (MLP's norm input = saved_post_attn[i])
            rms_norm_bwd(grad_normed, grad_normed, act->saved_post_attn[i], T, D, ctx->stream);

            // grad_lane0 = resid_mlp * grad_lane0 + grad_normed (from MLP path)
            fused_add_scale(act->grad_lane0, act->grad_lane0, grad_normed, resid_mlp, 1.0f, T * D, ctx->stream);

            // Skip connection backward at SKIP_IN_LAYER (layer 3):
            // Add grad_skip_save AFTER MLP backward, since skip_save = post_attn (before MLP)
            if (i == SKIP_IN_LAYER) {
                fused_add_scale(act->grad_lane0, act->grad_lane0, act->skip_save,
                               1.0f, 1.0f, T * D, ctx->stream);
            }

            // x0_inject backward: grad_x0 += x0_lambda * grad_lane0, grad_x0_bigram += bigram_lambda * grad_lane0
            // Also compute scalar gradients for x0_lambda, bigram_lambda, resid_attn
            bf16_dot_product(scalar_grad_acc + 22 + i, act->grad_lane0, act->x0, T * D, ctx->stream);  // x0_lambda
            if (i > 0) {
                bf16_dot_product(scalar_grad_acc + 33 + i, act->grad_lane0, act->x0_bigram, T * D, ctx->stream);  // bigram_lambda
            }
            bf16_dot_product(scalar_grad_acc + 0 + i, act->grad_lane0, act->saved_lane0[i], T * D, ctx->stream);  // resid_attn

            fused_add_scale(grad_x0, grad_x0, act->grad_lane0, 1.0f, x0_lambda, T * D, ctx->stream);
            if (i > 0) {
                fused_add_scale(grad_x0_bigram, grad_x0_bigram, act->grad_lane0, 1.0f, bigram_lambda, T * D, ctx->stream);
            }

            // Residual backward: grad_lane0 *= resid_attn
            scale_tensor(act->grad_lane0, resid_attn, T * D, ctx->stream);

            // Attention backward: output projection + approximate attention grad
            {
                const bf16* qkvo_w = p->attn_bank + attn_bank_idx * 4 * D * D;
                // Scalars are indexed by LAYER index, not attn_bank_idx
                float sa_lambda0 = __bfloat162float(h_scalars[2 * i]);
                float sa_lambda1 = __bfloat162float(h_scalars[2 * i + 1]);
                bf16* g_attn = g->attn_bank + attn_bank_idx * 4 * D * D;

                bf16* grad_normed_attn = act->scratch2;
                bf16* y_cos = is_paired_layer[i] ? act->yarn_paired_cos : act->yarn_cos;
                bf16* y_sin = is_paired_layer[i] ? act->yarn_paired_sin : act->yarn_sin;

                // Gate pointers for this layer
                const bf16* attn_gw = (attn_gate_map[i] >= 0) ?
                    p->attn_gate_bank + attn_gate_map[i] * NUM_HEADS * 12 : NULL;
                bf16* g_attn_gw = (attn_gate_map[i] >= 0) ?
                    g->attn_gate_bank + attn_gate_map[i] * NUM_HEADS * 12 : NULL;

                // VE pointers for this layer
                const bf16* ve_gw = (ve_gate_map[i] >= 0) ?
                    p->ve_gate_bank + ve_gate_map[i] * NUM_HEADS * 12 : NULL;
                bf16* g_ve_gw = (ve_gate_map[i] >= 0) ?
                    g->ve_gate_bank + ve_gate_map[i] * NUM_HEADS * 12 : NULL;

                attention_backward(ctx, attn_bank_idx, act->grad_lane0, grad_normed_attn,
                                  qkvo_w, sa_lambda0, sa_lambda1, g_attn,
                                  act->saved_normed[i], cum_seqlens, num_seqs,
                                  T, bm_sizes[i], train_max_seq_len,
                                  is_paired_layer[i], attn_scale_val,
                                  y_cos, y_sin,
                                  attn_gw, g_attn_gw, gate_grad_scratch,
                                  ve_gw, g_ve_gw, g->value_embeds,
                                  inputs, ve_gate_map[i],
                                  (ve_gate_map[i] >= 0) ? ve_gate_map[i] : 0,
                                  p->value_embeds,
                                  i, scalar_grad_acc);

                // RMS norm backward for attention input
                rms_norm_bwd(grad_normed_attn, grad_normed_attn, act->saved_lane0[i], T, D, ctx->stream);
                fused_add_scale(act->grad_lane0, act->grad_lane0, grad_normed_attn, 1.0f, 1.0f, T * D, ctx->stream);
            }

            attn_bank_idx--;
        } else {
            // Parallel layers backward (7-10)
            // Forward: lane0_new = resid_mlp * lane0_post_attn + pl_mlp_ln0 * mlp_out
            //          lane1_new = resid_mlp * lane1_post_attn + pl_mlp_ln1 * mlp_out

            // Scalar gradients (before MLP backward modifies grad_lane0/lane1)
            bf16_dot_product(scalar_grad_acc + 11 + i, act->grad_lane0, act->saved_post_attn[i], T * D, ctx->stream);  // resid_mlp (from lane0)
            bf16_dot_product(scalar_grad_acc + 11 + i, act->grad_lane1, act->saved_lane1_post_attn[i], T * D, ctx->stream);  // resid_mlp (from lane1, accumulate)
            bf16_dot_product(scalar_grad_acc + 44 + i * 4 + 2, act->grad_lane0, act->saved_mlp_out[i], T * D, ctx->stream);  // pl_mlp_ln0
            bf16_dot_product(scalar_grad_acc + 44 + i * 4 + 3, act->grad_lane1, act->saved_mlp_out[i], T * D, ctx->stream);  // pl_mlp_ln1

            // grad_mlp_out = pl_mlp_ln0 * grad_lane0 + pl_mlp_ln1 * grad_lane1
            bf16* grad_mlp_out = act->scratch1;
            fused_add_scale(grad_mlp_out, act->grad_lane0, act->grad_lane1,
                           pl_mlp_ln0, pl_mlp_ln1, T * D, ctx->stream);

            // MLP backward through c_proj
            bf16* g_c_proj = g->mlp_bank + i * 2 * MLP_HDIM * D + MLP_HDIM * D;
            gemm_bf16_At(ctx->cublas_handle, act->saved_mlp_post[i], grad_mlp_out, g_c_proj,
                        MLP_HDIM, D, T, 1.0f, 1.0f);

            // grad_post = grad_mlp_out @ c_proj.T
            gemm_bf16_Bt(ctx->cublas_handle, grad_mlp_out, c_proj, act->mlp_post,
                        T, MLP_HDIM, D);

            // grad through relu²
            linear_relu_square_bwd(act->mlp_post, act->saved_mlp_pre[i], act->mlp_post,
                                  T, MLP_HDIM, ctx->stream);

            // grad_normed_mlp = grad_pre @ c_fc
            bf16* grad_normed = act->scratch2;
            gemm_bf16(ctx->cublas_handle, act->mlp_post, c_fc, grad_normed,
                     T, D, MLP_HDIM);
            // grad_c_fc += grad_pre.T @ saved_normed_mlp (MLP input norm)
            bf16* g_c_fc = g->mlp_bank + i * 2 * MLP_HDIM * D;
            gemm_bf16_At(ctx->cublas_handle, act->mlp_post, act->saved_normed_mlp[i], g_c_fc,
                        MLP_HDIM, D, T, 1.0f, 1.0f);

            // RMS norm backward (original input = lane1 after attn update)
            rms_norm_bwd(grad_normed, grad_normed, act->saved_lane1_post_attn[i], T, D, ctx->stream);

            // Update grad_lane0 and grad_lane1 with residual terms
            scale_tensor(act->grad_lane0, resid_mlp, T * D, ctx->stream);
            fused_add_scale(act->grad_lane1, act->grad_lane1, grad_normed, resid_mlp, 1.0f, T * D, ctx->stream);

            // Attention backward for parallel layers
            // grad_attn_out = pl_attn_ln0 * grad_lane0 + pl_attn_ln1 * grad_lane1
            bf16* grad_attn = act->scratch1;
            fused_add_scale(grad_attn, act->grad_lane0, act->grad_lane1,
                           pl_attn_ln0, pl_attn_ln1, T * D, ctx->stream);

            // Scalar gradients for pl_attn, x0, bigram, resid_attn
            // At this point, grad_lane0/lane1 have been updated through MLP residual backward
            // Now they represent gradient w.r.t. lane0/lane1 post-attn-update
            bf16_dot_product(scalar_grad_acc + 44 + i * 4 + 0, act->grad_lane0, act->saved_attn_proj[i], T * D, ctx->stream);  // pl_attn_ln0
            bf16_dot_product(scalar_grad_acc + 44 + i * 4 + 1, act->grad_lane1, act->saved_attn_proj[i], T * D, ctx->stream);  // pl_attn_ln1
            bf16_dot_product(scalar_grad_acc + 22 + i, act->grad_lane0, act->x0, T * D, ctx->stream);  // x0_lambda
            bf16_dot_product(scalar_grad_acc + 0 + i, act->grad_lane0, act->saved_lane0[i], T * D, ctx->stream);  // resid_attn (lane0)
            bf16_dot_product(scalar_grad_acc + 0 + i, act->grad_lane1, act->saved_lane1[i], T * D, ctx->stream);  // resid_attn (lane1, accumulate)

            // x0_inject backward
            fused_add_scale(grad_x0, grad_x0, act->grad_lane0, 1.0f, x0_lambda, T * D, ctx->stream);
            if (i > 0) {
                bf16_dot_product(scalar_grad_acc + 33 + i, act->grad_lane0, act->x0_bigram, T * D, ctx->stream);  // bigram_lambda
                fused_add_scale(grad_x0_bigram, grad_x0_bigram, act->grad_lane0, 1.0f, bigram_lambda, T * D, ctx->stream);
            }

            // Residual backward
            scale_tensor(act->grad_lane0, resid_attn, T * D, ctx->stream);
            scale_tensor(act->grad_lane1, resid_attn, T * D, ctx->stream);

            // Attention backward through output projection
            {
                const bf16* qkvo_w = p->attn_bank + attn_bank_idx * 4 * D * D;
                // Scalars indexed by LAYER index, not attn_bank_idx
                float sa_lambda0 = __bfloat162float(h_scalars[2 * i]);
                float sa_lambda1 = __bfloat162float(h_scalars[2 * i + 1]);
                bf16* g_attn = g->attn_bank + attn_bank_idx * 4 * D * D;

                bf16* grad_normed_attn = act->scratch1;
                bf16* y_cos = is_paired_layer[i] ? act->yarn_paired_cos : act->yarn_cos;
                bf16* y_sin = is_paired_layer[i] ? act->yarn_paired_sin : act->yarn_sin;

                // Gate pointers for this layer
                const bf16* attn_gw = (attn_gate_map[i] >= 0) ?
                    p->attn_gate_bank + attn_gate_map[i] * NUM_HEADS * 12 : NULL;
                bf16* g_attn_gw = (attn_gate_map[i] >= 0) ?
                    g->attn_gate_bank + attn_gate_map[i] * NUM_HEADS * 12 : NULL;

                // VE pointers for this layer
                const bf16* ve_gw = (ve_gate_map[i] >= 0) ?
                    p->ve_gate_bank + ve_gate_map[i] * NUM_HEADS * 12 : NULL;
                bf16* g_ve_gw = (ve_gate_map[i] >= 0) ?
                    g->ve_gate_bank + ve_gate_map[i] * NUM_HEADS * 12 : NULL;

                attention_backward(ctx, attn_bank_idx, grad_attn, grad_normed_attn,
                                  qkvo_w, sa_lambda0, sa_lambda1, g_attn,
                                  act->saved_normed[i], cum_seqlens, num_seqs,
                                  T, bm_sizes[i], train_max_seq_len,
                                  is_paired_layer[i], attn_scale_val,
                                  y_cos, y_sin,
                                  attn_gw, g_attn_gw, gate_grad_scratch,
                                  ve_gw, g_ve_gw, g->value_embeds,
                                  inputs, ve_gate_map[i],
                                  (ve_gate_map[i] >= 0) ? ve_gate_map[i] : 0,
                                  p->value_embeds,
                                  i, scalar_grad_acc);

                // RMS norm backward for attention input (lane0)
                rms_norm_bwd(grad_normed_attn, grad_normed_attn, act->saved_lane0[i], T, D, ctx->stream);
                fused_add_scale(act->grad_lane0, act->grad_lane0, grad_normed_attn, 1.0f, 1.0f, T * D, ctx->stream);
            }

            attn_bank_idx--;
        }

        // Merge lanes at PARALLEL_START backward
        if (i == PARALLEL_START) {
            // Forward: lane1 = lane0 (copy)
            // Backward: grad_lane0 += grad_lane1
            fused_add_scale(act->grad_lane0, act->grad_lane0, act->grad_lane1, 1.0f, 1.0f, T * D, ctx->stream);
        }
    }

    BWD_TIMER_END(g_bwd_layers_ms);
    BWD_TIMER_START();

    // 5b. Initial lane0 backward: lane0 = x0 + bigram_lambda0 * x0_bigram
    // grad_lane0 holds gradient flowing to the initial lane0 — must be added
    // to grad_x0 and grad_x0_bigram before embedding backward
    {
        float bigram_lambda0 = __bfloat162float(cs->bigram_lambdas[0]);
        fused_add_scale(grad_x0, grad_x0, act->grad_lane0, 1.0f, 1.0f, T * D, ctx->stream);
        fused_add_scale(grad_x0_bigram, grad_x0_bigram, act->grad_lane0, 1.0f, bigram_lambda0, T * D, ctx->stream);
        // bigram_lambda[0] scalar gradient (skipped in loop since i > 0 guard)
        bf16_dot_product(scalar_grad_acc + 33, act->grad_lane0, act->x0_bigram, T * D, ctx->stream);
    }

    // 6. Embedding backward
    // Recompute smeared embeddings for correct rms_norm_bwd input.
    // Forward was: gather_embed -> smear_forward -> rms_norm_fwd -> x0
    {
        float smear_lambda = __bfloat162float(cs->scalars[2 * NUM_LAYERS]);
        bf16* smeared_embed = act->scratch1;
        gather_embed(smeared_embed, p->embed, inputs, T, D, ctx->stream);
        smear_forward(smeared_embed, smeared_embed, p->smear_gate, smear_lambda, T, D, ctx->stream);
        rms_norm_bwd(grad_x0, grad_x0, smeared_embed, T, D, ctx->stream);
    }

    // scatter-add into embed gradient
    scatter_add_embed(g->embed, grad_x0, inputs, T, D, ctx->stream);

    // scatter-add into bigram_embed gradient
    scatter_add_embed(g->bigram_embed, grad_x0_bigram, bigram_inputs, T, D, ctx->stream);

    BWD_TIMER_END(g_bwd_embed_ms);
    BWD_TIMER_START();

    // 7. Accumulate scalar gradient accumulators into gradient buffers (GPU-side)
    // scalar_grad_acc layout: [0..10] resid_attn, [11..21] resid_mlp,
    //   [22..32] x0_lambda, [33..43] bigram_lambda, [44..87] post_lambdas, [88] backout_lambda,
    //   [89..110] sa_lambda (2 per layer), [111] skip_lambda raw dot
    float skip_lambda_raw = __bfloat162float(cs->scalars[2 * NUM_LAYERS + 2]);
    float sl_sig = 1.0f / (1.0f + expf(-skip_lambda_raw));
    float skip_lambda_factor = 1.0f - sl_sig;
    accumulate_scalar_grads(g->resid_lambdas, g->x0_lambdas, g->bigram_lambdas,
                            g->post_lambdas, g->scalars,
                            scalar_grad_acc, NUM_LAYERS,
                            skip_lambda_factor, ctx->stream);

    BWD_TIMER_END(g_bwd_scalgrad_ms);
    #undef BWD_TIMER_START
    #undef BWD_TIMER_END
}

// ============================================================================
// Section F: YaRN Rotary Position Embeddings
// ============================================================================

// YarnState struct and globals declared above Section D

static void yarn_init(YarnState* yarn, int head_dim, int max_seq_len, int paired) {
    yarn->head_dim = head_dim;
    yarn->max_seq_len = max_seq_len;
    yarn->paired = paired;
    yarn->attn_scale = 0.1f;

    // angular_freq = (1/1024)^linspace(0,1, head_dim//4)
    // Then repeat_interleave(2), pad with zeros to head_dim
    int quarter = head_dim / 4;
    float freq[HEAD_DIM];
    for (int i = 0; i < quarter; i++) {
        float t = (float)i / (quarter - 1);
        float base_freq = powf(1.0f / 1024.0f, t);
        // repeat_interleave(2)
        freq[2 * i] = base_freq;
        freq[2 * i + 1] = base_freq;
    }
    // Zero pad second half
    for (int i = head_dim / 2; i < head_dim; i++) {
        freq[i] = 0.0f;
    }

    memcpy(yarn->angular_freq, freq, head_dim * sizeof(float));
}

// d_freq_scratch: caller provides HEAD_DIM floats of GPU scratch for frequency upload
static void yarn_compute(YarnState* yarn, bf16* cos_out, bf16* sin_out,
                         float* d_freq_scratch, cudaStream_t stream) {
    CUDA_CHECK(cudaMemcpyAsync(d_freq_scratch, yarn->angular_freq,
               yarn->head_dim * sizeof(float), cudaMemcpyHostToDevice, stream));

    yarn_compute_tables(cos_out, sin_out, d_freq_scratch, yarn->head_dim,
                        yarn->max_seq_len, yarn->paired, stream);
}

static void yarn_apply_extension(YarnState* yarn, int old_window, int new_window) {
    // YaRN window extension
    float scaling_factor = (float)old_window / new_window;
    int alpha = 1, beta = 32;

    for (int i = 0; i < yarn->head_dim; i++) {
        float rotations = old_window * yarn->angular_freq[i] / (2.0f * M_PI);
        float interp_weight = fminf(fmaxf((rotations - alpha) / (float)(beta - alpha), 0.0f), 1.0f);
        yarn->angular_freq[i] *= scaling_factor + interp_weight * (1.0f - scaling_factor);
    }

    yarn->attn_scale *= 0.2f * logf((float)new_window / old_window) + 1.0f;
}

// ============================================================================
// Section G: Optimizer Step
// ============================================================================

// Nesterov momentum + Polar Express orthogonalization
static void polar_express(TrainingContext* ctx,
                          bf16* grad_chunk, float* momentum_buffer,
                          float momentum, int batch, int rows, int cols,
                          int is_tall) {
    // 1. Nesterov momentum: update buffer, produce lookahead gradient in BF16
    nesterov_momentum(grad_chunk, momentum_buffer, grad_chunk, momentum,
                      batch * rows * cols, ctx->stream);

    // 2. Normalize spectral norm: X /= (norm * 1.02 + eps)
    // Coefficients are computed for safety_factor=2e-2, must match
    float* norms = ctx->opt.polar_norms;
    tensor_norm(grad_chunk, norms, batch, rows, cols, ctx->stream);
    norm_divide(grad_chunk, norms, batch, rows * cols, 2e-2f, 1e-6f, ctx->stream);

    // 3. Polar Express iterations (using pre-allocated scratch)
    bf16* A = ctx->opt.polar_A;
    bf16* B = ctx->opt.polar_B;
    bf16* C = ctx->opt.polar_C;
    bf16* orig_grad = grad_chunk;  // save original pointer for copy-back
    int small_dim = is_tall ? cols : rows;

    for (int iter = 0; iter < POLAR_NUM_ITERS; iter++) {
        float a = POLAR_COEFFS[iter][0];
        float b = POLAR_COEFFS[iter][1];
        float c = POLAR_COEFFS[iter][2];

        if (is_tall) {
            // A = X.T @ X
            xtx_kernel_launch(grad_chunk, A, batch, rows, cols,
                             rows * cols, cols, 1,
                             small_dim * small_dim, small_dim, 1,
                             ctx->stream);
            // B = b*A + c*(A@A)
            ba_plus_caa_kernel_launch(A, B, batch, small_dim,
                                     small_dim * small_dim, small_dim, 1,
                                     small_dim * small_dim, small_dim, 1,
                                     c, b, ctx->stream);
            // C = a*X + X@B
            gemm_bf16_batched(ctx->cublas_handle, grad_chunk, B, C,
                            rows, cols, cols, batch,
                            rows * cols, cols * cols, rows * cols,
                            1.0f, 0.0f);
            // Add a*X
            fused_add_scale(C, C, grad_chunk, 1.0f, a, batch * rows * cols, ctx->stream);
        } else {
            // A = X @ X.T
            xxt_kernel_launch(grad_chunk, A, batch, rows, cols,
                             rows * cols, cols, 1,
                             small_dim * small_dim, small_dim, 1,
                             ctx->stream);
            // B = b*A + c*(A@A)
            ba_plus_caa_kernel_launch(A, B, batch, small_dim,
                                     small_dim * small_dim, small_dim, 1,
                                     small_dim * small_dim, small_dim, 1,
                                     c, b, ctx->stream);
            // C = a*X + B@X
            gemm_bf16_batched(ctx->cublas_handle, B, grad_chunk, C,
                            rows, cols, rows, batch,
                            rows * rows, rows * cols, rows * cols,
                            1.0f, 0.0f);
            fused_add_scale(C, C, grad_chunk, 1.0f, a, batch * rows * cols, ctx->stream);
        }

        // Swap X and C
        bf16* tmp = grad_chunk;
        grad_chunk = C;
        C = tmp;
    }

    // After POLAR_NUM_ITERS swaps, result may be in scratch buffer.
    // Copy back to original gradient buffer if needed.
    if (grad_chunk != orig_grad) {
        CUDA_CHECK(cudaMemcpyAsync(orig_grad, grad_chunk, batch * rows * cols * sizeof(bf16),
                   cudaMemcpyDeviceToDevice, ctx->stream));
    }
}

// Forward declaration (defined in Section H)
static float get_lr(int step);

void optimizer_step(TrainingContext* ctx, int step, int do_adam) {
    ModelParams* p = &ctx->params;
    ModelGrads* g = &ctx->grads;
    OptimizerState* opt = &ctx->opt;
    cudaStream_t stream = ctx->stream;

    // Get schedule parameters
    float step_lr = get_lr(step);  // Stage LR multiplier + cooldown
    float muon_momentum = MUON_MOMENTUM;

    // Muon momentum warmup/cooldown
    int muon_warmup = 300;
    int muon_cd_start = TOTAL_STEPS - 50;
    if (step < muon_warmup) {
        float frac = (float)step / muon_warmup;
        muon_momentum = 0.85f + frac * 0.10f;
    } else if (step > muon_cd_start) {
        float frac = (float)(step - muon_cd_start) / 50.0f;
        muon_momentum = 0.95f - frac * 0.10f;
    }

    // ---- Adam updates (only on odd steps) ----
    if (do_adam) {
        // Update each Adam parameter
        // scalars
        {
            AdamState* s = &opt->adam_scalars;
            s->step++;
            float b1 = 0.9f, b2 = 0.99f;
            float bias1 = 1.0f - powf(b1, (float)s->step);
            float bias2 = 1.0f - powf(b2, (float)s->step);
            float lr = ADAM_LR * step_lr * 5.0f;  // lr_mul=5.0
            float step_size = lr * sqrtf(bias2) / bias1;
            float eff_wd = lr * lr * ADAM_WD * 0.0f;  // wd_mul=0.0
            adam_update(p->scalars, g->scalars, s->exp_avg, s->exp_avg_sq,
                       b1, b2, ADAM_EPS, step_size, eff_wd, s->numel, stream);
        }

        // smear_gate
        {
            AdamState* s = &opt->adam_smear_gate;
            s->step++;
            float b1 = 0.9f, b2 = 0.99f;
            float bias1 = 1.0f - powf(b1, (float)s->step);
            float bias2 = 1.0f - powf(b2, (float)s->step);
            float lr = ADAM_LR * step_lr * 0.01f;
            float step_size = lr * sqrtf(bias2) / bias1;
            adam_update(p->smear_gate, g->smear_gate, s->exp_avg, s->exp_avg_sq,
                       b1, b2, ADAM_EPS, step_size, 0.0f, s->numel, stream);
        }

        // skip_gate
        {
            AdamState* s = &opt->adam_skip_gate;
            s->step++;
            float b1 = 0.9f, b2 = 0.99f;
            float bias1 = 1.0f - powf(b1, (float)s->step);
            float bias2 = 1.0f - powf(b2, (float)s->step);
            float lr = ADAM_LR * step_lr * 0.05f;
            float step_size = lr * sqrtf(bias2) / bias1;
            adam_update(p->skip_gate, g->skip_gate, s->exp_avg, s->exp_avg_sq,
                       b1, b2, ADAM_EPS, step_size, 0.0f, s->numel, stream);
        }

        // attn_gate_bank
        {
            AdamState* s = &opt->adam_attn_gate_bank;
            s->step++;
            float b1 = 0.9f, b2 = 0.99f;
            float bias1 = 1.0f - powf(b1, (float)s->step);
            float bias2 = 1.0f - powf(b2, (float)s->step);
            float lr = ADAM_LR * step_lr * 1.0f;
            float step_size = lr * sqrtf(bias2) / bias1;
            float eff_wd = lr * lr * ADAM_WD * 1.0f;
            adam_update(p->attn_gate_bank, g->attn_gate_bank, s->exp_avg, s->exp_avg_sq,
                       b1, b2, ADAM_EPS, step_size, eff_wd, s->numel, stream);
        }

        // ve_gate_bank
        {
            AdamState* s = &opt->adam_ve_gate_bank;
            s->step++;
            float b1 = 0.9f, b2 = 0.99f;
            float bias1 = 1.0f - powf(b1, (float)s->step);
            float bias2 = 1.0f - powf(b2, (float)s->step);
            float lr = ADAM_LR * step_lr;
            float step_size = lr * sqrtf(bias2) / bias1;
            float eff_wd = lr * lr * ADAM_WD;
            adam_update(p->ve_gate_bank, g->ve_gate_bank, s->exp_avg, s->exp_avg_sq,
                       b1, b2, ADAM_EPS, step_size, eff_wd, s->numel, stream);
        }

        // lm_head (with embed tying)
        {
            int lm_numel = MODEL_DIM * VOCAB_SIZE;

            // When tied: add embed.grad.T into lm_head.grad
            if (!opt->split_embed) {
                transpose_add(g->embed, g->lm_head, VOCAB_SIZE, MODEL_DIM, stream);
            }

            AdamState* s = &opt->adam_lm_head;
            s->step++;
            float b1 = 0.5f, b2 = 0.95f;
            float bias1 = 1.0f - powf(b1, (float)s->step);
            float bias2 = 1.0f - powf(b2, (float)s->step);
            float lr = ADAM_LR * step_lr;
            float step_size = lr * sqrtf(bias2) / bias1;
            float eff_wd = lr * lr * ADAM_WD * 150.0f;
            adam_update(p->lm_head, g->lm_head, s->exp_avg, s->exp_avg_sq,
                       b1, b2, ADAM_EPS, step_size, eff_wd, lm_numel, stream);

            // Copy lm_head.T to embed when tied
            if (!opt->split_embed) {
                transpose_copy(p->lm_head, p->embed, MODEL_DIM, VOCAB_SIZE, stream);
            }
        }

        // embed (only when split)
        if (opt->split_embed) {
            AdamState* s = &opt->adam_embed;
            s->step++;
            float b1 = 0.5f, b2 = 0.95f;
            float bias1 = 1.0f - powf(b1, (float)s->step);
            float bias2 = 1.0f - powf(b2, (float)s->step);
            float lr = ADAM_LR * step_lr;
            float step_size = lr * sqrtf(bias2) / bias1;
            float eff_wd = lr * lr * ADAM_WD * 150.0f;
            adam_update(p->embed, g->embed, s->exp_avg, s->exp_avg_sq,
                       b1, b2, ADAM_EPS, step_size, eff_wd, s->numel, stream);
        }

        // bigram_embed
        {
            AdamState* s = &opt->adam_bigram_embed;
            s->step++;
            float b1 = 0.75f, b2 = 0.95f;
            float bias1 = 1.0f - powf(b1, (float)s->step);
            float bias2 = 1.0f - powf(b2, (float)s->step);
            float lr = ADAM_LR * step_lr * 75.0f;
            float step_size = lr * sqrtf(bias2) / bias1;
            float eff_wd = lr * lr * ADAM_WD * 5.0f;
            adam_update(p->bigram_embed, g->bigram_embed, s->exp_avg, s->exp_avg_sq,
                       b1, b2, ADAM_EPS, step_size, eff_wd, s->numel, stream);
        }

        // value_embeds
        {
            AdamState* s = &opt->adam_value_embeds;
            s->step++;
            float b1 = 0.75f, b2 = 0.95f;
            float bias1 = 1.0f - powf(b1, (float)s->step);
            float bias2 = 1.0f - powf(b2, (float)s->step);
            float lr = ADAM_LR * step_lr * 75.0f;
            float step_size = lr * sqrtf(bias2) / bias1;
            float eff_wd = lr * lr * ADAM_WD * 5.0f;
            adam_update(p->value_embeds, g->value_embeds, s->exp_avg, s->exp_avg_sq,
                       b1, b2, ADAM_EPS, step_size, eff_wd, s->numel, stream);
        }

        // post_lambdas, x0_lambdas, bigram_lambdas, resid_lambdas
        {
            AdamState* s = &opt->adam_post_lambdas;
            s->step++;
            float b1 = 0.9f, b2 = 0.95f;
            float bias1 = 1.0f - powf(b1, (float)s->step);
            float bias2 = 1.0f - powf(b2, (float)s->step);
            float lr = ADAM_LR * step_lr;
            float step_size = lr * sqrtf(bias2) / bias1;
            adam_update(p->post_lambdas, g->post_lambdas, s->exp_avg, s->exp_avg_sq,
                       b1, b2, ADAM_EPS, step_size, 0.0f, s->numel, stream);
        }
        {
            AdamState* s = &opt->adam_x0_lambdas;
            s->step++;
            float b1 = 0.9f, b2 = 0.95f;
            float bias1 = 1.0f - powf(b1, (float)s->step);
            float bias2 = 1.0f - powf(b2, (float)s->step);
            float lr = ADAM_LR * step_lr;
            float step_size = lr * sqrtf(bias2) / bias1;
            adam_update(p->x0_lambdas, g->x0_lambdas, s->exp_avg, s->exp_avg_sq,
                       b1, b2, ADAM_EPS, step_size, 0.0f, s->numel, stream);
        }
        {
            AdamState* s = &opt->adam_bigram_lambdas;
            s->step++;
            float b1 = 0.9f, b2 = 0.95f;
            float bias1 = 1.0f - powf(b1, (float)s->step);
            float bias2 = 1.0f - powf(b2, (float)s->step);
            float lr = ADAM_LR * step_lr;
            float step_size = lr * sqrtf(bias2) / bias1;
            adam_update(p->bigram_lambdas, g->bigram_lambdas, s->exp_avg, s->exp_avg_sq,
                       b1, b2, ADAM_EPS, step_size, 0.0f, s->numel, stream);
        }
        {
            AdamState* s = &opt->adam_resid_lambdas;
            s->step++;
            float b1 = 0.9f, b2 = 0.95f;
            float bias1 = 1.0f - powf(b1, (float)s->step);
            float bias2 = 1.0f - powf(b2, (float)s->step);
            float lr = ADAM_LR * step_lr * 5.0f;
            float step_size = lr * sqrtf(bias2) / bias1;
            adam_update(p->resid_lambdas, g->resid_lambdas, s->exp_avg, s->exp_avg_sq,
                       b1, b2, ADAM_EPS, step_size, 0.0f, s->numel, stream);
        }

        // Split embed at designated step
        if (step == opt->split_step) {
            // Copy lm_head optimizer state to embed (with transpose)
            // For single GPU: simple transpose of exp_avg and exp_avg_sq
            // TODO: implement optimizer state copy with transpose
            opt->split_embed = 1;
        }
    }

    // ---- NorMuon updates (always) ----
    // attn_bank: tall path (768x768 matrices), use XTX
    {
        NorMuonState* s = &opt->muon_attn_bank;
        float eff_lr = MUON_LR * step_lr;
        // Shape multiplier: max(1, rows/cols)^0.5 * 1.0 for attn
        float shape_mult = 1.0f;  // 768/768 = 1, so no scaling
        float final_lr = shape_mult * eff_lr;
        float eff_wd = MUON_WD * MUON_LR * step_lr;

        // Reshape grad to [40, 768, 768]
        bf16* grad = g->attn_bank;  // already in correct shape
        int batch = NUM_ATTN_LAYERS * 4;  // 40
        int rows = MODEL_DIM;  // 768
        int cols = MODEL_DIM;  // 768

        // Polar Express (fused Nesterov + orthogonalization)
        polar_express(ctx, grad, s->momentum_buffer, muon_momentum,
                      batch, rows, cols, 1);  // tall: rows >= cols

        // Variance reduction
        normuon_variance_reduction(grad, s->second_momentum_buffer,
                                   MUON_BETA2, -1,  // reduce along cols (d1)
                                   batch, rows, cols, stream);

        // Cautious WD + update
        muon_cautious_update((uint16_t*)p->attn_bank, s->mantissa,
                            grad, eff_wd, final_lr, s->numel, stream);
    }

    // mlp_bank: tall path (3072x768 matrices), use XTX
    {
        NorMuonState* s = &opt->muon_mlp_bank;
        float eff_lr_base = MUON_LR * step_lr;
        float shape_mult = sqrtf(fmaxf(1.0f, (float)MLP_HDIM / MODEL_DIM));  // sqrt(4) = 2
        float eff_wd = MUON_WD * MUON_LR * step_lr;

        bf16* grad = g->mlp_bank;
        int batch = 24;
        int rows = MLP_HDIM;   // 3072
        int cols = MODEL_DIM;  // 768

        polar_express(ctx, grad, s->momentum_buffer, muon_momentum,
                      batch, rows, cols, 1);  // tall: rows > cols

        normuon_variance_reduction(grad, s->second_momentum_buffer,
                                   MUON_BETA2, -1,  // reduce along cols (d1) since rows >= cols
                                   batch, rows, cols, stream);

        // Per-matrix LR: c_proj (odd indices) get 2x LR
        for (int mat = 0; mat < batch; mat++) {
            int is_c_proj = (mat % 2 == 1);
            float mat_lr = shape_mult * eff_lr_base * (is_c_proj ? 2.0f : 1.0f);
            muon_cautious_update((uint16_t*)p->mlp_bank + mat * rows * cols,
                                s->mantissa + mat * rows * cols,
                                grad + mat * rows * cols,
                                eff_wd, mat_lr, rows * cols, stream);
        }
    }
}

// ============================================================================
// Section H: Training Schedule + Loop
// ============================================================================

static TrainingStageConfig STAGES[4] = {
    {1.0f,   8*2048*8,  1, 3,  896, {1.0f, 0.5f, 0.25f}, {1.0f, 0.5f, 0.0f}, 3, 1.0f/3.0f},
    {1.52f, 16*2048*8,  3, 7, 2048, {1.0f, 0.5f, 0.0f},  {1.0f, 0.0f, 0.0f}, 2, 1.0f/3.0f},
    {1.73f, 24*2048*8,  5, 11, 2048, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, 1, 1.0f/3.0f},
    {1.0f,  24*2048*8,  6, 13, 2048, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, 1, 0.0f},  // extension
};

static void get_stage(int step, int* stage_idx, float* stage_t) {
    int boundaries[5];
    boundaries[0] = 0;
    boundaries[1] = (int)roundf(NUM_SCHEDULED_ITERS * STAGES[0].duration);
    boundaries[2] = (int)roundf(NUM_SCHEDULED_ITERS * (STAGES[0].duration + STAGES[1].duration));
    boundaries[3] = NUM_SCHEDULED_ITERS;
    boundaries[4] = TOTAL_STEPS;

    for (int i = 0; i < 4; i++) {
        if (step < boundaries[i + 1]) {
            *stage_idx = i;
            *stage_t = (float)(step - boundaries[i]) / (float)(boundaries[i + 1] - boundaries[i]);
            return;
        }
    }
    *stage_idx = 3;
    *stage_t = 1.0f;
}

static float get_lr(int step) {
    int stage_idx;
    float stage_t;
    get_stage(step, &stage_idx, &stage_t);
    float lr = STAGES[stage_idx].lr_mul;
    int cd_start = (int)(NUM_SCHEDULED_ITERS * (1.0f - COOLDOWN_FRAC));
    if (step >= cd_start) {
        float t = fminf(1.0f, (float)(step - cd_start) / (NUM_SCHEDULED_ITERS - cd_start));
        lr = lr * (1.0f - t) + 0.15f * t;
    }
    return lr;
}

static void get_mtp_weights(int step, float* weights, int* n_weights) {
    int stage_idx;
    float t;
    get_stage(step, &stage_idx, &t);
    *n_weights = STAGES[stage_idx].n_mtp_weights;
    for (int i = 0; i < *n_weights; i++) {
        weights[i] = STAGES[stage_idx].mtp_weights_start[i]
            + (STAGES[stage_idx].mtp_weights_end[i] - STAGES[stage_idx].mtp_weights_start[i]) * t;
    }
}

// ============================================================================
// Main
// ============================================================================

void init_training_context(TrainingContext* ctx) {
    memset(ctx, 0, sizeof(*ctx));
    // Reconstruct the C++ unordered_map after memset (which corrupts it)
    new (&ctx->cudnn_graph_cache) CudnnGraphCache();

    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaStreamCreate(&ctx->stream));
    CUBLAS_CHECK(cublasCreate(&ctx->cublas_handle));
    CUBLAS_CHECK(cublasSetStream(ctx->cublas_handle, ctx->stream));
    CUBLAS_CHECK(cublasSetMathMode(ctx->cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH));
    CUBLAS_CHECK(cublasLtCreate(&ctx->cublaslt_handle));
    CUDNN_CHECK(cudnnCreate(&ctx->cudnn_handle));
    CUDNN_CHECK(cudnnSetStream(ctx->cudnn_handle, ctx->stream));
    CURAND_CHECK(curandCreateGenerator(&ctx->curand_gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(ctx->curand_gen, 42));

    // Arena allocator available but not currently used
    // Individual buffers use cudaMalloc directly
}

void free_training_context(TrainingContext* ctx) {
    cublasDestroy(ctx->cublas_handle);
    cublasLtDestroy(ctx->cublaslt_handle);
    cudnnDestroy(ctx->cudnn_handle);
    curandDestroyGenerator(ctx->curand_gen);
    cudaStreamDestroy(ctx->stream);
    arena_free();
}

int main(int argc, char** argv) {
    // Prevent CUDA runtime's internal allocations from raising glibc's dynamic mmap threshold,
    // which causes our large host mallocs to use sbrk (conflicting with CUDA's heap structures).
    mallopt(M_MMAP_THRESHOLD, 65536);
    setlinebuf(stdout);
    setlinebuf(stderr);

    printf("=== modded-nanogpt-vk CUDA C Training ===\n");

    TrainingContext ctx;
    init_training_context(&ctx);
    fprintf(stderr, "init_model_params...\n");
    init_model_params(&ctx);
    fprintf(stderr, "alloc_grads...\n");
    alloc_grads(&ctx.grads);
    alloc_optimizer_state(&ctx.opt);

    // Determine max tokens across all stages
    // Training: max is STAGES[2].batch_size / GRAD_ACCUM_STEPS = 24*2048*8/8 = 49152
    int max_tokens = 24 * 2048;  // 49152
    fprintf(stderr, "alloc_activations (max_tokens=%d)...\n", max_tokens);
    alloc_activations(&ctx.act, max_tokens);
    fprintf(stderr, "alloc_activations done\n");

    // Initialize YaRN
    int max_sl = VAL_BATCH_SIZE / GRAD_ACCUM_STEPS;
    fprintf(stderr, "yarn_init...\n");
    yarn_init(&g_yarn, HEAD_DIM, max_sl, 0);
    yarn_init(&g_yarn_paired, HEAD_DIM, max_sl, 1);
    fprintf(stderr, "yarn_compute...\n");
    yarn_compute(&g_yarn, ctx.act.yarn_cos, ctx.act.yarn_sin, ctx.act.scratch_f32, ctx.stream);
    yarn_compute(&g_yarn_paired, ctx.act.yarn_paired_cos, ctx.act.yarn_paired_sin, ctx.act.scratch_f32, ctx.stream);
    fprintf(stderr, "yarn done\n");

    // Load data
    fprintf(stderr, "Loading training data...\n");
    char data_path[512];
    const char* dp = getenv("DATA_PATH");
    if (!dp) dp = ".";

    // Find training files
    char pattern[512];
    snprintf(pattern, sizeof(pattern), "%s/data/fineweb10B/fineweb_train_*.bin", dp);
    glob_t train_glob;
    if (glob(pattern, 0, NULL, &train_glob) != 0 || train_glob.gl_pathc == 0) {
        fprintf(stderr, "No training files found matching %s\n", pattern);
        exit(1);
    }
    printf("Found %zu training shards\n", train_glob.gl_pathc);

    snprintf(pattern, sizeof(pattern), "%s/data/fineweb10B/fineweb_val_*.bin", dp);
    glob_t val_glob;
    if (glob(pattern, 0, NULL, &val_glob) != 0 || val_glob.gl_pathc == 0) {
        fprintf(stderr, "No validation files found matching %s\n", pattern);
        exit(1);
    }

    // Load first training shard
    fprintf(stderr, "loading shard 0: %s\n", train_glob.gl_pathv[0]);
    DataShard train_shard = load_data_shard(train_glob.gl_pathv[0]);
    int current_shard = 0;

    fprintf(stderr, "Training shard 0: %lld tokens\n", (long long)train_shard.num_tokens);

    // Find BOS positions
    fprintf(stderr, "finding BOS positions...\n");
    int max_bos = 2000000;
    int64_t* bos_positions = (int64_t*)malloc(max_bos * sizeof(int64_t));
    int num_bos = find_bos_positions(train_shard.tokens, train_shard.num_tokens, bos_positions, max_bos);
    fprintf(stderr, "Found %d BOS positions\n", num_bos);

    // GPU buffers for batch data
    fprintf(stderr, "allocating GPU batch buffers...\n");
    int32_t* d_inputs;
    int64_t* d_targets;
    int32_t* d_cum_seqlens;
    int32_t* d_bigram_inputs;
    CUDA_CHECK(cudaMalloc(&d_inputs, max_tokens * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_targets, max_tokens * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_cum_seqlens, (max_tokens / 300 + 128) * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_bigram_inputs, max_tokens * sizeof(int32_t)));
    fprintf(stderr, "GPU batch buffers allocated\n");

    // Host batch buffer
    uint16_t* h_batch = (uint16_t*)malloc((max_tokens + 1) * sizeof(uint16_t));
    int32_t* h_cum_lengths = (int32_t*)malloc((max_tokens / 300 + 128) * sizeof(int32_t));

    // ---- Training loop ----
    fprintf(stderr, "Starting training for %d steps...\n", TOTAL_STEPS);

    struct timespec t0, t1;
    double training_time_ms = 0.0;

    int bos_idx = 0;
    int ws_short = STAGES[0].ws_short * BLOCK_SIZE_SCHED;
    int ws_long = STAGES[0].ws_long * BLOCK_SIZE_SCHED;
    int prev_ws_long = ws_long;
    int batch_size = STAGES[0].batch_size;
    int train_max_seq_len = STAGES[0].train_max_seq_len;

    CUDA_CHECK(cudaDeviceSynchronize());
    fprintf(stderr, "entering training loop\n");
    clock_gettime(CLOCK_MONOTONIC, &t0);

    float train_loss = 0.0f;
    for (int step = 0; step <= TOTAL_STEPS; step++) {
        int last_step = (step == TOTAL_STEPS);

        // Advance schedule
        int stage_idx;
        float stage_t;
        get_stage(step, &stage_idx, &stage_t);
        ws_short = STAGES[stage_idx].ws_short * BLOCK_SIZE_SCHED;
        int new_ws_long = STAGES[stage_idx].ws_long * BLOCK_SIZE_SCHED;

        // YaRN extension on window change
        if (new_ws_long != prev_ws_long) {
            yarn_apply_extension(&g_yarn, prev_ws_long, new_ws_long);
            yarn_apply_extension(&g_yarn_paired, prev_ws_long, new_ws_long);
            yarn_compute(&g_yarn, ctx.act.yarn_cos, ctx.act.yarn_sin, ctx.act.scratch_f32, ctx.stream);
            yarn_compute(&g_yarn_paired, ctx.act.yarn_paired_cos, ctx.act.yarn_paired_sin, ctx.act.scratch_f32, ctx.stream);
        }
        prev_ws_long = new_ws_long;
        ws_long = new_ws_long;
        batch_size = STAGES[stage_idx].batch_size;
        train_max_seq_len = STAGES[stage_idx].train_max_seq_len;

        // ---- Validation ----
        if (last_step || (VAL_LOSS_EVERY > 0 && step % VAL_LOSS_EVERY == 0)) {
            fprintf(stderr, "step %d: validation\n", step);
            if (last_step) {
                // Final validation: extend window to 20
                yarn_apply_extension(&g_yarn, ws_long, 20 * BLOCK_SIZE_SCHED);
                yarn_apply_extension(&g_yarn_paired, ws_long, 20 * BLOCK_SIZE_SCHED);
                yarn_compute(&g_yarn, ctx.act.yarn_cos, ctx.act.yarn_sin, ctx.act.scratch_f32, ctx.stream);
                yarn_compute(&g_yarn_paired, ctx.act.yarn_paired_cos, ctx.act.yarn_paired_sin, ctx.act.scratch_f32, ctx.stream);
                ws_long = 20 * BLOCK_SIZE_SCHED;
            }

            CUDA_CHECK(cudaDeviceSynchronize());
            clock_gettime(CLOCK_MONOTONIC, &t1);
            training_time_ms += (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;

            // Validation: run forward passes over VAL_TOKENS
            {
                // Cache scalars for validation forward passes
                CachedScalars val_cached_scalars;
                CUDA_CHECK(cudaMemcpy(val_cached_scalars.scalars, ctx.params.scalars,
                                      (2 * NUM_LAYERS + 3) * sizeof(bf16), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(val_cached_scalars.resid_lambdas, ctx.params.resid_lambdas,
                                      sizeof(val_cached_scalars.resid_lambdas), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(val_cached_scalars.post_lambdas, ctx.params.post_lambdas,
                                      sizeof(val_cached_scalars.post_lambdas), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(val_cached_scalars.x0_lambdas, ctx.params.x0_lambdas,
                                      sizeof(val_cached_scalars.x0_lambdas), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(val_cached_scalars.bigram_lambdas, ctx.params.bigram_lambdas,
                                      sizeof(val_cached_scalars.bigram_lambdas), cudaMemcpyDeviceToHost));

                float val_loss_sum = 0.0f;
                int val_tokens_seen = 0;
                int val_batch = max_tokens;  // Use training-sized chunks (49152) to fit in GPU memory
                int val_shard_idx = 0;
                DataShard val_shard = load_data_shard(val_glob.gl_pathv[val_shard_idx]);
                int64_t* val_bos_pos = (int64_t*)malloc(max_bos * sizeof(int64_t));
                int val_num_bos = find_bos_positions(val_shard.tokens, val_shard.num_tokens, val_bos_pos, max_bos);
                int val_bos_idx = 0;

                fprintf(stderr, "  val: batch=%d, val_tokens=%d\n", val_batch, VAL_TOKENS);
                while (val_tokens_seen < VAL_TOKENS) {
                    int val_num_seqs = 0;
                    int got = construct_aligned_batch(
                        val_shard.tokens, val_shard.num_tokens,
                        val_bos_pos, val_num_bos, &val_bos_idx,
                        val_batch, 2048, h_batch, h_cum_lengths, val_batch / 300 + 128,
                        &val_num_seqs);
                    fprintf(stderr, "  val batch: got=%d, seen=%d\n", got, val_tokens_seen);

                    if (got < 0) {
                        // Advance to next val shard
                        val_shard_idx++;
                        if (val_shard_idx >= (int)val_glob.gl_pathc) {
                            val_shard_idx = 0; // wrap
                        }
                        unload_data_shard(&val_shard);
                        val_shard = load_data_shard(val_glob.gl_pathv[val_shard_idx]);
                        val_num_bos = find_bos_positions(val_shard.tokens, val_shard.num_tokens, val_bos_pos, max_bos);
                        val_bos_idx = 0;
                        continue;
                    }

                    int T_val = val_batch;
                    // Pad cum_lengths for cuDNN graph bucketing
                    int val_bucketed = bucket_num_seqs(val_num_seqs);
                    for (int j = val_num_seqs + 1; j <= val_bucketed; j++) {
                        h_cum_lengths[j] = T_val;
                    }
                    // Convert and upload
                    {
                        int32_t* h_inputs = (int32_t*)malloc(T_val * sizeof(int32_t));
                        int64_t* h_targets = (int64_t*)malloc(T_val * sizeof(int64_t));
                        for (int j = 0; j < T_val; j++) {
                            h_inputs[j] = (int32_t)h_batch[j];
                            h_targets[j] = (int64_t)h_batch[j + 1];
                        }
                        CUDA_CHECK(cudaMemcpyAsync(d_inputs, h_inputs, T_val * sizeof(int32_t), cudaMemcpyHostToDevice, ctx.stream));
                        CUDA_CHECK(cudaMemcpyAsync(d_targets, h_targets, T_val * sizeof(int64_t), cudaMemcpyHostToDevice, ctx.stream));
                        free(h_inputs);
                        free(h_targets);
                    }
                    bigram_hash(d_bigram_inputs, d_inputs, T_val, ctx.stream);
                    CUDA_CHECK(cudaMemcpyAsync(d_cum_seqlens, h_cum_lengths, (T_val / 300 + 128) * sizeof(int32_t), cudaMemcpyHostToDevice, ctx.stream));

                    float loss;
                    float eval_mtp[1] = {1.0f};
                    fprintf(stderr, "  val forward_pass (T=%d)...\n", T_val);
                    forward_pass(&ctx, d_inputs, d_targets, d_cum_seqlens, d_bigram_inputs,
                                T_val, val_num_seqs, &val_cached_scalars, eval_mtp, 1,
                                ws_short, ws_long, 2048,
                                0, &loss);
                    fprintf(stderr, "  val forward done, loss=%f\n", loss);

                    val_loss_sum += loss;
                    val_tokens_seen += T_val;
                }

                int n_batches = val_tokens_seen / val_batch;
                float mean_val_loss = (n_batches > 0) ? val_loss_sum / n_batches : 0.0f;
                printf("step:%d/%d val_loss:%.4f train_time:%.0fms step_avg:%.2fms\n",
                       step, TOTAL_STEPS, mean_val_loss, training_time_ms,
                       step > 0 ? training_time_ms / step : 0.0);

                unload_data_shard(&val_shard);
                free(val_bos_pos);
            }

            CUDA_CHECK(cudaDeviceSynchronize());
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }

        if (last_step) break;

        // ---- Training step ----
        ctx.current_step = step;
        float mtp_weights[3];
        int n_mtp;
        get_mtp_weights(step, mtp_weights, &n_mtp);

        // NorMuon grads always zeroed (attn_bank, mlp_bank update every step)
        // Adam grads only zeroed on even steps — on odd steps they accumulate
        // with the previous even step's grads (Python clears Adam grads only
        // after odd-step optimizer updates)
        zero_normuon_grads(&ctx.grads, ctx.stream);
        if (step % 2 == 0) {
            zero_adam_grads(&ctx.grads, ctx.stream);
        }

        // Cache all scalar parameters from GPU once per step (constant across microsteps)
        CachedScalars cached_scalars;
        CUDA_CHECK(cudaMemcpy(cached_scalars.scalars, ctx.params.scalars,
                              (2 * NUM_LAYERS + 3) * sizeof(bf16), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(cached_scalars.resid_lambdas, ctx.params.resid_lambdas,
                              sizeof(cached_scalars.resid_lambdas), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(cached_scalars.post_lambdas, ctx.params.post_lambdas,
                              sizeof(cached_scalars.post_lambdas), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(cached_scalars.x0_lambdas, ctx.params.x0_lambdas,
                              sizeof(cached_scalars.x0_lambdas), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(cached_scalars.bigram_lambdas, ctx.params.bigram_lambdas,
                              sizeof(cached_scalars.bigram_lambdas), cudaMemcpyDeviceToHost));

        int num_tokens_local = batch_size / GRAD_ACCUM_STEPS;
        int32_t* h_inputs = (int32_t*)malloc(num_tokens_local * sizeof(int32_t));
        int64_t* h_targets = (int64_t*)malloc(num_tokens_local * sizeof(int64_t));

        for (int accum = 0; accum < GRAD_ACCUM_STEPS; accum++) {
            // Construct batch
            int train_num_seqs = 0;
            int got = construct_aligned_batch(
                train_shard.tokens, train_shard.num_tokens,
                bos_positions, num_bos, &bos_idx,
                num_tokens_local, train_max_seq_len,
                h_batch, h_cum_lengths, num_tokens_local / 300 + 128,
                &train_num_seqs);

            if (got < 0) {
                // Load next shard (wrap around if all exhausted)
                current_shard++;
                if (current_shard >= (int)train_glob.gl_pathc) {
                    current_shard = 0;  // wrap around to first shard
                }
                unload_data_shard(&train_shard);
                train_shard = load_data_shard(train_glob.gl_pathv[current_shard]);
                num_bos = find_bos_positions(train_shard.tokens, train_shard.num_tokens, bos_positions, max_bos);
                bos_idx = 0;
                accum--;  // retry this accumulation step
                continue;
            }

            int T = num_tokens_local;

            // Pad cum_lengths for cuDNN graph caching (bucket num_seqs to power of 2)
            int bucketed_seqs = bucket_num_seqs(train_num_seqs);
            for (int i = train_num_seqs + 1; i <= bucketed_seqs; i++) {
                h_cum_lengths[i] = T;  // padded sequences start at end of data (zero length)
            }

            // Convert to int32 inputs and int64 targets, upload to GPU
            for (int i = 0; i < T; i++) {
                h_inputs[i] = (int32_t)h_batch[i];
                h_targets[i] = (int64_t)h_batch[i + 1];
            }
            CUDA_CHECK(cudaMemcpyAsync(d_inputs, h_inputs, T * sizeof(int32_t), cudaMemcpyHostToDevice, ctx.stream));
            CUDA_CHECK(cudaMemcpyAsync(d_targets, h_targets, T * sizeof(int64_t), cudaMemcpyHostToDevice, ctx.stream));

            // Compute bigram hash on GPU
            bigram_hash(d_bigram_inputs, d_inputs, T, ctx.stream);

            // Upload cum_lengths
            CUDA_CHECK(cudaMemcpyAsync(d_cum_seqlens, h_cum_lengths, (T / 300 + 128) * sizeof(int32_t), cudaMemcpyHostToDevice, ctx.stream));

            // Forward pass (only compute loss for first microstep)
            float loss = 0.0f;
            forward_pass(&ctx, d_inputs, d_targets, d_cum_seqlens, d_bigram_inputs,
                        T, train_num_seqs, &cached_scalars, mtp_weights, n_mtp,
                        ws_short, ws_long, train_max_seq_len,
                        1, (accum == 0) ? &loss : NULL);
            if (accum == 0) train_loss = loss;

            // Backward pass
            backward_pass(&ctx, d_inputs, d_targets, d_cum_seqlens, d_bigram_inputs,
                         T, train_num_seqs, &cached_scalars, mtp_weights, n_mtp,
                         ws_short, ws_long, train_max_seq_len);
        }

        free(h_inputs);
        free(h_targets);

        // Optimizer step
        int do_adam = (step % 2 == 1);
        optimizer_step(&ctx, step, do_adam);

        // Logging (every step for debugging)
        {
            CUDA_CHECK(cudaDeviceSynchronize());
            clock_gettime(CLOCK_MONOTONIC, &t1);
            double elapsed = training_time_ms + (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
            printf("step:%d/%d train_loss:%.4f train_time:%.0fms step_avg:%.2fms\n",
                   step + 1, TOTAL_STEPS, train_loss, elapsed, elapsed / (step + 1));
        }
    }

    // Cleanup
    printf("Training complete.\n");
    unload_data_shard(&train_shard);
    globfree(&train_glob);
    globfree(&val_glob);
    free(bos_positions);
    free(h_batch);
    free(h_cum_lengths);
    CUDA_CHECK(cudaFree(d_inputs));
    CUDA_CHECK(cudaFree(d_targets));
    CUDA_CHECK(cudaFree(d_cum_seqlens));
    CUDA_CHECK(cudaFree(d_bigram_inputs));

    free_training_context(&ctx);

    return 0;
}
