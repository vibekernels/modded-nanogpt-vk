// kernels.cu - Custom CUDA kernels for GPT training
// Ports of triton_kernels.py + additional utility kernels
#include "kernels.h"
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <float.h>

namespace cg = cooperative_groups;

// ============================================================================
// Helper device functions
// ============================================================================

__device__ __forceinline__ float bf16_to_f(bf16 x) { return __bfloat162float(x); }
__device__ __forceinline__ bf16 f_to_bf16(float x) { return __float2bfloat16(x); }

// Warp-level reduction
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    return val;
}

// Block-level reduction using shared memory
__device__ float block_reduce_sum(float val) {
    __shared__ float shared[32]; // one per warp
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    val = warp_reduce_sum(val);
    if (lane == 0) shared[warp_id] = val;
    __syncthreads();

    int num_warps = (blockDim.x + 31) / 32;
    val = (threadIdx.x < num_warps) ? shared[threadIdx.x] : 0.0f;
    if (warp_id == 0) val = warp_reduce_sum(val);
    return val;
}

__device__ float block_reduce_max(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    val = warp_reduce_max(val);
    if (lane == 0) shared[warp_id] = val;
    __syncthreads();

    int num_warps = (blockDim.x + 31) / 32;
    val = (threadIdx.x < num_warps) ? shared[threadIdx.x] : -FLT_MAX;
    if (warp_id == 0) val = warp_reduce_max(val);
    return val;
}

// ============================================================================
// RMS Norm
// ============================================================================

__global__ void rms_norm_fwd_kernel(bf16* __restrict__ out,
                                     const bf16* __restrict__ x,
                                     int cols) {
    int row = blockIdx.x;
    const bf16* x_row = x + row * cols;
    bf16* out_row = out + row * cols;

    // Compute sum of squares
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float val = bf16_to_f(x_row[i]);
        sum_sq += val * val;
    }
    sum_sq = block_reduce_sum(sum_sq);

    __shared__ float s_rms;
    if (threadIdx.x == 0) {
        s_rms = rsqrtf(sum_sq / cols + 1e-6f);
    }
    __syncthreads();
    float rms = s_rms;

    // Normalize
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float val = bf16_to_f(x_row[i]);
        out_row[i] = f_to_bf16(val * rms);
    }
}

void rms_norm_fwd(bf16* out, const bf16* x, int rows, int cols, cudaStream_t stream) {
    int threads = min(1024, cdiv(cols, 1) );
    // Use at least enough threads to cover cols
    threads = min(1024, max(256, (cols + 31) / 32 * 32));
    rms_norm_fwd_kernel<<<rows, threads, 0, stream>>>(out, x, cols);
}

__global__ void rms_norm_bwd_kernel(bf16* __restrict__ grad_x,
                                     const bf16* __restrict__ grad_out,
                                     const bf16* __restrict__ x,
                                     int cols) {
    int row = blockIdx.x;
    const bf16* x_row = x + row * cols;
    const bf16* go_row = grad_out + row * cols;
    bf16* gx_row = grad_x + row * cols;

    // Compute rms
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float val = bf16_to_f(x_row[i]);
        sum_sq += val * val;
    }
    sum_sq = block_reduce_sum(sum_sq);

    __shared__ float s_rms_inv;
    if (threadIdx.x == 0) {
        s_rms_inv = rsqrtf(sum_sq / cols + 1e-6f);
    }
    __syncthreads();
    float rms_inv = s_rms_inv;

    // Compute dot product of grad_out and normed_x
    float dot = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float xi = bf16_to_f(x_row[i]);
        float gi = bf16_to_f(go_row[i]);
        dot += gi * xi * rms_inv;
    }
    dot = block_reduce_sum(dot);

    __shared__ float s_dot;
    if (threadIdx.x == 0) {
        s_dot = dot / cols;
    }
    __syncthreads();
    float dot_avg = s_dot;

    // Compute gradient
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float xi = bf16_to_f(x_row[i]);
        float gi = bf16_to_f(go_row[i]);
        float normed = xi * rms_inv;
        float gx = rms_inv * (gi - normed * dot_avg);
        gx_row[i] = f_to_bf16(gx);
    }
}

void rms_norm_bwd(bf16* grad_x, const bf16* grad_out, const bf16* x, int rows, int cols, cudaStream_t stream) {
    int threads = min(1024, max(256, (cols + 31) / 32 * 32));
    rms_norm_bwd_kernel<<<rows, threads, 0, stream>>>(grad_x, grad_out, x, cols);
}

// ============================================================================
// XXT kernel: C = A @ A.T (symmetric, upper triangle + mirror)
// ============================================================================

// Tile sizes for XXT/XTX
#define TILE_M 64
#define TILE_N 64
#define TILE_K 32

__global__ void xxt_kernel(const bf16* __restrict__ A, bf16* __restrict__ C,
                           int M, int K,
                           int a_stride_b, int a_stride_r, int a_stride_c,
                           int c_stride_b, int c_stride_r, int c_stride_c) {
    // 2D grid: blockIdx.x covers (M/TILE_M * M/TILE_N), blockIdx.y = batch
    int batch = blockIdx.y;
    int num_n = (M + TILE_N - 1) / TILE_N;
    int pid_m = blockIdx.x / num_n;
    int pid_n = blockIdx.x % num_n;

    int m_start = pid_m * TILE_M;
    int n_start = pid_n * TILE_N;

    // Only compute upper triangle (including diagonal)
    if (m_start > n_start) return;

    const bf16* A_batch = A + batch * a_stride_b;
    bf16* C_batch = C + batch * c_stride_b;

    // Shared memory for tiles
    __shared__ bf16 As[TILE_M][TILE_K + 1];  // +1 to avoid bank conflicts
    __shared__ bf16 Bs[TILE_N][TILE_K + 1];

    float acc[4][4] = {};  // Each thread computes a 4x4 sub-tile
    int tx = threadIdx.x % 16;  // 16 threads across N
    int ty = threadIdx.x / 16;  // 16 threads across M

    for (int k = 0; k < K; k += TILE_K) {
        // Load A[m_start:m_start+TILE_M, k:k+TILE_K] into shared memory
        for (int i = threadIdx.x; i < TILE_M * TILE_K; i += blockDim.x) {
            int r = i / TILE_K;
            int c = i % TILE_K;
            int global_r = m_start + r;
            int global_c = k + c;
            As[r][c] = (global_r < M && global_c < K)
                ? A_batch[global_r * a_stride_r + global_c * a_stride_c]
                : f_to_bf16(0.0f);
        }
        // Load A[n_start:n_start+TILE_N, k:k+TILE_K] for A.T part
        for (int i = threadIdx.x; i < TILE_N * TILE_K; i += blockDim.x) {
            int r = i / TILE_K;
            int c = i % TILE_K;
            int global_r = n_start + r;
            int global_c = k + c;
            Bs[r][c] = (global_r < M && global_c < K)
                ? A_batch[global_r * a_stride_r + global_c * a_stride_c]
                : f_to_bf16(0.0f);
        }
        __syncthreads();

        // Compute partial dot products
        #pragma unroll
        for (int ki = 0; ki < TILE_K; ki++) {
            float a_vals[4], b_vals[4];
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                a_vals[i] = bf16_to_f(As[ty * 4 + i][ki]);
                b_vals[i] = bf16_to_f(Bs[tx * 4 + i][ki]);
            }
            #pragma unroll
            for (int i = 0; i < 4; i++)
                #pragma unroll
                for (int j = 0; j < 4; j++)
                    acc[i][j] += a_vals[i] * b_vals[j];
        }
        __syncthreads();
    }

    // Store results (upper triangle)
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int row = m_start + ty * 4 + i;
        if (row >= M) continue;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int col = n_start + tx * 4 + j;
            if (col >= M) continue;
            bf16 val = f_to_bf16(acc[i][j]);
            C_batch[row * c_stride_r + col * c_stride_c] = val;
            // Mirror across diagonal
            if (row != col) {
                C_batch[col * c_stride_r + row * c_stride_c] = val;
            }
        }
    }
}

void xxt_kernel_launch(const bf16* A, bf16* C,
                       int batch_size, int M, int K,
                       int a_stride_b, int a_stride_r, int a_stride_c,
                       int c_stride_b, int c_stride_r, int c_stride_c,
                       cudaStream_t stream) {
    int num_m = cdiv(M, TILE_M);
    int num_n = cdiv(M, TILE_N);
    dim3 grid(num_m * num_n, batch_size);
    dim3 block(256);
    xxt_kernel<<<grid, block, 0, stream>>>(A, C, M, K,
        a_stride_b, a_stride_r, a_stride_c,
        c_stride_b, c_stride_r, c_stride_c);
}

// ============================================================================
// XTX kernel: C = A.T @ A where A is [M, K], C is [K, K]
// ============================================================================

__global__ void xtx_kernel(const bf16* __restrict__ A, bf16* __restrict__ C,
                           int M, int K,
                           int a_stride_b, int a_stride_r, int a_stride_c,
                           int c_stride_b, int c_stride_r, int c_stride_c) {
    int batch = blockIdx.y;
    int num_n = (K + TILE_N - 1) / TILE_N;
    int pid_k = blockIdx.x / num_n;
    int pid_n = blockIdx.x % num_n;

    int k_start = pid_k * TILE_M;
    int n_start = pid_n * TILE_N;

    // Only compute upper triangle
    if (k_start > n_start) return;

    const bf16* A_batch = A + batch * a_stride_b;
    bf16* C_batch = C + batch * c_stride_b;

    __shared__ bf16 As[TILE_K + 1][TILE_M];  // A[:, k_start:k_start+TILE_M] transposed for coalesced access
    __shared__ bf16 Bs[TILE_K + 1][TILE_N];  // A[:, n_start:n_start+TILE_N]

    float acc[4][4] = {};
    int tx = threadIdx.x % 16;
    int ty = threadIdx.x / 16;

    for (int m = 0; m < M; m += TILE_K) {
        // Load A[m:m+TILE_K, k_start:k_start+TILE_M]
        for (int i = threadIdx.x; i < TILE_K * TILE_M; i += blockDim.x) {
            int r = i / TILE_M;  // reduction dim
            int c = i % TILE_M;  // output row dim
            int global_r = m + r;
            int global_c = k_start + c;
            As[r][c] = (global_r < M && global_c < K)
                ? A_batch[global_r * a_stride_r + global_c * a_stride_c]
                : f_to_bf16(0.0f);
        }
        for (int i = threadIdx.x; i < TILE_K * TILE_N; i += blockDim.x) {
            int r = i / TILE_N;
            int c = i % TILE_N;
            int global_r = m + r;
            int global_c = n_start + c;
            Bs[r][c] = (global_r < M && global_c < K)
                ? A_batch[global_r * a_stride_r + global_c * a_stride_c]
                : f_to_bf16(0.0f);
        }
        __syncthreads();

        // A.T @ A: sum over M dimension
        // C[k,n] = sum_m A[m,k] * A[m,n]
        #pragma unroll
        for (int mi = 0; mi < TILE_K; mi++) {
            float a_vals[4], b_vals[4];
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                a_vals[i] = bf16_to_f(As[mi][ty * 4 + i]);  // A[m, k]
                b_vals[i] = bf16_to_f(Bs[mi][tx * 4 + i]);  // A[m, n]
            }
            #pragma unroll
            for (int i = 0; i < 4; i++)
                #pragma unroll
                for (int j = 0; j < 4; j++)
                    acc[i][j] += a_vals[i] * b_vals[j];
        }
        __syncthreads();
    }

    // Store with symmetry
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int row = k_start + ty * 4 + i;
        if (row >= K) continue;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int col = n_start + tx * 4 + j;
            if (col >= K) continue;
            bf16 val = f_to_bf16(acc[i][j]);
            C_batch[row * c_stride_r + col * c_stride_c] = val;
            if (row != col)
                C_batch[col * c_stride_r + row * c_stride_c] = val;
        }
    }
}

void xtx_kernel_launch(const bf16* A, bf16* C,
                       int batch_size, int M, int K,
                       int a_stride_b, int a_stride_r, int a_stride_c,
                       int c_stride_b, int c_stride_r, int c_stride_c,
                       cudaStream_t stream) {
    int num_k = cdiv(K, TILE_M);
    int num_n = cdiv(K, TILE_N);
    dim3 grid(num_k * num_n, batch_size);
    dim3 block(256);
    xtx_kernel<<<grid, block, 0, stream>>>(A, C, M, K,
        a_stride_b, a_stride_r, a_stride_c,
        c_stride_b, c_stride_r, c_stride_c);
}

// ============================================================================
// ba + cAA kernel: C = alpha * A @ A.T + beta * A (for square matrices)
// ============================================================================

__global__ void ba_plus_caa_kernel(const bf16* __restrict__ A, bf16* __restrict__ C,
                                   int M,
                                   int a_stride_b, int a_stride_r, int a_stride_c,
                                   int c_stride_b, int c_stride_r, int c_stride_c,
                                   float alpha, float beta) {
    int batch = blockIdx.y;
    int num_n = (M + TILE_N - 1) / TILE_N;
    int pid_m = blockIdx.x / num_n;
    int pid_n = blockIdx.x % num_n;

    int m_start = pid_m * TILE_M;
    int n_start = pid_n * TILE_N;

    if (m_start > n_start) return;

    const bf16* A_batch = A + batch * a_stride_b;
    bf16* C_batch = C + batch * c_stride_b;

    __shared__ bf16 As[TILE_M][TILE_K + 1];
    __shared__ bf16 Bs[TILE_N][TILE_K + 1];

    float acc[4][4] = {};
    int tx = threadIdx.x % 16;
    int ty = threadIdx.x / 16;

    // Compute A @ A.T portion
    for (int k = 0; k < M; k += TILE_K) {
        for (int i = threadIdx.x; i < TILE_M * TILE_K; i += blockDim.x) {
            int r = i / TILE_K;
            int c = i % TILE_K;
            int global_r = m_start + r;
            int global_c = k + c;
            As[r][c] = (global_r < M && global_c < M)
                ? A_batch[global_r * a_stride_r + global_c * a_stride_c]
                : f_to_bf16(0.0f);
        }
        for (int i = threadIdx.x; i < TILE_N * TILE_K; i += blockDim.x) {
            int r = i / TILE_K;
            int c = i % TILE_K;
            int global_r = n_start + r;
            int global_c = k + c;
            Bs[r][c] = (global_r < M && global_c < M)
                ? A_batch[global_r * a_stride_r + global_c * a_stride_c]
                : f_to_bf16(0.0f);
        }
        __syncthreads();

        #pragma unroll
        for (int ki = 0; ki < TILE_K; ki++) {
            float a_vals[4], b_vals[4];
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                a_vals[i] = bf16_to_f(As[ty * 4 + i][ki]);
                b_vals[i] = bf16_to_f(Bs[tx * 4 + i][ki]);
            }
            #pragma unroll
            for (int i = 0; i < 4; i++)
                #pragma unroll
                for (int j = 0; j < 4; j++)
                    acc[i][j] += a_vals[i] * b_vals[j];
        }
        __syncthreads();
    }

    // Add beta * A and store
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int row = m_start + ty * 4 + i;
        if (row >= M) continue;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int col = n_start + tx * 4 + j;
            if (col >= M) continue;
            // Load A[row, col]
            float a_val = bf16_to_f(A_batch[row * a_stride_r + col * a_stride_c]);
            float result = alpha * acc[i][j] + beta * a_val;
            bf16 val = f_to_bf16(result);
            C_batch[row * c_stride_r + col * c_stride_c] = val;
            if (row != col)
                C_batch[col * c_stride_r + row * c_stride_c] = val;
        }
    }
}

void ba_plus_caa_kernel_launch(const bf16* A, bf16* C,
                               int batch_size, int M,
                               int a_stride_b, int a_stride_r, int a_stride_c,
                               int c_stride_b, int c_stride_r, int c_stride_c,
                               float alpha, float beta,
                               cudaStream_t stream) {
    int num_m = cdiv(M, TILE_M);
    int num_n = cdiv(M, TILE_N);
    dim3 grid(num_m * num_n, batch_size);
    dim3 block(256);
    ba_plus_caa_kernel<<<grid, block, 0, stream>>>(A, C, M,
        a_stride_b, a_stride_r, a_stride_c,
        c_stride_b, c_stride_r, c_stride_c,
        alpha, beta);
}

// ============================================================================
// Fused linear + ReLU² (forward and backward)
// ============================================================================

// Forward: computes pre = x @ W1.T, post = relu(pre)²
// Note: The Triton kernel does a deinterleaving trick for TMA. We use a simpler approach.
__global__ void linear_relu_square_fwd_kernel(const bf16* __restrict__ x,
                                              const bf16* __restrict__ W1,
                                              bf16* __restrict__ pre,
                                              bf16* __restrict__ post,
                                              int M, int N, int K) {
    // Each thread computes one output element
    int row = blockIdx.x;
    int col = blockIdx.y * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        sum += bf16_to_f(x[row * K + k]) * bf16_to_f(W1[col * K + k]);  // W1 is [N, K]
    }

    bf16 pre_val = f_to_bf16(sum);
    pre[row * N + col] = pre_val;

    // relu(x)²
    float f = fmaxf(sum, 0.0f);
    post[row * N + col] = f_to_bf16(f * f);
}

void linear_relu_square_fwd(const bf16* x, const bf16* W1, bf16* pre, bf16* post,
                            int M, int N, int K, cudaStream_t stream) {
    // Fallback: fused matmul + activation (slow, use cuBLAS + relu_square_fwd instead)
    dim3 block(256);
    dim3 grid(M, cdiv(N, 256));
    linear_relu_square_fwd_kernel<<<grid, block, 0, stream>>>(x, W1, pre, post, M, N, K);
}

// Pure relu² activation: post[i] = max(pre[i], 0)^2
__global__ void relu_square_fwd_kernel(const bf16* __restrict__ pre,
                                        bf16* __restrict__ post, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float val = fmaxf(bf16_to_f(pre[idx]), 0.0f);
    post[idx] = f_to_bf16(val * val);
}

void relu_square_fwd(const bf16* pre, bf16* post, int n, cudaStream_t stream) {
    int threads = 256;
    relu_square_fwd_kernel<<<cdiv(n, threads), threads, 0, stream>>>(pre, post, n);
}

// Backward: grad_input = 2 * grad * relu_mask(pre) * pre
__global__ void linear_relu_square_bwd_kernel(const bf16* __restrict__ grad_out,
                                              const bf16* __restrict__ pre,
                                              bf16* __restrict__ out,
                                              int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float g = bf16_to_f(grad_out[idx]);
    float p = bf16_to_f(pre[idx]);
    // d/dx [relu(x)²] = 2 * relu(x) * (x > 0) = 2 * max(x, 0) * (x > 0) = 2 * max(x, 0)
    float relu_val = fmaxf(p, 0.0f);
    out[idx] = f_to_bf16(2.0f * g * relu_val);
}

void linear_relu_square_bwd(const bf16* grad_out, const bf16* pre, bf16* out,
                            int M, int N, cudaStream_t stream) {
    int n = M * N;
    int threads = 256;
    linear_relu_square_bwd_kernel<<<cdiv(n, threads), threads, 0, stream>>>(grad_out, pre, out, n);
}

// ============================================================================
// Softcapped Cross Entropy Loss
// ============================================================================

__global__ void softcapped_ce_fwd_kernel(const bf16* __restrict__ logits,
                                          float* __restrict__ losses,
                                          float* __restrict__ lse,
                                          const int64_t* __restrict__ targets,
                                          const float* __restrict__ mtp_weights,
                                          int n_rows, int n_cols, int n_predict,
                                          float A, float B, float C_val) {
    int row = blockIdx.x;
    if (row >= n_rows) return;

    const bf16* logits_row = logits + (int64_t)row * n_cols;
    float inv_C = 1.0f / C_val;
    float B_div_C = B * inv_C;

    // Pass 1: compute log-sum-exp
    float max_val = -FLT_MAX;
    for (int col = threadIdx.x; col < n_cols; col += blockDim.x) {
        float val = bf16_to_f(logits_row[col]);
        float u = val * inv_C + B_div_C;
        float sigmoid_u = 1.0f / (1.0f + expf(-u));
        float z = A * sigmoid_u;
        max_val = fmaxf(max_val, z);
    }
    max_val = block_reduce_max(max_val);

    __shared__ float s_max;
    if (threadIdx.x == 0) s_max = max_val;
    __syncthreads();
    max_val = s_max;

    float sum_exp = 0.0f;
    for (int col = threadIdx.x; col < n_cols; col += blockDim.x) {
        float val = bf16_to_f(logits_row[col]);
        float u = val * inv_C + B_div_C;
        float sigmoid_u = 1.0f / (1.0f + expf(-u));
        float z = A * sigmoid_u;
        sum_exp += expf(z - max_val);
    }
    sum_exp = block_reduce_sum(sum_exp);

    __shared__ float s_lse;
    if (threadIdx.x == 0) {
        s_lse = max_val + logf(sum_exp);
        lse[row] = s_lse;
    }
    __syncthreads();
    float row_lse = s_lse;

    // Pass 2: compute loss (only thread 0)
    if (threadIdx.x == 0) {
        float total_loss = 0.0f;
        for (int k = 0; k < n_predict; k++) {
            int target_idx = row + k;
            if (target_idx < n_rows) {
                float weight = mtp_weights[k];
                if (weight > 0.0f) {
                    int target = (int)targets[target_idx];
                    if (target >= 0 && target < n_cols) {
                        float val_target = bf16_to_f(logits_row[target]);
                        float u_t = val_target * inv_C + B_div_C;
                        float z_target = A / (1.0f + expf(-u_t));
                        total_loss += weight * (row_lse - z_target);
                    }
                }
            }
        }
        losses[row] = total_loss;
    }
}

void softcapped_ce_fwd(const bf16* logits, float* losses, float* lse,
                       const int64_t* targets, const float* mtp_weights,
                       int n_rows, int n_cols, int n_predict,
                       float A, float B, float C,
                       cudaStream_t stream) {
    // Use enough threads for the reduction across vocab
    int threads = min(1024, max(256, (n_cols + 31) / 32 * 32));
    softcapped_ce_fwd_kernel<<<n_rows, threads, 0, stream>>>(
        logits, losses, lse, targets, mtp_weights,
        n_rows, n_cols, n_predict, A, B, C);
}

__global__ void softcapped_ce_bwd_kernel(fp8e5m2* __restrict__ grad_input,
                                          const float* __restrict__ grad_output,
                                          const float* __restrict__ lse,
                                          const bf16* __restrict__ logits,
                                          const int64_t* __restrict__ targets,
                                          const float* __restrict__ mtp_weights,
                                          int n_rows, int n_cols, int n_predict,
                                          float A, float B, float C_val,
                                          float grad_s) {
    int row = blockIdx.x;
    if (row >= n_rows) return;

    const bf16* logits_row = logits + (int64_t)row * n_cols;
    fp8e5m2* grad_row = grad_input + (int64_t)row * n_cols;

    float row_lse = lse[row];
    float grad_loss = grad_output[row];
    float inv_C = 1.0f / C_val;
    float B_div_C = B * inv_C;
    float inv_C_A = inv_C * A;

    // Sum of MTP weights for this row
    float S_w = 0.0f;
    for (int k = 0; k < n_predict; k++) {
        if (row + k < n_rows) S_w += mtp_weights[k];
    }

    for (int col = threadIdx.x; col < n_cols; col += blockDim.x) {
        float val = bf16_to_f(logits_row[col]);
        float u = val * inv_C + B_div_C;
        float sigmoid_u = 1.0f / (1.0f + expf(-u));
        float z = A * sigmoid_u;
        float p = expf(z - row_lse);

        float term1 = S_w * p;
        float term2 = 0.0f;
        for (int k = 0; k < n_predict; k++) {
            if (row + k < n_rows) {
                int target = (int)targets[row + k];
                float weight = mtp_weights[k];
                if (col == target) term2 += weight;
            }
        }

        float grad_z = grad_loss * (term1 - term2);
        float dz_dx = inv_C_A * sigmoid_u * (1.0f - sigmoid_u);
        float grad_x = grad_z * dz_dx / grad_s;

        // Convert to FP8 e5m2
        grad_row[col] = __nv_fp8_e5m2(grad_x);
    }
}

void softcapped_ce_bwd(fp8e5m2* grad_input, const float* grad_output, const float* lse,
                       const bf16* logits, const int64_t* targets, const float* mtp_weights,
                       int n_rows, int n_cols, int n_predict,
                       float A, float B, float C, float grad_s,
                       cudaStream_t stream) {
    int threads = min(1024, max(256, (n_cols + 31) / 32 * 32));
    softcapped_ce_bwd_kernel<<<n_rows, threads, 0, stream>>>(
        grad_input, grad_output, lse, logits, targets, mtp_weights,
        n_rows, n_cols, n_predict, A, B, C, grad_s);
}

// BF16 variant of softcapped CE backward (no FP8 output)
__global__ void softcapped_ce_bwd_bf16_kernel(bf16* __restrict__ grad_input,
                                               const float* __restrict__ grad_output,
                                               const float* __restrict__ lse,
                                               const bf16* __restrict__ logits,
                                               const int64_t* __restrict__ targets,
                                               const float* __restrict__ mtp_weights,
                                               int n_rows, int n_cols, int n_predict,
                                               float A, float B, float C_val,
                                               float grad_s) {
    int row = blockIdx.x;
    if (row >= n_rows) return;

    const bf16* logits_row = logits + (int64_t)row * n_cols;
    bf16* grad_row = grad_input + (int64_t)row * n_cols;

    float row_lse = lse[row];
    float grad_loss = grad_output[row];
    float inv_C = 1.0f / C_val;
    float B_div_C = B * inv_C;
    float inv_C_A = inv_C * A;

    float S_w = 0.0f;
    for (int k = 0; k < n_predict; k++) {
        if (row + k < n_rows) S_w += mtp_weights[k];
    }

    for (int col = threadIdx.x; col < n_cols; col += blockDim.x) {
        float val = bf16_to_f(logits_row[col]);
        float u = val * inv_C + B_div_C;
        float sigmoid_u = 1.0f / (1.0f + expf(-u));
        float z = A * sigmoid_u;
        float p = expf(z - row_lse);

        float term1 = S_w * p;
        float term2 = 0.0f;
        for (int k = 0; k < n_predict; k++) {
            if (row + k < n_rows) {
                int target = (int)targets[row + k];
                float weight = mtp_weights[k];
                if (col == target) term2 += weight;
            }
        }

        float grad_z = grad_loss * (term1 - term2);
        float dz_dx = inv_C_A * sigmoid_u * (1.0f - sigmoid_u);
        float grad_x = grad_z * dz_dx / grad_s;

        grad_row[col] = f_to_bf16(grad_x);
    }
}

void softcapped_ce_bwd_bf16(bf16* grad_input, const float* grad_output, const float* lse,
                             const bf16* logits, const int64_t* targets, const float* mtp_weights,
                             int n_rows, int n_cols, int n_predict,
                             float A, float B, float C, float grad_s,
                             cudaStream_t stream) {
    int threads = min(1024, max(256, (n_cols + 31) / 32 * 32));
    softcapped_ce_bwd_bf16_kernel<<<n_rows, threads, 0, stream>>>(
        grad_input, grad_output, lse, logits, targets, mtp_weights,
        n_rows, n_cols, n_predict, A, B, C, grad_s);
}

// ============================================================================
// Transpose kernels (32x32 tiled)
// ============================================================================

#define TRANSPOSE_TILE 32

__global__ void transpose_copy_kernel(const bf16* __restrict__ src,
                                       bf16* __restrict__ dst,
                                       int M, int N) {
    __shared__ bf16 tile[TRANSPOSE_TILE][TRANSPOSE_TILE + 1]; // +1 avoids bank conflicts

    int bx = blockIdx.x * TRANSPOSE_TILE;
    int by = blockIdx.y * TRANSPOSE_TILE;

    // Read src[M, N] tile (coalesced along N)
    int x = bx + threadIdx.x;
    int y = by + threadIdx.y;
    if (x < M && y < N)
        tile[threadIdx.x][threadIdx.y] = src[x * N + y];
    __syncthreads();

    // Write dst[N, M] tile (coalesced along M)
    // dst[y, x] = src[x, y] = tile[threadIdx.x][threadIdx.y]
    // But we need to map: dst[(by + threadIdx.x), (bx + threadIdx.y)] = tile[threadIdx.y][threadIdx.x]
    x = by + threadIdx.x;  // dst row
    y = bx + threadIdx.y;  // dst col
    if (x < N && y < M)
        dst[x * M + y] = tile[threadIdx.y][threadIdx.x];
}

void transpose_copy(const bf16* src, bf16* dst, int M, int N, cudaStream_t stream) {
    dim3 block(TRANSPOSE_TILE, TRANSPOSE_TILE);
    dim3 grid(cdiv(M, TRANSPOSE_TILE), cdiv(N, TRANSPOSE_TILE));
    transpose_copy_kernel<<<grid, block, 0, stream>>>(src, dst, M, N);
}

__global__ void transpose_add_kernel(const bf16* __restrict__ src,
                                      bf16* __restrict__ dst,
                                      int M, int N) {
    __shared__ bf16 tile[TRANSPOSE_TILE][TRANSPOSE_TILE + 1];

    int bx = blockIdx.x * TRANSPOSE_TILE;
    int by = blockIdx.y * TRANSPOSE_TILE;

    // Read src[M, N]
    int x = bx + threadIdx.x;
    int y = by + threadIdx.y;
    if (x < M && y < N)
        tile[threadIdx.x][threadIdx.y] = src[x * N + y];
    __syncthreads();

    // dst[N, M] += src.T -> dst[y, x] += src[x, y]
    x = by + threadIdx.x;
    y = bx + threadIdx.y;
    if (x < N && y < M) {
        float existing = bf16_to_f(dst[x * M + y]);
        float to_add = bf16_to_f(tile[threadIdx.y][threadIdx.x]);
        dst[x * M + y] = f_to_bf16(existing + to_add);
    }
}

void transpose_add(const bf16* src, bf16* dst, int M, int N, cudaStream_t stream) {
    dim3 block(TRANSPOSE_TILE, TRANSPOSE_TILE);
    dim3 grid(cdiv(M, TRANSPOSE_TILE), cdiv(N, TRANSPOSE_TILE));
    transpose_add_kernel<<<grid, block, 0, stream>>>(src, dst, M, N);
}

// ============================================================================
// RoPE (Rotary Position Embeddings)
// ============================================================================

__global__ void rope_apply_kernel(bf16* __restrict__ x,
                                   const bf16* __restrict__ cos_table,
                                   const bf16* __restrict__ sin_table,
                                   int T, int num_heads, int head_dim) {
    // Grid: [T, num_heads], Block: [head_dim/2]
    int t = blockIdx.x;
    int h = blockIdx.y;
    int d = threadIdx.x;
    if (t >= T || h >= num_heads || d >= head_dim / 2) return;

    int idx_base = t * num_heads * head_dim + h * head_dim;

    // x_rot = cos * x + sin * flip(x)
    // flip pairs: (x[2i], x[2i+1]) -> (x[2i+1], -x[2i]) when sin[2i+1] has negation
    // Actually the Python does: x_flip = x.view(..., hd//2, 2).flip(-1).view(x.shape)
    // So x_flip[2i] = x[2i+1], x_flip[2i+1] = x[2i]
    // Then: out[2i] = cos[2i]*x[2i] + sin[2i]*x[2i+1]
    //        out[2i+1] = cos[2i+1]*x[2i+1] + sin[2i+1]*x[2i]
    // And sin has factor2[..., 1::2] *= -1 applied, so sin[2i+1] is negated

    int idx0 = idx_base + 2 * d;
    int idx1 = idx_base + 2 * d + 1;

    float x0 = bf16_to_f(x[idx0]);
    float x1 = bf16_to_f(x[idx1]);
    float c0 = bf16_to_f(cos_table[t * head_dim + 2 * d]);
    float c1 = bf16_to_f(cos_table[t * head_dim + 2 * d + 1]);
    float s0 = bf16_to_f(sin_table[t * head_dim + 2 * d]);
    float s1 = bf16_to_f(sin_table[t * head_dim + 2 * d + 1]);

    // flip: swap x0 and x1
    x[idx0] = f_to_bf16(c0 * x0 + s0 * x1);
    x[idx1] = f_to_bf16(c1 * x1 + s1 * x0);
}

void rope_apply(bf16* x, const bf16* cos_table, const bf16* sin_table,
                int T, int num_heads, int head_dim, cudaStream_t stream) {
    dim3 grid(T, num_heads);
    dim3 block(head_dim / 2);
    rope_apply_kernel<<<grid, block, 0, stream>>>(x, cos_table, sin_table, T, num_heads, head_dim);
}

// Compute YaRN cos/sin tables
__global__ void yarn_compute_tables_kernel(bf16* __restrict__ cos_out,
                                            bf16* __restrict__ sin_out,
                                            const float* __restrict__ angular_freq,
                                            int head_dim, int max_seq_len, int paired) {
    int t = blockIdx.x;
    int d = threadIdx.x;
    if (t >= 2 * max_seq_len || d >= head_dim) return;

    float freq = angular_freq[d];
    float time_val;

    if (!paired) {
        time_val = (float)t;
        float theta = time_val * freq;
        cos_out[t * head_dim + d] = f_to_bf16(cosf(theta));
        float s = sinf(theta);
        // factor2[..., 1::2] *= -1
        if (d % 2 == 1) s = -s;
        sin_out[t * head_dim + d] = f_to_bf16(s);
    } else {
        // Paired: interleave even/odd time steps
        // cos_out has width 2*head_dim: [cos(t_even), cos(t_odd)]
        float t_even = 2.0f * t;
        float t_odd = 2.0f * t + 1.0f;

        if (d < head_dim) {
            // Even part
            float theta_even = t_even * freq;
            cos_out[t * 2 * head_dim + d] = f_to_bf16(cosf(theta_even));
            float s = sinf(theta_even);
            if (d % 2 == 1) s = -s;
            sin_out[t * 2 * head_dim + d] = f_to_bf16(s);
        }
        // Odd part at offset head_dim
        float theta_odd = t_odd * freq;
        cos_out[t * 2 * head_dim + head_dim + d] = f_to_bf16(cosf(theta_odd));
        float s_odd = sinf(theta_odd);
        if (d % 2 == 1) s_odd = -s_odd;
        sin_out[t * 2 * head_dim + head_dim + d] = f_to_bf16(s_odd);
    }
}

void yarn_compute_tables(bf16* cos_out, bf16* sin_out,
                         const float* angular_freq, int head_dim,
                         int max_seq_len, int paired, cudaStream_t stream) {
    int width = paired ? head_dim : head_dim;
    dim3 grid(2 * max_seq_len);
    dim3 block(width);
    yarn_compute_tables_kernel<<<grid, block, 0, stream>>>(
        cos_out, sin_out, angular_freq, head_dim, max_seq_len, paired);
}

// ============================================================================
// Embedding kernels
// ============================================================================

__global__ void gather_embed_kernel(bf16* __restrict__ out,
                                     const bf16* __restrict__ embed,
                                     const int32_t* __restrict__ indices,
                                     int num_tokens, int embed_dim) {
    int token = blockIdx.x;
    int d = threadIdx.x;
    if (token >= num_tokens || d >= embed_dim) return;

    int idx = indices[token];
    out[token * embed_dim + d] = embed[idx * embed_dim + d];
}

void gather_embed(bf16* out, const bf16* embed, const int32_t* indices,
                  int num_tokens, int embed_dim, cudaStream_t stream) {
    dim3 grid(num_tokens);
    // Use multiple threads per token for large embed_dim
    int threads = min(1024, embed_dim);
    gather_embed_kernel<<<grid, threads, 0, stream>>>(out, embed, indices, num_tokens, embed_dim);

    // Handle case where embed_dim > 1024 with a loop
    if (embed_dim > 1024) {
        // Need a version with loop
    }
}

__global__ void scatter_add_embed_kernel(bf16* __restrict__ embed_grad,
                                          const bf16* __restrict__ grad,
                                          const int32_t* __restrict__ indices,
                                          int num_tokens, int embed_dim) {
    int token = blockIdx.x;
    int d = threadIdx.x;
    if (token >= num_tokens || d >= embed_dim) return;

    int idx = indices[token];
    // SM 8.0+ supports native bf16 atomicAdd
    atomicAdd(&embed_grad[idx * embed_dim + d], grad[token * embed_dim + d]);
}

// Note: scatter_add for bf16 needs a float accumulator. We'll accumulate into float grad.
__global__ void scatter_add_embed_float_kernel(float* __restrict__ embed_grad,
                                                const bf16* __restrict__ grad,
                                                const int32_t* __restrict__ indices,
                                                int num_tokens, int embed_dim) {
    int token = blockIdx.x;
    int d = threadIdx.x;
    if (token >= num_tokens || d >= embed_dim) return;

    int idx = indices[token];
    atomicAdd(&embed_grad[(int64_t)idx * embed_dim + d],
              bf16_to_f(grad[token * embed_dim + d]));
}

void scatter_add_embed(bf16* embed_grad, const bf16* grad, const int32_t* indices,
                       int num_tokens, int embed_dim, cudaStream_t stream) {
    // For correctness, we should accumulate into float. The caller should provide float* workspace.
    dim3 grid(num_tokens);
    int threads = min(1024, embed_dim);
    scatter_add_embed_kernel<<<grid, threads, 0, stream>>>(embed_grad, grad, indices, num_tokens, embed_dim);
}

// ============================================================================
// Bigram hash
// ============================================================================

__global__ void bigram_hash_kernel(int32_t* __restrict__ out,
                                    const int32_t* __restrict__ input,
                                    int num_tokens) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_tokens) return;

    int mod = BIGRAM_VOCAB_SIZE - 1;
    if (idx == 0) {
        out[0] = mod;
    } else {
        int curr = input[idx];
        int prev = input[idx - 1];
        out[idx] = ((BIGRAM_RAND_INT_1 * curr) ^ (BIGRAM_RAND_INT_2 * prev)) % mod;
        // Handle negative modulo
        if (out[idx] < 0) out[idx] += mod;
    }
}

void bigram_hash(int32_t* out, const int32_t* input, int num_tokens, cudaStream_t stream) {
    int threads = 256;
    bigram_hash_kernel<<<cdiv(num_tokens, threads), threads, 0, stream>>>(out, input, num_tokens);
}

// ============================================================================
// Sigmoid gate
// ============================================================================

// out = scale * sigmoid(x_slice @ gate_w.T)
// x_slice: [num_tokens, in_dim], gate_w: [out_dim, in_dim]
// out: [num_tokens, out_dim]
__global__ void sigmoid_gate_kernel(bf16* __restrict__ out,
                                     const bf16* __restrict__ x,
                                     const bf16* __restrict__ gate_w,
                                     int num_tokens, int in_dim, int out_dim,
                                     int x_stride, // stride between tokens in x (full model_dim)
                                     float scale) {
    int t = blockIdx.x;
    int o = threadIdx.x;
    if (t >= num_tokens || o >= out_dim) return;

    float sum = 0.0f;
    for (int i = 0; i < in_dim; i++) {
        sum += bf16_to_f(x[t * x_stride + i]) * bf16_to_f(gate_w[o * in_dim + i]);
    }
    float sigmoid_val = scale / (1.0f + expf(-sum));
    out[t * out_dim + o] = f_to_bf16(sigmoid_val);
}

void sigmoid_gate(bf16* out, const bf16* x, const bf16* gate_w,
                  int num_tokens, int in_dim, int out_dim, float scale,
                  cudaStream_t stream) {
    dim3 grid(num_tokens);
    dim3 block(out_dim);
    sigmoid_gate_kernel<<<grid, block, 0, stream>>>(
        out, x, gate_w, num_tokens, in_dim, out_dim, in_dim, scale);
}

// ============================================================================
// Elementwise operations
// ============================================================================

__global__ void fused_add_scale_kernel(bf16* __restrict__ out,
                                        const bf16* __restrict__ x,
                                        const bf16* __restrict__ y,
                                        float scale_x, float scale_y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = f_to_bf16(scale_x * bf16_to_f(x[idx]) + scale_y * bf16_to_f(y[idx]));
}

void fused_add_scale(bf16* out, const bf16* x, const bf16* y,
                     float scale_x, float scale_y, int n, cudaStream_t stream) {
    int threads = 256;
    fused_add_scale_kernel<<<cdiv(n, threads), threads, 0, stream>>>(out, x, y, scale_x, scale_y, n);
}

__global__ void fused_add3_kernel(bf16* __restrict__ out,
                                   const bf16* __restrict__ x,
                                   const bf16* __restrict__ y,
                                   const bf16* __restrict__ z,
                                   float a, float b, float c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = f_to_bf16(a * bf16_to_f(x[idx]) + b * bf16_to_f(y[idx]) + c * bf16_to_f(z[idx]));
}

void fused_add3(bf16* out, const bf16* x, const bf16* y, const bf16* z,
                float a, float b, float c, int n, cudaStream_t stream) {
    int threads = 256;
    fused_add3_kernel<<<cdiv(n, threads), threads, 0, stream>>>(out, x, y, z, a, b, c, n);
}

// Smear forward: out[0] = x[0], out[i] = x[i] + gate * x[i-1]
// where gate = smear_lambda * sigmoid(smear_gate_w @ x[i, :12])
__global__ void smear_forward_kernel(bf16* __restrict__ out,
                                      const bf16* __restrict__ x,
                                      const bf16* __restrict__ smear_gate_w,
                                      float smear_lambda,
                                      int num_tokens, int model_dim) {
    int d = blockIdx.y * blockDim.x + threadIdx.x;
    if (d >= model_dim) return;

    // Position 0: just copy
    if (blockIdx.x == 0) {
        out[d] = x[d];
        return;
    }

    int t = blockIdx.x;
    if (t >= num_tokens) return;

    // Compute gate: smear_lambda * sigmoid(smear_gate_w @ x[t, :12])
    // smear_gate_w is [1, 12], output is scalar
    // This needs to be computed per-token but is the same for all d
    // We use shared memory for the gate value
    __shared__ float s_gate;
    if (threadIdx.x == 0) {
        float dot = 0.0f;
        for (int i = 0; i < 12; i++) {
            dot += bf16_to_f(x[t * model_dim + i]) * bf16_to_f(smear_gate_w[i]);
        }
        s_gate = smear_lambda / (1.0f + expf(-dot));
    }
    __syncthreads();
    float gate = s_gate;

    float curr = bf16_to_f(x[t * model_dim + d]);
    float prev = bf16_to_f(x[(t - 1) * model_dim + d]);
    out[t * model_dim + d] = f_to_bf16(curr + gate * prev);
}

void smear_forward(bf16* out, const bf16* x, const bf16* smear_gate_w,
                   float smear_lambda, int num_tokens, int model_dim,
                   cudaStream_t stream) {
    dim3 grid(num_tokens, cdiv(model_dim, 256));
    dim3 block(256);
    smear_forward_kernel<<<grid, block, 0, stream>>>(
        out, x, smear_gate_w, smear_lambda, num_tokens, model_dim);
}

// ============================================================================
// Optimizer kernels
// ============================================================================

__global__ void adam_update_kernel(bf16* __restrict__ param,
                                   const bf16* __restrict__ grad,
                                   float* __restrict__ exp_avg,
                                   float* __restrict__ exp_avg_sq,
                                   float beta1, float beta2, float eps,
                                   float step_size, float eff_wd, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float g = bf16_to_f(grad[idx]);
    float ea = exp_avg[idx];
    float eas = exp_avg_sq[idx];
    float p = bf16_to_f(param[idx]);

    // Adam update
    ea = beta1 * ea + (1.0f - beta1) * g;
    eas = beta2 * eas + (1.0f - beta2) * g * g;
    float update = ea / (sqrtf(eas) + eps) * step_size;

    // Cautious weight decay
    float mask = (update * p > 0.0f) ? 1.0f : 0.0f;
    update += p * mask * eff_wd;

    p -= update;

    exp_avg[idx] = ea;
    exp_avg_sq[idx] = eas;
    param[idx] = f_to_bf16(p);
}

void adam_update(bf16* param, const bf16* grad,
                float* exp_avg, float* exp_avg_sq,
                float beta1, float beta2, float eps,
                float step_size, float eff_wd,
                int numel, cudaStream_t stream) {
    int threads = 256;
    adam_update_kernel<<<cdiv(numel, threads), threads, 0, stream>>>(
        param, grad, exp_avg, exp_avg_sq, beta1, beta2, eps, step_size, eff_wd, numel);
}

// NorMuon cautious WD + mantissa-tracked update
__global__ void muon_cautious_update_kernel(uint16_t* __restrict__ p,
                                             uint16_t* __restrict__ mantissa,
                                             const bf16* __restrict__ grad,
                                             float eff_wd, float eff_lr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float g = bf16_to_f(grad[idx]);

    // Reconstruct FP32 from BF16 upper bits + uint16 mantissa lower bits
    uint32_t p_upper = (uint32_t)p[idx] << 16;
    uint32_t p_lower = (uint32_t)mantissa[idx];
    uint32_t p_bits = p_upper | p_lower;
    float p_precise;
    memcpy(&p_precise, &p_bits, sizeof(float));

    // Cautious weight decay: mask = (grad * p_precise) >= 0
    float mask = (g * p_precise >= 0.0f) ? 1.0f : 0.0f;
    p_precise = p_precise - (p_precise * mask * eff_wd * eff_lr) - (g * eff_lr);

    // Split back into BF16 + mantissa
    uint32_t result_bits;
    memcpy(&result_bits, &p_precise, sizeof(float));
    p[idx] = (uint16_t)(result_bits >> 16);
    mantissa[idx] = (uint16_t)(result_bits & 0xFFFF);
}

void muon_cautious_update(uint16_t* param_u16, uint16_t* mantissa,
                          const bf16* grad, float eff_wd, float eff_lr,
                          int numel, cudaStream_t stream) {
    int threads = 256;
    muon_cautious_update_kernel<<<cdiv(numel, threads), threads, 0, stream>>>(
        param_u16, mantissa, grad, eff_wd, eff_lr, numel);
}

// ============================================================================
// NorMuon variance reduction
// ============================================================================

// Variance reduction: normalize v_chunk using second momentum buffer
// Implements _apply_normuon_variance_reduction from Python
__global__ void normuon_variance_reduction_kernel(bf16* __restrict__ v_chunk,
                                                   float* __restrict__ second_mom,
                                                   float beta2,
                                                   int red_dim,  // -1 or -2
                                                   int outer, int d0, int d1) {
    // This is a complex operation - simplified implementation
    // Full implementation would need multiple passes
    // For now, do per-matrix processing
    int mat_idx = blockIdx.x;
    if (mat_idx >= outer) return;

    bf16* v = v_chunk + mat_idx * d0 * d1;

    // Compute v_mean = v^2.mean(dim=red_dim)
    // If red_dim == -1 (reduce along d1): result shape [d0, 1]
    // If red_dim == -2 (reduce along d0): result shape [1, d1]

    if (red_dim == -1) {
        // Reduce along columns (d1), output per row
        for (int r = threadIdx.x; r < d0; r += blockDim.x) {
            float sq_sum = 0.0f;
            for (int c = 0; c < d1; c++) {
                float val = bf16_to_f(v[r * d1 + c]);
                sq_sum += val * val;
            }
            float v_mean = sq_sum / d1;

            // Update second momentum (EMA)
            int sm_idx = mat_idx * d0 + r;
            float sm = second_mom[sm_idx];
            sm = sm * beta2 + v_mean * (1.0f - beta2);
            second_mom[sm_idx] = sm;
        }
    } else {
        // Reduce along rows (d0), output per column
        for (int c = threadIdx.x; c < d1; c += blockDim.x) {
            float sq_sum = 0.0f;
            for (int r = 0; r < d0; r++) {
                float val = bf16_to_f(v[r * d1 + c]);
                sq_sum += val * val;
            }
            float v_mean = sq_sum / d0;

            int sm_idx = mat_idx * d1 + c;
            float sm = second_mom[sm_idx];
            sm = sm * beta2 + v_mean * (1.0f - beta2);
            second_mom[sm_idx] = sm;
        }
    }
    __syncthreads();

    // Compute v_norm and v_norm_new, then scale
    // This is a simplified version - the full Python does:
    // v_norm_sq = v_mean.sum((-2,-1)).mul_(red_dim_size)
    // v_norm = sqrt(v_norm_sq)
    // step_size = second_mom.clamp_min(1e-10).rsqrt()
    // scaled_sq_sum = (v_mean * red_dim_size) * step_size^2
    // v_norm_new = sqrt(scaled_sq_sum.sum((-2,-1)))
    // final_scale = step_size * (v_norm / v_norm_new)
    // v *= final_scale

    // For simplicity, compute the full operation
    float v_norm_sq = 0.0f;
    float scaled_sq_sum_total = 0.0f;

    int red_size = (red_dim == -1) ? d1 : d0;
    int other_size = (red_dim == -1) ? d0 : d1;

    for (int i = threadIdx.x; i < other_size; i += blockDim.x) {
        int sm_idx = mat_idx * other_size + i;
        float sm = second_mom[sm_idx];

        // Recompute v_mean for this index
        float sq_sum = 0.0f;
        if (red_dim == -1) {
            for (int c = 0; c < d1; c++) {
                float val = bf16_to_f(v[i * d1 + c]);
                sq_sum += val * val;
            }
        } else {
            for (int r = 0; r < d0; r++) {
                float val = bf16_to_f(v[r * d1 + i]);
                sq_sum += val * val;
            }
        }
        float v_mean = sq_sum / red_size;

        v_norm_sq += v_mean * red_size;
        float step_size = rsqrtf(fmaxf(sm, 1e-10f));
        scaled_sq_sum_total += v_mean * red_size * step_size * step_size;
    }

    // Block reduce
    v_norm_sq = block_reduce_sum(v_norm_sq);
    scaled_sq_sum_total = block_reduce_sum(scaled_sq_sum_total);

    __shared__ float s_ratio;
    if (threadIdx.x == 0) {
        float v_norm = sqrtf(v_norm_sq);
        float v_norm_new = sqrtf(fmaxf(scaled_sq_sum_total, 1e-10f));
        s_ratio = v_norm / v_norm_new;
    }
    __syncthreads();

    // Apply per-element scaling
    for (int i = threadIdx.x; i < d0 * d1; i += blockDim.x) {
        int r = i / d1;
        int c = i % d1;
        int sm_idx = (red_dim == -1) ? (mat_idx * d0 + r) : (mat_idx * d1 + c);
        float sm = second_mom[sm_idx];
        float step_size = rsqrtf(fmaxf(sm, 1e-10f));
        float final_scale = step_size * s_ratio;
        float val = bf16_to_f(v[i]);
        v[i] = f_to_bf16(val * final_scale);
    }
}

void normuon_variance_reduction(bf16* v_chunk, float* second_momentum_buffer,
                                float beta2, int red_dim,
                                int outer_size, int dim0, int dim1,
                                cudaStream_t stream) {
    int threads = min(1024, max(dim0, dim1));
    normuon_variance_reduction_kernel<<<outer_size, threads, 0, stream>>>(
        v_chunk, second_momentum_buffer, beta2, red_dim, outer_size, dim0, dim1);
}

// ============================================================================
// Scale tensor
// ============================================================================

__global__ void scale_tensor_kernel(bf16* __restrict__ x, float scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    x[idx] = f_to_bf16(bf16_to_f(x[idx]) * scale);
}

void scale_tensor(bf16* x, float scale, int n, cudaStream_t stream) {
    int threads = 256;
    scale_tensor_kernel<<<cdiv(n, threads), threads, 0, stream>>>(x, scale, n);
}

// ============================================================================
// Tensor norm (for Polar Express)
// ============================================================================

__global__ void tensor_norm_kernel(const bf16* __restrict__ x, float* __restrict__ out_norm,
                                    int batch, int rows, int cols) {
    int b = blockIdx.x;
    if (b >= batch) return;

    const bf16* x_mat = x + b * rows * cols;
    float sum_sq = 0.0f;

    for (int i = threadIdx.x; i < rows * cols; i += blockDim.x) {
        float val = bf16_to_f(x_mat[i]);
        sum_sq += val * val;
    }
    sum_sq = block_reduce_sum(sum_sq);

    if (threadIdx.x == 0) {
        out_norm[b] = sqrtf(sum_sq);
    }
}

void tensor_norm(const bf16* x, float* out_norm, int batch, int rows, int cols,
                 cudaStream_t stream) {
    int threads = min(1024, rows * cols);
    tensor_norm_kernel<<<batch, threads, 0, stream>>>(x, out_norm, batch, rows, cols);
}

// ============================================================================
// FP8 quantization kernels
// ============================================================================

__global__ void bf16_to_fp8_e4m3_kernel(fp8e4m3* __restrict__ out,
                                         const bf16* __restrict__ x,
                                         float scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float val = bf16_to_f(x[idx]) / scale;  // divide to match Python: x.div(scale)
    // Clamp to FP8 e4m3 range [-448, 448]
    val = fminf(fmaxf(val, -448.0f), 448.0f);
    out[idx] = __nv_fp8_e4m3(val);
}

void bf16_to_fp8_e4m3(fp8e4m3* out, const bf16* x, float scale, int n, cudaStream_t stream) {
    int threads = 256;
    bf16_to_fp8_e4m3_kernel<<<cdiv(n, threads), threads, 0, stream>>>(out, x, scale, n);
}

__global__ void bf16_to_fp8_e5m2_kernel(fp8e5m2* __restrict__ out,
                                         const bf16* __restrict__ x,
                                         float scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float val = bf16_to_f(x[idx]) / scale;  // divide to match Python: x.div(scale)
    val = fminf(fmaxf(val, -57344.0f), 57344.0f);
    out[idx] = __nv_fp8_e5m2(val);
}

void bf16_to_fp8_e5m2(fp8e5m2* out, const bf16* x, float scale, int n, cudaStream_t stream) {
    int threads = 256;
    bf16_to_fp8_e5m2_kernel<<<cdiv(n, threads), threads, 0, stream>>>(out, x, scale, n);
}

// ============================================================================
// Nesterov momentum
// ============================================================================

__global__ void nesterov_momentum_kernel(bf16* __restrict__ grad_out,
                                          float* __restrict__ buf,
                                          const bf16* __restrict__ grad_in,
                                          float momentum, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float g = bf16_to_f(grad_in[idx]);
    float b = buf[idx];

    // buf = momentum * buf + (1-momentum) * g
    b = momentum * b + (1.0f - momentum) * g;
    buf[idx] = b;

    // Nesterov lookahead: out = momentum * buf + (1-momentum) * g
    // = momentum * (momentum*old_buf + (1-m)*g) + (1-m)*g
    float out = momentum * b + (1.0f - momentum) * g;
    grad_out[idx] = f_to_bf16(out);
}

void nesterov_momentum(bf16* grad_out, float* momentum_buffer, const bf16* grad_in,
                       float momentum, int n, cudaStream_t stream) {
    int threads = 256;
    nesterov_momentum_kernel<<<cdiv(n, threads), threads, 0, stream>>>(
        grad_out, momentum_buffer, grad_in, momentum, n);
}

// Norm divide: x[b,i] /= (norms[b] * (1+safety) + eps)
__global__ void norm_divide_kernel(bf16* __restrict__ x, const float* __restrict__ norms,
                                    int batch, int elems, float safety, float eps) {
    int b = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch || idx >= elems) return;

    float scale = 1.0f / (norms[b] * (1.0f + safety) + eps);
    int offset = b * elems + idx;
    x[offset] = f_to_bf16(bf16_to_f(x[offset]) * scale);
}

void norm_divide(bf16* x, const float* norms, int batch, int elems_per_batch,
                 float safety, float eps, cudaStream_t stream) {
    int threads = 256;
    dim3 grid(cdiv(elems_per_batch, threads), batch);
    norm_divide_kernel<<<grid, threads, 0, stream>>>(x, norms, batch, elems_per_batch, safety, eps);
}

// ============================================================================
// GPU reduction
// ============================================================================

__global__ void reduce_sum_kernel(float* __restrict__ out, const float* __restrict__ x, int n) {
    float sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        sum += x[i];
    }
    sum = block_reduce_sum(sum);
    if (threadIdx.x == 0) out[0] = sum;
}

void gpu_reduce_sum(float* out, const float* x, int n, cudaStream_t stream) {
    int threads = min(1024, n);
    reduce_sum_kernel<<<1, threads, 0, stream>>>(out, x, n);
}

__global__ void fill_constant_f32_kernel(float* __restrict__ out, float val, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = val;
}

void fill_constant_f32(float* out, float val, int n, cudaStream_t stream) {
    int threads = 256;
    fill_constant_f32_kernel<<<cdiv(n, threads), threads, 0, stream>>>(out, val, n);
}

// Dot product: *out += sum(a[i] * b[i])
__global__ void bf16_dot_product_kernel(float* __restrict__ out,
                                         const bf16* __restrict__ a,
                                         const bf16* __restrict__ b,
                                         int n) {
    float sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        sum += bf16_to_f(a[i]) * bf16_to_f(b[i]);
    }
    sum = block_reduce_sum(sum);
    if (threadIdx.x == 0) {
        atomicAdd(out, sum);
    }
}

void bf16_dot_product(float* out, const bf16* a, const bf16* b, int n, cudaStream_t stream) {
    int threads = min(1024, n);
    bf16_dot_product_kernel<<<1, threads, 0, stream>>>(out, a, b, n);
}

// ============================================================================
// Elementwise multiply broadcast (for per-head gating)
// ============================================================================

// gate[i % gate_stride] applied to x
__global__ void elementwise_mul_broadcast_kernel(bf16* __restrict__ out,
                                                  const bf16* __restrict__ x,
                                                  const bf16* __restrict__ gate,
                                                  int total, int inner_dim,
                                                  int gate_stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    // Determine which gate element to use
    // Layout: [T, H, HD] or similar, gate broadcasts over last dim
    int gate_idx = (idx / inner_dim) % gate_stride;
    float g = bf16_to_f(gate[gate_idx]);
    out[idx] = f_to_bf16(bf16_to_f(x[idx]) * g);
}

void elementwise_mul_broadcast(bf16* out, const bf16* x, const bf16* gate,
                               int total, int inner_dim, int gate_stride,
                               cudaStream_t stream) {
    int threads = 256;
    elementwise_mul_broadcast_kernel<<<cdiv(total, threads), threads, 0, stream>>>(
        out, x, gate, total, inner_dim, gate_stride);
}

// Fused gate add: out[i] = x[i] + gate[gate_idx] * y[i]
__global__ void fused_gate_add_kernel(bf16* __restrict__ out,
                                       const bf16* __restrict__ x,
                                       const bf16* __restrict__ gate,
                                       const bf16* __restrict__ y,
                                       int total, int inner_dim, int gate_stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int gate_idx = (idx / inner_dim) % gate_stride;
    float g = bf16_to_f(gate[gate_idx]);
    out[idx] = f_to_bf16(bf16_to_f(x[idx]) + g * bf16_to_f(y[idx]));
}

void fused_gate_add(bf16* out, const bf16* x, const bf16* gate, const bf16* y,
                    int total, int inner_dim, int gate_stride, cudaStream_t stream) {
    int threads = 256;
    fused_gate_add_kernel<<<cdiv(total, threads), threads, 0, stream>>>(
        out, x, gate, y, total, inner_dim, gate_stride);
}

// ============================================================================
// Naive variable-length attention
// ============================================================================

// One block per (query_position, head).
// Threads collaborate on dot products and weighted V sum.
// Shared memory stores attention scores for all valid key positions.
__global__ void naive_varlen_attention_kernel(
        bf16* __restrict__ out,
        const bf16* __restrict__ Q,
        const bf16* __restrict__ K,
        const bf16* __restrict__ V,
        const int32_t* __restrict__ cu_seqlens,
        int num_seqs, int H, int HD,
        float scale, int window_left) {
    int pos = blockIdx.x;   // query position in flat sequence
    int head = blockIdx.y;  // head index

    // Binary search for document containing this position
    int lo = 0, hi = num_seqs;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (cu_seqlens[mid + 1] <= pos) lo = mid + 1;
        else hi = mid;
    }
    int doc_start = cu_seqlens[lo];

    // Key range: causal (can only attend to positions <= pos) + window
    int key_start = doc_start;
    if (window_left > 0) key_start = max(key_start, pos - window_left + 1);
    int key_end = pos + 1;
    int num_keys = key_end - key_start;
    if (num_keys <= 0) {
        // No valid keys (shouldn't happen with causal, but guard)
        for (int d = threadIdx.x; d < HD; d += blockDim.x)
            out[pos * H * HD + head * HD + d] = f_to_bf16(0.0f);
        return;
    }

    // Shared memory for attention scores
    extern __shared__ float smem[];

    const bf16* q_vec = Q + pos * H * HD + head * HD;

    // Phase 1: Compute Q @ K.T scores for all valid key positions
    // Each thread handles some key positions, computing full HD-dim dot products
    float local_max = -FLT_MAX;
    for (int k = threadIdx.x; k < num_keys; k += blockDim.x) {
        int k_pos = key_start + k;
        const bf16* k_vec = K + k_pos * H * HD + head * HD;
        float dot = 0.0f;
        for (int d = 0; d < HD; d++) {
            dot += bf16_to_f(q_vec[d]) * bf16_to_f(k_vec[d]);
        }
        smem[k] = dot * scale;
        local_max = fmaxf(local_max, smem[k]);
    }
    __syncthreads();

    // Phase 2: Find max across all threads (for numerically stable softmax)
    local_max = block_reduce_max(local_max);
    __shared__ float s_max;
    if (threadIdx.x == 0) s_max = local_max;
    __syncthreads();

    // Phase 3: Compute exp(score - max) and sum
    float local_sum = 0.0f;
    for (int k = threadIdx.x; k < num_keys; k += blockDim.x) {
        smem[k] = expf(smem[k] - s_max);
        local_sum += smem[k];
    }
    __syncthreads();

    local_sum = block_reduce_sum(local_sum);
    __shared__ float s_sum_inv;
    if (threadIdx.x == 0) s_sum_inv = 1.0f / (local_sum + 1e-10f);
    __syncthreads();

    // Phase 4: Normalize scores
    for (int k = threadIdx.x; k < num_keys; k += blockDim.x) {
        smem[k] *= s_sum_inv;
    }
    __syncthreads();

    // Phase 5: Weighted sum of V vectors
    // Each thread handles some output dimensions
    bf16* out_vec = out + pos * H * HD + head * HD;
    for (int d = threadIdx.x; d < HD; d += blockDim.x) {
        float acc = 0.0f;
        for (int k = 0; k < num_keys; k++) {
            int k_pos = key_start + k;
            acc += smem[k] * bf16_to_f(V[k_pos * H * HD + head * HD + d]);
        }
        out_vec[d] = f_to_bf16(acc);
    }
}

void naive_varlen_attention(bf16* out,
                            const bf16* Q, const bf16* K, const bf16* V,
                            const int32_t* cu_seqlens, int num_seqs,
                            int total_T, int H, int HD,
                            float scale, int window_left,
                            cudaStream_t stream) {
    // Grid: [total_T, H], Block: [min(256, max_window)]
    // Shared memory: max_possible_keys * sizeof(float)
    // The max number of keys for any query is min(max_seq_len, window_left)
    // We allocate conservatively based on window_left (or max_seq_len if window_left==0)
    int max_keys = window_left > 0 ? window_left : 4096; // upper bound
    int threads = 256;
    int smem_bytes = max_keys * sizeof(float);
    // Clamp shared memory to 48KB (default limit)
    if (smem_bytes > 48 * 1024) smem_bytes = 48 * 1024;

    dim3 grid(total_T, H);
    naive_varlen_attention_kernel<<<grid, threads, smem_bytes, stream>>>(
        out, Q, K, V, cu_seqlens, num_seqs, H, HD, scale, window_left);
}

// Key offset: shift k[t, :, HD/2:] = k[t-1, :, HD/2:] for t > doc_start
__global__ void key_offset_shift_kernel(bf16* __restrict__ k,
                                         const int32_t* __restrict__ cu_seqlens,
                                         int num_seqs, int H, int HD) {
    // Grid: [total_T, H], Block: [HD/2]
    int pos = blockIdx.x;
    int head = blockIdx.y;
    int d = threadIdx.x;  // dimension offset within second half
    if (d >= HD / 2) return;

    // Find document for this position
    int doc_start = 0;
    for (int s = 0; s < num_seqs; s++) {
        if (pos >= cu_seqlens[s] && pos < cu_seqlens[s + 1]) {
            doc_start = cu_seqlens[s];
            break;
        }
    }

    // Only shift for positions > doc_start
    // Work backward to avoid overwriting: process from end to start
    // But since we're running in parallel, we need to copy from t-1 to t
    // for all t simultaneously. This is a forward shift of the second half.
    // k[t, h, HD/2+d] = k[t-1, h, HD/2+d] for t > doc_start
    if (pos > doc_start) {
        int src = (pos - 1) * H * HD + head * HD + HD / 2 + d;
        int dst = pos * H * HD + head * HD + HD / 2 + d;
        k[dst] = k[src];
    }
}

void key_offset_shift(bf16* k, const int32_t* cu_seqlens, int num_seqs,
                      int total_T, int H, int HD, cudaStream_t stream) {
    // Must process positions from back to front to avoid overwriting
    // But CUDA kernels are parallel. Need to use a sequential loop or temp buffer.
    // Simple approach: use a temp buffer for the shifted values, then copy back.
    // Alternative: process in reverse order per document.
    // For correctness with parallel execution, we read from k (source) before any writes.
    // The issue is that position t reads from t-1, and position t+1 reads from t.
    // If t+1 writes before t reads, we get wrong results.
    // Solution: make a copy of the second half first, then shift.

    // Actually, the shift is: for each doc, move k[doc_start:doc_end-1, :, HD/2:]
    // to k[doc_start+1:doc_end, :, HD/2:]. This is a memcpy within each doc.
    // Since all docs are independent, we can do per-doc memcpy.
    // But cu_seqlens is on GPU. Let's copy it to host.
    int32_t* h_seqlens = (int32_t*)malloc((num_seqs + 1) * sizeof(int32_t));
    cudaMemcpy(h_seqlens, cu_seqlens, (num_seqs + 1) * sizeof(int32_t), cudaMemcpyDeviceToHost);

    for (int s = 0; s < num_seqs; s++) {
        int start = h_seqlens[s];
        int end = h_seqlens[s + 1];
        int doc_len = end - start;
        if (doc_len <= 1) continue;

        // For each head, shift the second half of HD backward
        // k[start+1:end, h, HD/2:] = k[start:end-1, h, HD/2:]
        // Since data is [T, H, HD], each position is H*HD elements apart
        // The second half of each head starts at offset h*HD + HD/2

        // Do it with cudaMemcpy per head (strided, so need to loop)
        // Actually, the data for consecutive positions is contiguous in the T dimension
        // k layout: [T, H, HD] = contiguous in HD, then H, then T
        // So k[t, h, d] = k[(t*H + h)*HD + d]
        // k[t, :, HD/2:] spans h=0..H-1, d=HD/2..HD-1
        // For a single position t, the second half is at offsets:
        //   t*H*HD + h*HD + HD/2 for each h
        // These are NOT contiguous across heads.

        // Simplest correct approach: copy per-head per-doc using cudaMemcpy
        for (int h = 0; h < H; h++) {
            // Source: positions [start, start+1, ..., end-2], head h, dims [HD/2, HD)
            // Dest:   positions [start+1, start+2, ..., end-1], head h, dims [HD/2, HD)
            // Source stride between positions: H*HD elements
            // This is a strided copy, not a contiguous memcpy.
            // Use cudaMemcpy2D: src_pitch = H*HD*sizeof(bf16), width = HD/2*sizeof(bf16)
            cudaMemcpy2D(
                k + (int64_t)(start + 1) * H * HD + h * HD + HD / 2,  // dst
                H * HD * sizeof(bf16),                                  // dst pitch
                k + (int64_t)start * H * HD + h * HD + HD / 2,         // src
                H * HD * sizeof(bf16),                                  // src pitch
                (HD / 2) * sizeof(bf16),                                // width
                doc_len - 1,                                            // height (num rows to copy)
                cudaMemcpyDeviceToDevice
            );
        }
    }
    free(h_seqlens);
}

// ============================================================================
// Naive attention backward
// ============================================================================

// Computes dQ per query position (no atomics needed for dQ).
// For dK and dV, accumulates via float atomicAdd into float buffers.
// Each block handles one (query_position, head) pair.
//
// This is O(T²) per head. For production, use cuDNN flash attention backward.
__global__ void naive_varlen_attention_backward_dq_kernel(
        float* __restrict__ dQ_f32,   // [T, H, HD] float accumulator
        float* __restrict__ dK_f32,   // [T, H, HD] float accumulator
        float* __restrict__ dV_f32,   // [T, H, HD] float accumulator
        const bf16* __restrict__ d_out,
        const bf16* __restrict__ Q,
        const bf16* __restrict__ K,
        const bf16* __restrict__ V,
        const int32_t* __restrict__ cu_seqlens,
        int num_seqs, int H, int HD,
        float scale, int window_left) {
    int pos = blockIdx.x;   // query position
    int head = blockIdx.y;  // head index

    // Find document containing this position
    int lo = 0, hi = num_seqs;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (cu_seqlens[mid + 1] <= pos) lo = mid + 1;
        else hi = mid;
    }
    int doc_start = cu_seqlens[lo];

    // Key range: causal + window
    int key_start = doc_start;
    if (window_left > 0) key_start = max(key_start, pos - window_left + 1);
    int key_end = pos + 1;
    int num_keys = key_end - key_start;
    if (num_keys <= 0) return;

    extern __shared__ float smem[];
    // smem layout: [num_keys] for attention weights P

    const bf16* q_vec = Q + pos * H * HD + head * HD;
    const bf16* do_vec = d_out + pos * H * HD + head * HD;

    // Phase 1: Recompute attention weights (same as forward)
    float local_max = -FLT_MAX;
    for (int k = threadIdx.x; k < num_keys; k += blockDim.x) {
        int k_pos = key_start + k;
        const bf16* k_vec = K + k_pos * H * HD + head * HD;
        float dot = 0.0f;
        for (int d = 0; d < HD; d++) {
            dot += bf16_to_f(q_vec[d]) * bf16_to_f(k_vec[d]);
        }
        smem[k] = dot * scale;
        local_max = fmaxf(local_max, smem[k]);
    }
    __syncthreads();

    local_max = block_reduce_max(local_max);
    __shared__ float s_max;
    if (threadIdx.x == 0) s_max = local_max;
    __syncthreads();

    float local_sum = 0.0f;
    for (int k = threadIdx.x; k < num_keys; k += blockDim.x) {
        smem[k] = expf(smem[k] - s_max);
        local_sum += smem[k];
    }
    __syncthreads();

    local_sum = block_reduce_sum(local_sum);
    __shared__ float s_sum_inv;
    if (threadIdx.x == 0) s_sum_inv = 1.0f / (local_sum + 1e-10f);
    __syncthreads();

    for (int k = threadIdx.x; k < num_keys; k += blockDim.x) {
        smem[k] *= s_sum_inv;
    }
    __syncthreads();

    // Phase 2: Compute dp = Σ_k P[k] * (d_out · V[k]) and accumulate dV
    float dp = 0.0f;
    for (int k = 0; k < num_keys; k++) {
        int k_pos = key_start + k;
        const bf16* v_vec = V + k_pos * H * HD + head * HD;
        float d_attn_k = 0.0f;
        for (int d = threadIdx.x; d < HD; d += blockDim.x) {
            d_attn_k += bf16_to_f(do_vec[d]) * bf16_to_f(v_vec[d]);
        }
        d_attn_k = block_reduce_sum(d_attn_k);
        __shared__ float s_dattn;
        if (threadIdx.x == 0) s_dattn = d_attn_k;
        __syncthreads();
        d_attn_k = s_dattn;

        dp += smem[k] * d_attn_k;

        // dV[k] += P[k] * d_out (float atomicAdd for precision)
        float pk = smem[k];
        for (int d = threadIdx.x; d < HD; d += blockDim.x) {
            float val = pk * bf16_to_f(do_vec[d]);
            atomicAdd(&dV_f32[k_pos * H * HD + head * HD + d], val);
        }
    }
    __syncthreads();

    // Phase 3: Compute d_score[k] and accumulate dQ (directly) and dK (atomicAdd)
    for (int k = 0; k < num_keys; k++) {
        int k_pos = key_start + k;
        const bf16* k_vec = K + k_pos * H * HD + head * HD;
        const bf16* v_vec = V + k_pos * H * HD + head * HD;

        // Recompute d_attn[k]
        float d_attn_k = 0.0f;
        for (int d = threadIdx.x; d < HD; d += blockDim.x) {
            d_attn_k += bf16_to_f(do_vec[d]) * bf16_to_f(v_vec[d]);
        }
        d_attn_k = block_reduce_sum(d_attn_k);
        __shared__ float s_dattn2;
        if (threadIdx.x == 0) s_dattn2 = d_attn_k;
        __syncthreads();
        d_attn_k = s_dattn2;

        float d_score = smem[k] * (d_attn_k - dp);

        // dQ[pos] += scale * d_score * K[k] (no atomic needed, one block per pos)
        for (int d = threadIdx.x; d < HD; d += blockDim.x) {
            dQ_f32[pos * H * HD + head * HD + d] += scale * d_score * bf16_to_f(k_vec[d]);
        }

        // dK[k] += scale * d_score * Q[pos] (atomic across query positions)
        for (int d = threadIdx.x; d < HD; d += blockDim.x) {
            float val = scale * d_score * bf16_to_f(q_vec[d]);
            atomicAdd(&dK_f32[k_pos * H * HD + head * HD + d], val);
        }
    }
}

// Convert float buffer to bf16
__global__ void f32_to_bf16_kernel(bf16* __restrict__ out, const float* __restrict__ in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = f_to_bf16(in[i]);
}

void f32_to_bf16(bf16* out, const float* x, int n, cudaStream_t stream) {
    int threads = 256;
    f32_to_bf16_kernel<<<(n + threads - 1) / threads, threads, 0, stream>>>(out, x, n);
}

void naive_varlen_attention_backward(
        bf16* dQ, bf16* dK, bf16* dV,
        float* dQ_f32, float* dK_f32, float* dV_f32,
        const bf16* d_out, const bf16* Q, const bf16* K, const bf16* V,
        const int32_t* cu_seqlens, int num_seqs,
        int total_T, int H, int HD,
        float scale, int window_left,
        cudaStream_t stream) {
    int total_elems = total_T * H * HD;

    // Zero pre-allocated float accumulators
    cudaMemsetAsync(dQ_f32, 0, total_elems * sizeof(float), stream);
    cudaMemsetAsync(dK_f32, 0, total_elems * sizeof(float), stream);
    cudaMemsetAsync(dV_f32, 0, total_elems * sizeof(float), stream);

    int max_keys = window_left > 0 ? window_left : 4096;
    int threads = 256;
    int smem_bytes = max_keys * sizeof(float);
    if (smem_bytes > 48 * 1024) smem_bytes = 48 * 1024;

    dim3 grid(total_T, H);
    naive_varlen_attention_backward_dq_kernel<<<grid, threads, smem_bytes, stream>>>(
        dQ_f32, dK_f32, dV_f32, d_out, Q, K, V,
        cu_seqlens, num_seqs, H, HD, scale, window_left);

    // Convert float accumulators back to bf16
    int block = 256;
    int ngrid = (total_elems + block - 1) / block;
    f32_to_bf16_kernel<<<ngrid, block, 0, stream>>>(dQ, dQ_f32, total_elems);
    f32_to_bf16_kernel<<<ngrid, block, 0, stream>>>(dK, dK_f32, total_elems);
    f32_to_bf16_kernel<<<ngrid, block, 0, stream>>>(dV, dV_f32, total_elems);
}
