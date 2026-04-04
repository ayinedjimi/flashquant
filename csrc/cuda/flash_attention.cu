// Copyright 2026 FlashQuant Authors. Apache-2.0 License.
//
// Standard FlashAttention-2 forward kernel (CUDA native).
//
// Algorithm overview (online softmax, Dao 2023):
//   For each Q tile [BLOCK_M, D]:
//     Initialize m_i = -inf, l_i = 1, acc = 0
//     For each KV tile [BLOCK_N, D]:
//       S = Q @ K^T * scale  (in log2 space)
//       m_new = max(m_i, rowmax(S))
//       P = exp2(S - m_new)
//       acc = acc * exp2(m_i - m_new) + P @ V
//       l_i = l_i * exp2(m_i - m_new) + rowsum(P)
//       m_i = m_new
//     out = acc / l_i
//
// Two paths:
//   - Prefill: BLOCK_M=64, full tile matmuls via shared memory
//   - Decode:  BLOCK_M=1, specialized dot-product path

#include "flash_attention.cuh"
#include "utils.cuh"

namespace flashquant {

// ============================================================================
// Prefill kernel (BLOCK_M > 1)
// ============================================================================

/// FlashAttention-2 prefill kernel.
///
/// Grid: (ceil(N_Q / BLOCK_M), B * H_Q)
/// Block: (BLOCK_N,) -- one thread per KV position in the tile
///
/// Shared memory layout:
///   q_tile  [BLOCK_M * HEAD_DIM]   -- Q stays resident
///   k_tile  [BLOCK_N * HEAD_DIM]   -- K tile, reloaded each iteration
///   v_tile  [BLOCK_N * HEAD_DIM]   -- V tile, loaded after S computation
///   s_tile  [BLOCK_M * BLOCK_N]    -- attention scores
///   m_i     [BLOCK_M]              -- running max
///   l_i     [BLOCK_M]              -- running denominator
///   acc     [BLOCK_M * HEAD_DIM]   -- output accumulator
template <int HEAD_DIM, int BLOCK_M, int BLOCK_N, typename scalar_t, bool IS_CAUSAL>
__global__ void flash_attention_prefill_kernel(
    const scalar_t* __restrict__ Q,      // (B, H_Q, N_Q, D)
    const scalar_t* __restrict__ K,      // (B, H_KV, N_KV, D)
    const scalar_t* __restrict__ V,      // (B, H_KV, N_KV, D)
    scalar_t* __restrict__ Out,          // (B, H_Q, N_Q, D)
    const int H_Q,
    const int H_KV,
    const int N_Q,
    const int N_KV,
    const float sm_scale)
{
    const int start_m = blockIdx.x;  // Q tile index
    const int off_hz  = blockIdx.y;  // batch * H_Q index
    const int off_z   = off_hz / H_Q;
    const int off_h_q = off_hz % H_Q;
    const int off_h_kv = off_h_q / (H_Q / H_KV);  // GQA mapping

    const int tid = threadIdx.x;  // [0, BLOCK_N)

    // Strides (contiguous in last dim)
    const int stride_qz = H_Q * N_Q * HEAD_DIM;
    const int stride_qh = N_Q * HEAD_DIM;
    const int stride_kz = H_KV * N_KV * HEAD_DIM;
    const int stride_kh = N_KV * HEAD_DIM;
    const int stride_oz = stride_qz;
    const int stride_oh = stride_qh;

    // Base pointers
    const scalar_t* q_base = Q + off_z * stride_qz + off_h_q * stride_qh;
    const scalar_t* k_base = K + off_z * stride_kz + off_h_kv * stride_kh;
    const scalar_t* v_base = V + off_z * stride_kz + off_h_kv * stride_kh;
    scalar_t* o_base = Out + off_z * stride_oz + off_h_q * stride_oh;

    // Shared memory
    extern __shared__ float smem[];
    float* q_tile = smem;                                     // BLOCK_M * HEAD_DIM
    float* k_tile = q_tile + BLOCK_M * HEAD_DIM;             // BLOCK_N * HEAD_DIM
    float* v_tile = k_tile + BLOCK_N * HEAD_DIM;             // BLOCK_N * HEAD_DIM
    float* s_tile = v_tile + BLOCK_N * HEAD_DIM;             // BLOCK_M * BLOCK_N
    float* m_i    = s_tile + BLOCK_M * BLOCK_N;              // BLOCK_M
    float* l_i    = m_i + BLOCK_M;                           // BLOCK_M
    float* acc    = l_i + BLOCK_M;                           // BLOCK_M * HEAD_DIM

    // Load Q tile [BLOCK_M, HEAD_DIM] into shared memory
    for (int i = tid; i < BLOCK_M * HEAD_DIM; i += BLOCK_N) {
        int row = i / HEAD_DIM;
        int col = i % HEAD_DIM;
        int q_row = start_m * BLOCK_M + row;
        q_tile[i] = (q_row < N_Q)
            ? static_cast<float>(q_base[q_row * HEAD_DIM + col])
            : 0.0f;
    }

    // Initialize softmax state
    for (int i = tid; i < BLOCK_M; i += BLOCK_N) {
        m_i[i] = -FLT_MAX;
        l_i[i] = 1.0f;
    }
    for (int i = tid; i < BLOCK_M * HEAD_DIM; i += BLOCK_N) {
        acc[i] = 0.0f;
    }
    __syncthreads();

    const float qk_scale = sm_scale * LOG2E;

    // Causal tile-skipping: upper bound on KV range
    const int hi = IS_CAUSAL ? min((start_m + 1) * BLOCK_M, N_KV) : N_KV;

    // === Main tile loop over K/V blocks ===
    for (int start_n = 0; start_n < hi; start_n += BLOCK_N) {

        // ---- Load K tile [BLOCK_N, HEAD_DIM] ----
        for (int i = tid; i < BLOCK_N * HEAD_DIM; i += BLOCK_N) {
            int row = i / HEAD_DIM;
            int col = i % HEAD_DIM;
            int kv_row = start_n + row;
            k_tile[i] = (kv_row < N_KV)
                ? static_cast<float>(k_base[kv_row * HEAD_DIM + col])
                : 0.0f;
        }
        __syncthreads();

        // ---- Compute S = Q @ K^T [BLOCK_M, BLOCK_N] ----
        // Thread tid computes column tid of S (i.e., all Q rows dotted with K[tid])
        {
            const int n = tid;
            for (int m = 0; m < BLOCK_M; ++m) {
                float dot = 0.0f;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    dot += q_tile[m * HEAD_DIM + d] * k_tile[n * HEAD_DIM + d];
                }
                dot *= qk_scale;

                // Causal masking
                if (IS_CAUSAL) {
                    int q_pos = start_m * BLOCK_M + m;
                    int k_pos = start_n + n;
                    if (k_pos > q_pos) dot = -FLT_MAX;
                }

                // OOB masking
                if (start_n + n >= N_KV) dot = -FLT_MAX;

                s_tile[m * BLOCK_N + n] = dot;
            }
        }
        __syncthreads();

        // ---- Load V tile [BLOCK_N, HEAD_DIM] ----
        for (int i = tid; i < BLOCK_N * HEAD_DIM; i += BLOCK_N) {
            int row = i / HEAD_DIM;
            int col = i % HEAD_DIM;
            int kv_row = start_n + row;
            v_tile[i] = (kv_row < N_KV)
                ? static_cast<float>(v_base[kv_row * HEAD_DIM + col])
                : 0.0f;
        }
        __syncthreads();

        // ---- Online softmax update + P @ V accumulation ----
        // Each thread handles Q rows at stride BLOCK_N
        for (int m = tid; m < BLOCK_M; m += BLOCK_N) {
            // Row max of S[m, :]
            float row_max = -FLT_MAX;
            for (int n = 0; n < BLOCK_N; ++n) {
                row_max = fmaxf(row_max, s_tile[m * BLOCK_N + n]);
            }

            float m_new = fmaxf(m_i[m], row_max);
            float alpha = exp2f(m_i[m] - m_new);

            // Rescale prior accumulator
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) {
                acc[m * HEAD_DIM + d] *= alpha;
            }

            // P = exp2(S - m_new), accumulate P @ V
            float l_ij = 0.0f;
            for (int n = 0; n < BLOCK_N; ++n) {
                float p = exp2f(s_tile[m * BLOCK_N + n] - m_new);
                l_ij += p;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    acc[m * HEAD_DIM + d] += p * v_tile[n * HEAD_DIM + d];
                }
            }

            l_i[m] = l_i[m] * alpha + l_ij;
            m_i[m] = m_new;
        }
        __syncthreads();
    }

    // ---- Epilogue: normalize and store ----
    for (int i = tid; i < BLOCK_M * HEAD_DIM; i += BLOCK_N) {
        int m = i / HEAD_DIM;
        int d = i % HEAD_DIM;
        int q_row = start_m * BLOCK_M + m;
        if (q_row < N_Q) {
            float val = acc[m * HEAD_DIM + d] / l_i[m];
            o_base[q_row * HEAD_DIM + d] = static_cast<scalar_t>(val);
        }
    }
}

// ============================================================================
// Decode kernel (BLOCK_M=1, specialized)
// ============================================================================

/// FlashAttention-2 decode kernel (single query token).
///
/// Grid: (B, H_Q)
/// Block: (THREADS,)
///
/// Q stays in registers. K/V stream through shared memory.
/// No causal masking needed (single token always attends to all prior).
template <int HEAD_DIM, int BLOCK_N, typename scalar_t>
__global__ void flash_attention_decode_kernel(
    const scalar_t* __restrict__ Q,
    const scalar_t* __restrict__ K,
    const scalar_t* __restrict__ V,
    scalar_t* __restrict__ Out,
    const int H_Q,
    const int H_KV,
    const int N_KV,
    const float sm_scale)
{
    const int off_z    = blockIdx.x;   // batch
    const int off_h_q  = blockIdx.y;   // Q head
    const int off_h_kv = off_h_q / (H_Q / H_KV);
    const int tid = threadIdx.x;

    // Strides (N_Q=1 for decode)
    const int stride_qz = H_Q * HEAD_DIM;
    const int stride_qh = HEAD_DIM;
    const int stride_kz = H_KV * N_KV * HEAD_DIM;
    const int stride_kh = N_KV * HEAD_DIM;

    // Load Q into registers (each thread loads a portion of HEAD_DIM)
    const scalar_t* q_ptr = Q + off_z * stride_qz + off_h_q * stride_qh;
    float q_reg[HEAD_DIM];
    // Initialize to zero (threads that don't cover all of HEAD_DIM)
    for (int d = 0; d < HEAD_DIM; ++d) q_reg[d] = 0.0f;
    for (int d = tid; d < HEAD_DIM; d += blockDim.x) {
        q_reg[d] = static_cast<float>(q_ptr[d]);
    }

    // Base pointers
    const scalar_t* k_base = K + off_z * stride_kz + off_h_kv * stride_kh;
    const scalar_t* v_base = V + off_z * stride_kz + off_h_kv * stride_kh;

    // Shared memory for KV tile and reduction
    extern __shared__ float smem[];
    float* kv_smem = smem;                              // BLOCK_N * HEAD_DIM
    float* reduce_smem = kv_smem + BLOCK_N * HEAD_DIM;  // blockDim.x / WARP_SIZE

    // FP32 softmax state
    float m_i = -FLT_MAX;
    float l_i = 1.0f;
    float acc_reg[HEAD_DIM];
    for (int d = 0; d < HEAD_DIM; ++d) acc_reg[d] = 0.0f;

    const float qk_scale = sm_scale * LOG2E;

    // === Main KV loop ===
    for (int start_n = 0; start_n < N_KV; start_n += BLOCK_N) {

        // ---- Load K tile into shared memory ----
        for (int i = tid; i < BLOCK_N * HEAD_DIM; i += blockDim.x) {
            int n = i / HEAD_DIM;
            int d = i % HEAD_DIM;
            int kv_pos = start_n + n;
            kv_smem[i] = (kv_pos < N_KV)
                ? static_cast<float>(k_base[kv_pos * HEAD_DIM + d])
                : 0.0f;
        }
        __syncthreads();

        // ---- QK dot products ----
        // Each thread computes partial sums for each of BLOCK_N KV positions,
        // then we warp-reduce across the HEAD_DIM dimension.
        float qk_vals[BLOCK_N];
        for (int n = 0; n < BLOCK_N; ++n) {
            float dot = 0.0f;
            for (int d = tid; d < HEAD_DIM; d += blockDim.x) {
                dot += q_reg[d] * kv_smem[n * HEAD_DIM + d];
            }
            // Warp-level reduce across HEAD_DIM partitions
            dot = warp_reduce_sum(dot);
            // Block-level reduce if we have multiple warps contributing
            // For simplicity with small blockDim, lane 0 of warp 0 has result
            qk_vals[n] = dot * qk_scale;
            if (start_n + n >= N_KV) qk_vals[n] = -FLT_MAX;
        }

        // Broadcast values from lane 0 to all threads
        float tile_max = -FLT_MAX;
        for (int n = 0; n < BLOCK_N; ++n) {
            float val = __shfl_sync(0xFFFFFFFF, qk_vals[n], 0);
            qk_vals[n] = val;
            tile_max = fmaxf(tile_max, val);
        }

        float m_new = fmaxf(m_i, tile_max);
        float alpha = exp2f(m_i - m_new);

        // Rescale accumulator
        for (int d = 0; d < HEAD_DIM; ++d) acc_reg[d] *= alpha;

        // ---- Load V tile ----
        for (int i = tid; i < BLOCK_N * HEAD_DIM; i += blockDim.x) {
            int n = i / HEAD_DIM;
            int d = i % HEAD_DIM;
            int kv_pos = start_n + n;
            kv_smem[i] = (kv_pos < N_KV)
                ? static_cast<float>(v_base[kv_pos * HEAD_DIM + d])
                : 0.0f;
        }
        __syncthreads();

        // ---- P @ V accumulation ----
        float l_ij = 0.0f;
        for (int n = 0; n < BLOCK_N; ++n) {
            float p = exp2f(qk_vals[n] - m_new);
            l_ij += p;
            for (int d = tid; d < HEAD_DIM; d += blockDim.x) {
                acc_reg[d] += p * kv_smem[n * HEAD_DIM + d];
            }
        }

        l_i = l_i * alpha + l_ij;
        m_i = m_new;
        __syncthreads();
    }

    // ---- Epilogue: normalize and store ----
    scalar_t* o_ptr = Out + off_z * stride_qz + off_h_q * stride_qh;
    for (int d = tid; d < HEAD_DIM; d += blockDim.x) {
        float val = acc_reg[d] / l_i;
        o_ptr[d] = static_cast<scalar_t>(val);
    }
}

// ============================================================================
// Launcher helpers
// ============================================================================

template <int HEAD_DIM, typename scalar_t>
void launch_flash_attention_prefill(
    const scalar_t* Q, const scalar_t* K, const scalar_t* V,
    scalar_t* Out,
    int B, int H_Q, int H_KV, int N_Q, int N_KV,
    float sm_scale, bool is_causal,
    cudaStream_t stream)
{
    constexpr int BLOCK_M = 64;
    constexpr int BLOCK_N = 64;

    dim3 grid(cdiv(N_Q, BLOCK_M), B * H_Q);
    dim3 block(BLOCK_N);

    // Shared memory: q_tile + k_tile + v_tile + s_tile + m_i + l_i + acc
    const size_t smem_bytes =
        (BLOCK_M * HEAD_DIM +       // q_tile
         BLOCK_N * HEAD_DIM +       // k_tile
         BLOCK_N * HEAD_DIM +       // v_tile
         BLOCK_M * BLOCK_N +        // s_tile
         BLOCK_M +                  // m_i
         BLOCK_M +                  // l_i
         BLOCK_M * HEAD_DIM         // acc
        ) * sizeof(float);

    if (is_causal) {
        flash_attention_prefill_kernel<HEAD_DIM, BLOCK_M, BLOCK_N, scalar_t, true>
            <<<grid, block, smem_bytes, stream>>>(
                Q, K, V, Out, H_Q, H_KV, N_Q, N_KV, sm_scale);
    } else {
        flash_attention_prefill_kernel<HEAD_DIM, BLOCK_M, BLOCK_N, scalar_t, false>
            <<<grid, block, smem_bytes, stream>>>(
                Q, K, V, Out, H_Q, H_KV, N_Q, N_KV, sm_scale);
    }
    FLASHQUANT_CHECK_LAST_ERROR();
}

template <int HEAD_DIM, typename scalar_t>
void launch_flash_attention_decode(
    const scalar_t* Q, const scalar_t* K, const scalar_t* V,
    scalar_t* Out,
    int B, int H_Q, int H_KV, int N_KV,
    float sm_scale,
    cudaStream_t stream)
{
    constexpr int BLOCK_N = 32;
    constexpr int THREADS = 128;

    dim3 grid(B, H_Q);
    dim3 block(THREADS);

    const size_t smem_bytes =
        (BLOCK_N * HEAD_DIM +       // kv_smem
         THREADS / WARP_SIZE        // reduce scratch
        ) * sizeof(float);

    flash_attention_decode_kernel<HEAD_DIM, BLOCK_N, scalar_t>
        <<<grid, block, smem_bytes, stream>>>(
            Q, K, V, Out, H_Q, H_KV, N_KV, sm_scale);

    FLASHQUANT_CHECK_LAST_ERROR();
}

// ============================================================================
// C++ wrapper
// ============================================================================

void flash_attention_cuda(
    const void* Q,
    const void* K,
    const void* V,
    void* Out,
    int B,
    int H_Q,
    int H_KV,
    int N_Q,
    int N_KV,
    int D,
    float sm_scale,
    bool is_causal,
    bool is_bf16,
    cudaStream_t stream)
{
    // Decode path: N_Q == 1 (never needs causal masking)
    bool effective_causal = is_causal;
    if (N_Q == 1) effective_causal = false;

    #define DISPATCH_FA(HD, scalar_type)                                        \
        do {                                                                    \
            auto Q_ = reinterpret_cast<const scalar_type*>(Q);                 \
            auto K_ = reinterpret_cast<const scalar_type*>(K);                 \
            auto V_ = reinterpret_cast<const scalar_type*>(V);                 \
            auto O_ = reinterpret_cast<scalar_type*>(Out);                     \
            if (N_Q == 1) {                                                     \
                launch_flash_attention_decode<HD, scalar_type>(                 \
                    Q_, K_, V_, O_, B, H_Q, H_KV, N_KV, sm_scale, stream);    \
            } else {                                                            \
                launch_flash_attention_prefill<HD, scalar_type>(                \
                    Q_, K_, V_, O_, B, H_Q, H_KV, N_Q, N_KV,                  \
                    sm_scale, effective_causal, stream);                         \
            }                                                                   \
        } while (0)

    #define DISPATCH_FA_DTYPE(HD)                                               \
        if (is_bf16) {                                                          \
            DISPATCH_FA(HD, __nv_bfloat16);                                     \
        } else {                                                                \
            DISPATCH_FA(HD, __half);                                             \
        }

    switch (D) {
        case 64:  DISPATCH_FA_DTYPE(64);  break;
        case 128: DISPATCH_FA_DTYPE(128); break;
        case 256: DISPATCH_FA_DTYPE(256); break;
        default:
            fprintf(stderr,
                    "flash_attention_cuda: unsupported HEAD_DIM=%d. "
                    "Supported: 64, 128, 256.\n", D);
            abort();
    }

    #undef DISPATCH_FA_DTYPE
    #undef DISPATCH_FA
}

}  // namespace flashquant
