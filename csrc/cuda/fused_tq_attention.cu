// Copyright 2026 FlashQuant Authors. Apache-2.0 License.
//
// FlashAttention-2 with inline TQ4 K/V decompression.
//
// Fuses decompression into the attention inner loop to avoid materializing
// decompressed K/V in HBM. Q is pre-rotated by Pi^T (done outside this
// kernel). Output is in rotated space (caller applies post-rotation).
//
// Inner loop per KV tile:
//   1. Load packed K bytes -> decompress (unpack + centroid gather + norm scale)
//   2. Compute Q @ K^T -> online softmax
//   3. Load packed V bytes -> decompress
//   4. Accumulate P @ V
//
// Separate K and V centroids in shared memory (2 * 16 * sizeof(float) = 128 bytes).
// FP32 accumulators throughout.
// Causal masking with tile-skipping.

#include "utils.cuh"

namespace flashquant {

// ============================================================================
// Configuration
// ============================================================================

static constexpr int TQ4_LEVELS = 16;

// ============================================================================
// Kernel: Prefill path (BLOCK_M > 1)
// ============================================================================

/// FlashAttention-2 + inline TQ4 decompression (prefill).
///
/// Grid: (ceil(N_Q / BLOCK_M), B * H_Q)
/// Block: (BLOCK_N,)
///
/// Q_rot: (B, H_Q, N_Q, D) pre-rotated queries (fp16/bf16)
/// K_packed: (B, H_KV, N_KV, HALF_D) uint8 nibble-packed K indices
/// K_norms: (B, H_KV, N_KV) fp32 K norms
/// V_packed: (B, H_KV, N_KV, HALF_D) uint8 nibble-packed V indices
/// V_norms: (B, H_KV, N_KV) fp32 V norms
/// K_centroids: (16,) fp32 K centroid table
/// V_centroids: (16,) fp32 V centroid table
/// Out: (B, H_Q, N_Q, D) output in rotated space
///
/// Decompress layout (coalesced): [hi_values(HALF_D) | lo_values(HALF_D)]
template <int HEAD_DIM, int BLOCK_M, int BLOCK_N, typename scalar_t, bool IS_CAUSAL>
__global__ void fused_tq_attention_prefill_kernel(
    const scalar_t* __restrict__ Q_rot,
    const uint8_t* __restrict__ K_packed,
    const float* __restrict__ K_norms,
    const uint8_t* __restrict__ V_packed,
    const float* __restrict__ V_norms,
    const float* __restrict__ K_centroids,
    const float* __restrict__ V_centroids,
    scalar_t* __restrict__ Out,
    const int H_Q,
    const int H_KV,
    const int N_Q,
    const int N_KV,
    const float sm_scale)
{
    static constexpr int HALF_D = HEAD_DIM / 2;

    const int start_m = blockIdx.x;
    const int off_hz  = blockIdx.y;
    const int off_z   = off_hz / H_Q;
    const int off_h_q = off_hz % H_Q;
    const int off_h_kv = off_h_q / (H_Q / H_KV);
    const int tid = threadIdx.x;

    // Strides (contiguous last dim)
    const int stride_qz = H_Q * N_Q * HEAD_DIM;
    const int stride_qh = N_Q * HEAD_DIM;
    const int stride_kpz = H_KV * N_KV * HALF_D;
    const int stride_kph = N_KV * HALF_D;
    const int stride_knz = H_KV * N_KV;
    const int stride_knh = N_KV;

    // Base pointers
    const scalar_t* q_base = Q_rot + off_z * stride_qz + off_h_q * stride_qh;
    const uint8_t* kp_base = K_packed + off_z * stride_kpz + off_h_kv * stride_kph;
    const float* kn_base = K_norms + off_z * stride_knz + off_h_kv * stride_knh;
    const uint8_t* vp_base = V_packed + off_z * stride_kpz + off_h_kv * stride_kph;
    const float* vn_base = V_norms + off_z * stride_knz + off_h_kv * stride_knh;
    scalar_t* o_base = Out + off_z * stride_qz + off_h_q * stride_qh;

    // Shared memory layout
    extern __shared__ float smem[];
    float* k_centroids_smem = smem;                            // TQ4_LEVELS
    float* v_centroids_smem = k_centroids_smem + TQ4_LEVELS;  // TQ4_LEVELS
    float* q_tile = v_centroids_smem + TQ4_LEVELS;            // BLOCK_M * HEAD_DIM
    float* kv_tile = q_tile + BLOCK_M * HEAD_DIM;             // BLOCK_N * HEAD_DIM
    float* s_tile = kv_tile + BLOCK_N * HEAD_DIM;             // BLOCK_M * BLOCK_N
    float* m_i = s_tile + BLOCK_M * BLOCK_N;                  // BLOCK_M
    float* l_i = m_i + BLOCK_M;                               // BLOCK_M
    float* acc = l_i + BLOCK_M;                                // BLOCK_M * HEAD_DIM

    // Load centroids into shared memory
    if (tid < TQ4_LEVELS) {
        k_centroids_smem[tid] = K_centroids[tid];
        v_centroids_smem[tid] = V_centroids[tid];
    }

    // Load Q tile
    for (int i = tid; i < BLOCK_M * HEAD_DIM; i += BLOCK_N) {
        int m = i / HEAD_DIM;
        int d = i % HEAD_DIM;
        int q_row = start_m * BLOCK_M + m;
        q_tile[i] = (q_row < N_Q) ? static_cast<float>(q_base[q_row * HEAD_DIM + d]) : 0.0f;
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

    int hi = IS_CAUSAL ? min((start_m + 1) * BLOCK_M, N_KV) : N_KV;

    // === Main tile loop ===
    for (int start_n = 0; start_n < hi; start_n += BLOCK_N) {

        // ---- Decompress K tile into kv_tile [BLOCK_N, HEAD_DIM] ----
        // Each thread handles one KV position (tid = n within tile)
        {
            int n = tid;
            int kv_pos = start_n + n;
            if (kv_pos < N_KV) {
                float k_norm = kn_base[kv_pos];
                const uint8_t* kp_row = kp_base + kv_pos * HALF_D;

                // Decompress: sequential layout [hi(HALF_D) | lo(HALF_D)]
                for (int j = 0; j < HALF_D; ++j) {
                    uint8_t packed = kp_row[j];
                    int hi_idx = (packed >> 4) & 0xF;
                    int lo_idx =  packed       & 0xF;
                    kv_tile[n * HEAD_DIM + j]          = k_centroids_smem[hi_idx] * k_norm;
                    kv_tile[n * HEAD_DIM + HALF_D + j] = k_centroids_smem[lo_idx] * k_norm;
                }
            } else {
                for (int d = 0; d < HEAD_DIM; ++d) {
                    kv_tile[n * HEAD_DIM + d] = 0.0f;
                }
            }
        }
        __syncthreads();

        // ---- Compute S = Q_rot @ K^T [BLOCK_M, BLOCK_N] ----
        for (int m = 0; m < BLOCK_M; ++m) {
            float dot = 0.0f;
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) {
                dot += q_tile[m * HEAD_DIM + d] * kv_tile[tid * HEAD_DIM + d];
            }
            dot *= qk_scale;

            if (IS_CAUSAL) {
                if (start_n + tid > start_m * BLOCK_M + m) dot = -FLT_MAX;
            }
            if (start_n + tid >= N_KV) dot = -FLT_MAX;

            s_tile[m * BLOCK_N + tid] = dot;
        }
        __syncthreads();

        // ---- Decompress V tile into kv_tile [BLOCK_N, HEAD_DIM] ----
        {
            int n = tid;
            int kv_pos = start_n + n;
            if (kv_pos < N_KV) {
                float v_norm = vn_base[kv_pos];
                const uint8_t* vp_row = vp_base + kv_pos * HALF_D;

                for (int j = 0; j < HALF_D; ++j) {
                    uint8_t packed = vp_row[j];
                    int hi_idx = (packed >> 4) & 0xF;
                    int lo_idx =  packed       & 0xF;
                    kv_tile[n * HEAD_DIM + j]          = v_centroids_smem[hi_idx] * v_norm;
                    kv_tile[n * HEAD_DIM + HALF_D + j] = v_centroids_smem[lo_idx] * v_norm;
                }
            } else {
                for (int d = 0; d < HEAD_DIM; ++d) {
                    kv_tile[n * HEAD_DIM + d] = 0.0f;
                }
            }
        }
        __syncthreads();

        // ---- Online softmax + P @ V ----
        for (int m = tid; m < BLOCK_M; m += BLOCK_N) {
            float row_max = -FLT_MAX;
            for (int n = 0; n < BLOCK_N; ++n) {
                row_max = fmaxf(row_max, s_tile[m * BLOCK_N + n]);
            }

            float m_new = fmaxf(m_i[m], row_max);
            float alpha = exp2f(m_i[m] - m_new);

            #pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) {
                acc[m * HEAD_DIM + d] *= alpha;
            }

            float l_ij = 0.0f;
            for (int n = 0; n < BLOCK_N; ++n) {
                float p = exp2f(s_tile[m * BLOCK_N + n] - m_new);
                l_ij += p;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    acc[m * HEAD_DIM + d] += p * kv_tile[n * HEAD_DIM + d];
                }
            }

            l_i[m] = l_i[m] * alpha + l_ij;
            m_i[m] = m_new;
        }
        __syncthreads();
    }

    // ---- Epilogue ----
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
// Kernel: Decode path (BLOCK_M=1, specialized)
// ============================================================================

/// FlashAttention-2 + TQ4 decode kernel (single query token).
///
/// Grid: (B, H_Q)
/// Block: (THREADS,)
template <int HEAD_DIM, int BLOCK_N, typename scalar_t>
__global__ void fused_tq_attention_decode_kernel(
    const scalar_t* __restrict__ Q_rot,
    const uint8_t* __restrict__ K_packed,
    const float* __restrict__ K_norms,
    const uint8_t* __restrict__ V_packed,
    const float* __restrict__ V_norms,
    const float* __restrict__ K_centroids,
    const float* __restrict__ V_centroids,
    scalar_t* __restrict__ Out,
    const int H_Q,
    const int H_KV,
    const int N_KV,
    const float sm_scale)
{
    static constexpr int HALF_D = HEAD_DIM / 2;

    const int off_z   = blockIdx.x;
    const int off_h_q = blockIdx.y;
    const int off_h_kv = off_h_q / (H_Q / H_KV);
    const int tid = threadIdx.x;

    const int stride_qz = H_Q * HEAD_DIM;
    const int stride_kpz = H_KV * N_KV * HALF_D;
    const int stride_kph = N_KV * HALF_D;
    const int stride_knz = H_KV * N_KV;
    const int stride_knh = N_KV;

    const scalar_t* q_ptr = Q_rot + off_z * stride_qz + off_h_q * HEAD_DIM;
    const uint8_t* kp_base = K_packed + off_z * stride_kpz + off_h_kv * stride_kph;
    const float* kn_base = K_norms + off_z * stride_knz + off_h_kv * stride_knh;
    const uint8_t* vp_base = V_packed + off_z * stride_kpz + off_h_kv * stride_kph;
    const float* vn_base = V_norms + off_z * stride_knz + off_h_kv * stride_knh;

    // Shared memory
    extern __shared__ float smem[];
    float* k_cent = smem;                              // 16
    float* v_cent = k_cent + TQ4_LEVELS;               // 16
    float* kv_smem = v_cent + TQ4_LEVELS;              // BLOCK_N * HEAD_DIM
    float* reduce_smem = kv_smem + BLOCK_N * HEAD_DIM; // small

    // Load centroids
    if (tid < TQ4_LEVELS) {
        k_cent[tid] = K_centroids[tid];
        v_cent[tid] = V_centroids[tid];
    }

    // Load Q into registers
    float q_reg[HEAD_DIM];
    for (int d = tid; d < HEAD_DIM; d += blockDim.x) {
        q_reg[d] = static_cast<float>(q_ptr[d]);
    }
    __syncthreads();

    float m_i = -FLT_MAX;
    float l_i = 1.0f;
    float acc_reg[HEAD_DIM];
    for (int d = 0; d < HEAD_DIM; ++d) acc_reg[d] = 0.0f;

    const float qk_scale = sm_scale * LOG2E;

    for (int start_n = 0; start_n < N_KV; start_n += BLOCK_N) {

        // Decompress K tile into shared memory
        for (int i = tid; i < BLOCK_N; i += blockDim.x) {
            int kv_pos = start_n + i;
            if (kv_pos < N_KV) {
                float k_norm = kn_base[kv_pos];
                const uint8_t* kp_row = kp_base + kv_pos * HALF_D;
                for (int j = 0; j < HALF_D; ++j) {
                    uint8_t p = kp_row[j];
                    kv_smem[i * HEAD_DIM + j]          = k_cent[(p >> 4) & 0xF] * k_norm;
                    kv_smem[i * HEAD_DIM + HALF_D + j] = k_cent[p & 0xF] * k_norm;
                }
            } else {
                for (int d = 0; d < HEAD_DIM; ++d)
                    kv_smem[i * HEAD_DIM + d] = 0.0f;
            }
        }
        __syncthreads();

        // QK dot products
        float qk_vals[BLOCK_N];
        for (int n = 0; n < BLOCK_N; ++n) {
            float dot = 0.0f;
            for (int d = tid; d < HEAD_DIM; d += blockDim.x) {
                dot += q_reg[d] * kv_smem[n * HEAD_DIM + d];
            }
            dot = warp_reduce_sum(dot);
            qk_vals[n] = dot * qk_scale;
            if (start_n + n >= N_KV) qk_vals[n] = -FLT_MAX;
        }

        // Broadcast from lane 0
        float tile_max = -FLT_MAX;
        for (int n = 0; n < BLOCK_N; ++n) {
            qk_vals[n] = __shfl_sync(0xFFFFFFFF, qk_vals[n], 0);
            tile_max = fmaxf(tile_max, qk_vals[n]);
        }

        float m_new = fmaxf(m_i, tile_max);
        float alpha = exp2f(m_i - m_new);
        for (int d = 0; d < HEAD_DIM; ++d) acc_reg[d] *= alpha;

        // Decompress V and accumulate P @ V
        for (int i = tid; i < BLOCK_N; i += blockDim.x) {
            int kv_pos = start_n + i;
            if (kv_pos < N_KV) {
                float v_norm = vn_base[kv_pos];
                const uint8_t* vp_row = vp_base + kv_pos * HALF_D;
                for (int j = 0; j < HALF_D; ++j) {
                    uint8_t p = vp_row[j];
                    kv_smem[i * HEAD_DIM + j]          = v_cent[(p >> 4) & 0xF] * v_norm;
                    kv_smem[i * HEAD_DIM + HALF_D + j] = v_cent[p & 0xF] * v_norm;
                }
            } else {
                for (int d = 0; d < HEAD_DIM; ++d)
                    kv_smem[i * HEAD_DIM + d] = 0.0f;
            }
        }
        __syncthreads();

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

    // Epilogue
    scalar_t* o_ptr = Out + off_z * stride_qz + off_h_q * HEAD_DIM;
    for (int d = tid; d < HEAD_DIM; d += blockDim.x) {
        o_ptr[d] = static_cast<scalar_t>(acc_reg[d] / l_i);
    }
}

// ============================================================================
// Launcher
// ============================================================================

template <int HEAD_DIM, typename scalar_t>
void launch_fused_tq_attention(
    const scalar_t* Q_rot,
    const uint8_t* K_packed,
    const float* K_norms,
    const uint8_t* V_packed,
    const float* V_norms,
    const float* K_centroids,
    const float* V_centroids,
    scalar_t* Out,
    int B, int H_Q, int H_KV, int N_Q, int N_KV,
    float sm_scale, bool is_causal,
    cudaStream_t stream)
{
    static constexpr int HALF_D = HEAD_DIM / 2;

    if (N_Q == 1) {
        // Decode path
        constexpr int BLOCK_N = 32;
        constexpr int THREADS = 128;
        dim3 grid(B, H_Q);
        dim3 block(THREADS);

        const size_t smem_bytes =
            (2 * TQ4_LEVELS +
             BLOCK_N * HEAD_DIM +
             THREADS / WARP_SIZE) * sizeof(float);

        fused_tq_attention_decode_kernel<HEAD_DIM, BLOCK_N, scalar_t>
            <<<grid, block, smem_bytes, stream>>>(
                Q_rot, K_packed, K_norms, V_packed, V_norms,
                K_centroids, V_centroids, Out,
                H_Q, H_KV, N_KV, sm_scale);
    } else {
        // Prefill path
        constexpr int BLOCK_M = 64;
        constexpr int BLOCK_N = 64;
        dim3 grid(cdiv(N_Q, BLOCK_M), B * H_Q);
        dim3 block(BLOCK_N);

        const size_t smem_bytes =
            (2 * TQ4_LEVELS +
             BLOCK_M * HEAD_DIM +
             BLOCK_N * HEAD_DIM +
             BLOCK_M * BLOCK_N +
             BLOCK_M + BLOCK_M +
             BLOCK_M * HEAD_DIM) * sizeof(float);

        if (is_causal) {
            fused_tq_attention_prefill_kernel<HEAD_DIM, BLOCK_M, BLOCK_N, scalar_t, true>
                <<<grid, block, smem_bytes, stream>>>(
                    Q_rot, K_packed, K_norms, V_packed, V_norms,
                    K_centroids, V_centroids, Out,
                    H_Q, H_KV, N_Q, N_KV, sm_scale);
        } else {
            fused_tq_attention_prefill_kernel<HEAD_DIM, BLOCK_M, BLOCK_N, scalar_t, false>
                <<<grid, block, smem_bytes, stream>>>(
                    Q_rot, K_packed, K_norms, V_packed, V_norms,
                    K_centroids, V_centroids, Out,
                    H_Q, H_KV, N_Q, N_KV, sm_scale);
        }
    }
    FLASHQUANT_CHECK_LAST_ERROR();
}

// ============================================================================
// C++ wrapper
// ============================================================================

/// Launch fused TQ4 Flash Attention.
///
/// Q_rot must be pre-rotated by Pi^T. Output is in rotated space.
/// K and V centroids can be the same or different tensors.
void fused_tq_attention_cuda(
    const void* Q_rot,
    const uint8_t* K_packed,
    const float* K_norms,
    const uint8_t* V_packed,
    const float* V_norms,
    const float* K_centroids,
    const float* V_centroids,
    void* Out,
    int B, int H_Q, int H_KV, int N_Q, int N_KV, int D,
    float sm_scale, bool is_causal, bool is_bf16,
    cudaStream_t stream)
{
    if (N_Q == 1) is_causal = false;

    #define DISPATCH_TQ_FA(HD, scalar_type)                                     \
        launch_fused_tq_attention<HD, scalar_type>(                             \
            reinterpret_cast<const scalar_type*>(Q_rot),                        \
            K_packed, K_norms, V_packed, V_norms,                               \
            K_centroids, V_centroids,                                            \
            reinterpret_cast<scalar_type*>(Out),                                \
            B, H_Q, H_KV, N_Q, N_KV,                                           \
            sm_scale, is_causal, stream)

    #define DISPATCH_TQ_FA_DTYPE(HD)                                            \
        if (is_bf16) {                                                          \
            DISPATCH_TQ_FA(HD, __nv_bfloat16);                                  \
        } else {                                                                \
            DISPATCH_TQ_FA(HD, __half);                                          \
        }

    switch (D) {
        case 64:  DISPATCH_TQ_FA_DTYPE(64);  break;
        case 128: DISPATCH_TQ_FA_DTYPE(128); break;
        case 256: DISPATCH_TQ_FA_DTYPE(256); break;
        default:
            fprintf(stderr,
                    "fused_tq_attention_cuda: unsupported HEAD_DIM=%d.\n", D);
            abort();
    }

    #undef DISPATCH_TQ_FA_DTYPE
    #undef DISPATCH_TQ_FA
}

}  // namespace flashquant
