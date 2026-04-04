// Copyright 2026 FlashQuant Authors. Apache-2.0 License.
//
// THE critical kernel: Split-K FlashDecoding with TQ4 page tables.
//
// Grid: (num_seqs, H_Q, NUM_SPLITS)
//
// Each CTA handles a SPLIT of the KV sequence for one (seq, q_head) pair.
// Partial softmax states {acc, m_i, l_i} are written to a global scratch
// buffer and reduced by split_k_reduce.cu.
//
// Performance improvements over the Triton fused_paged_tq4_attention.py:
//   1. Split-K: NUM_SPLITS=4 -> 4x more CTAs active (critical for decode
//      where num_seqs * H_Q may not saturate SMs)
//   2. Norm load fused: reinterpret_cast<float> on uint32 load instead of
//      4 separate byte loads + manual reconstruction
//   3. BLOCK_M=1 specialized: no Q broadcast waste
//   4. sm_scale is RUNTIME (not constexpr) for flexibility
//   5. Shared memory for centroids (16 floats K + 16 floats V = 128 bytes)
//   6. Block size uses power-of-2 bit shifts (no division/modulo)
//   7. Writes partial {acc, m_i, l_i} to global scratch for reduction
//
// Cache layout per token per KV head (byte offsets):
//   [K_indices(HALF_D) | K_norm(4) | V_indices(HALF_D) | V_norm(4)]
// Total bytes per token per head = D/2 + 4 + D/2 + 4 = D + 8

#include "utils.cuh"

namespace flashquant {

// ============================================================================
// Configuration
// ============================================================================

static constexpr int PD_TQ4_LEVELS = 16;
static constexpr int PD_NUM_SPLITS = 4;  // Split-K factor

// ============================================================================
// Kernel
// ============================================================================

/// Split-K paged TQ4 decode attention kernel.
///
/// Grid: (num_seqs, H_Q, NUM_SPLITS)
/// Block: (THREADS_PER_BLOCK,)
///
/// Each CTA processes kv_start..kv_end for one (seq, head).
/// Writes partial {acc[HEAD_DIM], m_i, l_i} to global scratch.
///
/// Template parameters:
///   HEAD_DIM:   head dimension (64, 128, 256)
///   BLOCK_N:    KV tile size (power of 2)
///   BLOCK_SIZE_LOG2: log2(page block size) for bit-shift addressing
///   scalar_t:   __half or __nv_bfloat16
template <int HEAD_DIM, int BLOCK_N, int BLOCK_SIZE_LOG2, typename scalar_t>
__global__ void paged_tq4_decode_split_k_kernel(
    // ---- Inputs ----
    const scalar_t* __restrict__ Q_rot,       // (num_seqs, H_Q, HEAD_DIM)
    const uint8_t* __restrict__ KV_cache,     // (num_blocks, block_size, bytes_per_token)
    const int32_t* __restrict__ block_table,  // (num_seqs, max_blocks_per_seq)
    const int32_t* __restrict__ seq_lens,     // (num_seqs,)
    const float* __restrict__ centroids,      // (16,) shared K and V centroids
    // ---- Partial outputs (for split-K reduction) ----
    float* __restrict__ partial_acc,    // (num_seqs, H_Q, NUM_SPLITS, HEAD_DIM)
    float* __restrict__ partial_m,      // (num_seqs, H_Q, NUM_SPLITS)
    float* __restrict__ partial_l,      // (num_seqs, H_Q, NUM_SPLITS)
    // ---- Dimensions ----
    const int H_Q,
    const int H_KV,
    const int max_blocks_per_seq,
    const int stride_cache_block,     // bytes between blocks
    const int stride_cache_token,     // bytes between tokens within a block
    // ---- Per-head byte offsets within a token ----
    const int K_NORM_OFFSET,
    const int V_IDX_OFFSET,
    const int V_NORM_OFFSET,
    // ---- Runtime parameters ----
    const float sm_scale)
{
    static constexpr int HALF_D = HEAD_DIM / 2;
    static constexpr int BLOCK_SIZE = 1 << BLOCK_SIZE_LOG2;
    static constexpr int NUM_SPLITS = PD_NUM_SPLITS;

    const int off_seq    = blockIdx.x;
    const int off_h_q    = blockIdx.y;
    const int split_id   = blockIdx.z;
    const int off_h_kv   = off_h_q / (H_Q / H_KV);
    const int tid        = threadIdx.x;

    // Sequence length for this sequence
    const int seq_len = seq_lens[off_seq];

    // Compute split range
    const int chunk_size = (seq_len + NUM_SPLITS - 1) / NUM_SPLITS;
    const int kv_start = split_id * chunk_size;
    const int kv_end   = min(kv_start + chunk_size, seq_len);

    // Early exit if this split has no work
    if (kv_start >= seq_len) {
        // Write sentinel partial state
        const int part_idx = ((off_seq * H_Q + off_h_q) * NUM_SPLITS + split_id);
        partial_m[part_idx] = -FLT_MAX;
        partial_l[part_idx] = 1.0f;
        for (int d = tid; d < HEAD_DIM; d += blockDim.x) {
            partial_acc[part_idx * HEAD_DIM + d] = 0.0f;
        }
        return;
    }

    // ---- Shared memory ----
    extern __shared__ float smem[];
    float* cent_smem = smem;                              // PD_TQ4_LEVELS
    float* kv_smem = cent_smem + PD_TQ4_LEVELS;          // BLOCK_N * HEAD_DIM
    // (remaining: reduce scratch)

    // Load centroids into shared memory
    if (tid < PD_TQ4_LEVELS) {
        cent_smem[tid] = centroids[tid];
    }

    // Load Q into registers
    const scalar_t* q_ptr = Q_rot + (off_seq * H_Q + off_h_q) * HEAD_DIM;
    float q_reg[HEAD_DIM];
    for (int d = tid; d < HEAD_DIM; d += blockDim.x) {
        q_reg[d] = static_cast<float>(q_ptr[d]);
    }
    __syncthreads();

    // FP32 online softmax state
    float m_i = -FLT_MAX;
    float l_i = 1.0f;
    float acc_reg[HEAD_DIM];
    for (int d = 0; d < HEAD_DIM; ++d) acc_reg[d] = 0.0f;

    const float qk_scale = sm_scale * LOG2E;

    // Block table base for this sequence
    const int32_t* bt_base = block_table + off_seq * max_blocks_per_seq;

    // === Main KV tile loop (over this split's range) ===
    for (int start_n = kv_start; start_n < kv_end; start_n += BLOCK_N) {

        // ---- Decompress K tile ----
        for (int i = tid; i < BLOCK_N; i += blockDim.x) {
            int kv_pos = start_n + i;
            if (kv_pos < kv_end) {
                // Page table lookup using bit shifts (no division/modulo)
                int logical_block = kv_pos >> BLOCK_SIZE_LOG2;
                int within_block  = kv_pos & (BLOCK_SIZE - 1);
                int physical_block = bt_base[logical_block];

                // Compute byte address for this token
                int token_byte_base = physical_block * stride_cache_block
                                    + within_block * stride_cache_token;

                // Load K packed indices
                const uint8_t* k_packed_ptr = KV_cache + token_byte_base
                                            + off_h_kv * HALF_D;

                // Load K norm: fused uint32 load + reinterpret as float
                // (FIX #2: single 4-byte load instead of 4 separate byte loads)
                const uint8_t* k_norm_ptr = KV_cache + token_byte_base
                                          + K_NORM_OFFSET + off_h_kv * 4;
                uint32_t k_norm_bits = *reinterpret_cast<const uint32_t*>(k_norm_ptr);
                float k_norm = uint32_as_float(k_norm_bits);

                // Decompress K: sequential layout [hi(HALF_D) | lo(HALF_D)]
                for (int j = 0; j < HALF_D; ++j) {
                    uint8_t p = k_packed_ptr[j];
                    int hi = (p >> 4) & 0xF;
                    int lo =  p       & 0xF;
                    kv_smem[i * HEAD_DIM + j]          = cent_smem[hi] * k_norm;
                    kv_smem[i * HEAD_DIM + HALF_D + j] = cent_smem[lo] * k_norm;
                }
            } else {
                for (int d = 0; d < HEAD_DIM; ++d) {
                    kv_smem[i * HEAD_DIM + d] = 0.0f;
                }
            }
        }
        __syncthreads();

        // ---- Q @ K^T dot products ----
        float qk_vals[BLOCK_N];
        for (int n = 0; n < BLOCK_N; ++n) {
            float dot = 0.0f;
            for (int d = tid; d < HEAD_DIM; d += blockDim.x) {
                dot += q_reg[d] * kv_smem[n * HEAD_DIM + d];
            }
            // Warp-level sum (for decode, typically 1 warp is enough)
            dot = warp_reduce_sum(dot);
            qk_vals[n] = dot * qk_scale;

            if (start_n + n >= kv_end) {
                qk_vals[n] = -FLT_MAX;
            }
        }

        // Broadcast from lane 0
        float tile_max = -FLT_MAX;
        for (int n = 0; n < BLOCK_N; ++n) {
            qk_vals[n] = __shfl_sync(0xFFFFFFFF, qk_vals[n], 0);
            tile_max = fmaxf(tile_max, qk_vals[n]);
        }

        float m_new = fmaxf(m_i, tile_max);
        float alpha = exp2f(m_i - m_new);

        for (int d = 0; d < HEAD_DIM; ++d) {
            acc_reg[d] *= alpha;
        }

        // ---- Decompress V tile ----
        for (int i = tid; i < BLOCK_N; i += blockDim.x) {
            int kv_pos = start_n + i;
            if (kv_pos < kv_end) {
                int logical_block = kv_pos >> BLOCK_SIZE_LOG2;
                int within_block  = kv_pos & (BLOCK_SIZE - 1);
                int physical_block = bt_base[logical_block];

                int token_byte_base = physical_block * stride_cache_block
                                    + within_block * stride_cache_token;

                const uint8_t* v_packed_ptr = KV_cache + token_byte_base
                                            + V_IDX_OFFSET + off_h_kv * HALF_D;

                const uint8_t* v_norm_ptr = KV_cache + token_byte_base
                                          + V_NORM_OFFSET + off_h_kv * 4;
                uint32_t v_norm_bits = *reinterpret_cast<const uint32_t*>(v_norm_ptr);
                float v_norm = uint32_as_float(v_norm_bits);

                for (int j = 0; j < HALF_D; ++j) {
                    uint8_t p = v_packed_ptr[j];
                    int hi = (p >> 4) & 0xF;
                    int lo =  p       & 0xF;
                    kv_smem[i * HEAD_DIM + j]          = cent_smem[hi] * v_norm;
                    kv_smem[i * HEAD_DIM + HALF_D + j] = cent_smem[lo] * v_norm;
                }
            } else {
                for (int d = 0; d < HEAD_DIM; ++d) {
                    kv_smem[i * HEAD_DIM + d] = 0.0f;
                }
            }
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

    // ---- Write partial results to global memory ----
    const int part_idx = ((off_seq * H_Q + off_h_q) * NUM_SPLITS + split_id);

    // Only lane 0 writes scalar state (m_i, l_i are identical across lanes
    // after the warp reductions above, but only lane 0 is guaranteed correct
    // for the final reduced value).
    if (tid == 0) {
        partial_m[part_idx] = m_i;
        partial_l[part_idx] = l_i;
    }

    // All threads cooperate to write acc
    for (int d = tid; d < HEAD_DIM; d += blockDim.x) {
        partial_acc[part_idx * HEAD_DIM + d] = acc_reg[d];
    }
}

// ============================================================================
// Launcher
// ============================================================================

template <int HEAD_DIM, int BLOCK_SIZE_LOG2, typename scalar_t>
void launch_paged_decode(
    const scalar_t* Q_rot,
    const uint8_t* KV_cache,
    const int32_t* block_table,
    const int32_t* seq_lens,
    const float* centroids,
    float* partial_acc,
    float* partial_m,
    float* partial_l,
    int num_seqs,
    int H_Q,
    int H_KV,
    int max_blocks_per_seq,
    int stride_cache_block,
    int stride_cache_token,
    int K_NORM_OFFSET,
    int V_IDX_OFFSET,
    int V_NORM_OFFSET,
    float sm_scale,
    cudaStream_t stream)
{
    constexpr int BLOCK_N = 32;
    constexpr int THREADS = 128;
    constexpr int NUM_SPLITS = PD_NUM_SPLITS;

    dim3 grid(num_seqs, H_Q, NUM_SPLITS);
    dim3 block(THREADS);

    const size_t smem_bytes =
        (PD_TQ4_LEVELS +               // centroids
         BLOCK_N * HEAD_DIM             // kv_smem
        ) * sizeof(float);

    paged_tq4_decode_split_k_kernel<HEAD_DIM, BLOCK_N, BLOCK_SIZE_LOG2, scalar_t>
        <<<grid, block, smem_bytes, stream>>>(
            Q_rot, KV_cache, block_table, seq_lens, centroids,
            partial_acc, partial_m, partial_l,
            H_Q, H_KV, max_blocks_per_seq,
            stride_cache_block, stride_cache_token,
            K_NORM_OFFSET, V_IDX_OFFSET, V_NORM_OFFSET,
            sm_scale);

    FLASHQUANT_CHECK_LAST_ERROR();
}

// ============================================================================
// C++ wrapper
// ============================================================================

/// Launch Split-K paged TQ4 decode attention.
///
/// Q_rot: (num_seqs, H_Q, HEAD_DIM) pre-rotated queries
/// KV_cache: (num_blocks, block_size, bytes_per_token) packed paged cache
/// block_table: (num_seqs, max_blocks_per_seq) int32 page table
/// seq_lens: (num_seqs,) int32
/// centroids: (16,) fp32 shared codebook
///
/// Partial outputs (caller must allocate):
///   partial_acc: (num_seqs * H_Q * NUM_SPLITS * HEAD_DIM) fp32
///   partial_m:   (num_seqs * H_Q * NUM_SPLITS) fp32
///   partial_l:   (num_seqs * H_Q * NUM_SPLITS) fp32
///
/// After this kernel, call split_k_reduce_cuda() to combine partials.
void paged_tq4_decode_cuda(
    const void* Q_rot,
    const uint8_t* KV_cache,
    const int32_t* block_table,
    const int32_t* seq_lens,
    const float* centroids,
    float* partial_acc,
    float* partial_m,
    float* partial_l,
    int num_seqs,
    int H_Q,
    int H_KV,
    int D,
    int block_size,
    int max_blocks_per_seq,
    int stride_cache_block,
    int stride_cache_token,
    int K_NORM_OFFSET,
    int V_IDX_OFFSET,
    int V_NORM_OFFSET,
    float sm_scale,
    bool is_bf16,
    cudaStream_t stream)
{
    // Compute log2 of block_size using bit operations
    int block_size_log2 = 0;
    {
        int tmp = block_size;
        while (tmp > 1) { tmp >>= 1; ++block_size_log2; }
    }

    // Dispatch on HEAD_DIM, BLOCK_SIZE_LOG2, and dtype
    #define DISPATCH_PD(HD, BSL, scalar_type)                                   \
        launch_paged_decode<HD, BSL, scalar_type>(                              \
            reinterpret_cast<const scalar_type*>(Q_rot),                        \
            KV_cache, block_table, seq_lens, centroids,                         \
            partial_acc, partial_m, partial_l,                                   \
            num_seqs, H_Q, H_KV, max_blocks_per_seq,                           \
            stride_cache_block, stride_cache_token,                              \
            K_NORM_OFFSET, V_IDX_OFFSET, V_NORM_OFFSET,                        \
            sm_scale, stream)

    #define DISPATCH_PD_DTYPE(HD, BSL)                                          \
        if (is_bf16) {                                                          \
            DISPATCH_PD(HD, BSL, __nv_bfloat16);                                \
        } else {                                                                \
            DISPATCH_PD(HD, BSL, __half);                                        \
        }

    // Macro for block_size_log2 dispatch
    #define DISPATCH_PD_BSL(HD)                                                 \
        switch (block_size_log2) {                                              \
            case 3: DISPATCH_PD_DTYPE(HD, 3); break;  /* block_size=8  */       \
            case 4: DISPATCH_PD_DTYPE(HD, 4); break;  /* block_size=16 */       \
            case 5: DISPATCH_PD_DTYPE(HD, 5); break;  /* block_size=32 */       \
            default:                                                            \
                fprintf(stderr,                                                 \
                        "paged_tq4_decode_cuda: unsupported block_size=%d "     \
                        "(must be 8, 16, or 32).\n", block_size);              \
                abort();                                                        \
        }

    switch (D) {
        case 64:  DISPATCH_PD_BSL(64);  break;
        case 128: DISPATCH_PD_BSL(128); break;
        case 256: DISPATCH_PD_BSL(256); break;
        default:
            fprintf(stderr,
                    "paged_tq4_decode_cuda: unsupported HEAD_DIM=%d.\n", D);
            abort();
    }

    #undef DISPATCH_PD_BSL
    #undef DISPATCH_PD_DTYPE
    #undef DISPATCH_PD
}

/// Returns the number of splits used (compile-time constant).
int paged_tq4_decode_num_splits() {
    return PD_NUM_SPLITS;
}

}  // namespace flashquant
