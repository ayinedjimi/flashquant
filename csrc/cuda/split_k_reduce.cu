// Copyright 2026 FlashQuant Authors. Apache-2.0 License.
//
// Reduction kernel for Split-K partial softmax states.
//
// Combines NUM_SPLITS partial {acc_i, m_i, l_i} into a single final
// attention output per (sequence, query head) pair.
//
// Algorithm:
//   m_final = max(m_0, m_1, ..., m_{S-1})
//   l_final = sum_i l_i * exp(m_i - m_final)
//   acc_final = sum_i acc_i * (l_i * exp(m_i - m_final)) / l_final
//
// This is the numerically stable online softmax combination formula.
// Using exp2 + log2(e) prescaling for the same single-PTX-instruction
// trick as the attention kernel.
//
// Grid: (num_seqs * H_Q,)
// Block: (HEAD_DIM,) or (128,) -- one thread per output dimension

#include "utils.cuh"

namespace flashquant {

// ============================================================================
// Configuration
// ============================================================================

static constexpr int REDUCE_NUM_SPLITS = 4;  // Must match paged_decode.cu

// ============================================================================
// Kernel
// ============================================================================

/// Split-K reduction kernel.
///
/// Combines partial softmax states from NUM_SPLITS CTAs into one final
/// output.
///
/// Grid: (num_seqs * H_Q,)
/// Block: (THREADS,) where THREADS >= HEAD_DIM
template <int HEAD_DIM, typename scalar_t>
__global__ void split_k_reduce_kernel(
    // ---- Partial inputs from paged_decode ----
    const float* __restrict__ partial_acc,    // (num_seqs * H_Q * NUM_SPLITS, HEAD_DIM)
    const float* __restrict__ partial_m,      // (num_seqs * H_Q * NUM_SPLITS,)
    const float* __restrict__ partial_l,      // (num_seqs * H_Q * NUM_SPLITS,)
    // ---- Final output ----
    scalar_t* __restrict__ Out,               // (num_seqs, H_Q, HEAD_DIM)
    // ---- Dimensions ----
    const int num_seqs,
    const int H_Q)
{
    static constexpr int NUM_SPLITS = REDUCE_NUM_SPLITS;

    const int sh_idx = blockIdx.x;  // flattened (seq, head) index
    const int tid = threadIdx.x;

    // Base index into partial arrays
    const int base = sh_idx * NUM_SPLITS;

    // Step 1: Load partial m_i and find global max
    float m_vals[NUM_SPLITS];
    float l_vals[NUM_SPLITS];
    float m_final = -FLT_MAX;

    #pragma unroll
    for (int s = 0; s < NUM_SPLITS; ++s) {
        m_vals[s] = partial_m[base + s];
        l_vals[s] = partial_l[base + s];
        m_final = fmaxf(m_final, m_vals[s]);
    }

    // Step 2: Compute rescaled l_i values and l_final
    // Using exp2 with log2(e) prescaling for consistency with attention kernels
    float l_final = 0.0f;
    float weights[NUM_SPLITS];

    #pragma unroll
    for (int s = 0; s < NUM_SPLITS; ++s) {
        // exp(m_i - m_final) = exp2((m_i - m_final) * log2(e))
        // But m_i was already in log2 space (from the attention kernel's
        // exp2 prescaling), so we use exp2 directly.
        float scale = exp2f(m_vals[s] - m_final);
        weights[s] = l_vals[s] * scale;
        l_final += weights[s];
    }

    // Avoid division by zero
    float inv_l_final = (l_final > 0.0f) ? (1.0f / l_final) : 0.0f;

    // Step 3: Combine partial accumulators
    // Each thread handles a subset of HEAD_DIM
    for (int d = tid; d < HEAD_DIM; d += blockDim.x) {
        float combined = 0.0f;

        #pragma unroll
        for (int s = 0; s < NUM_SPLITS; ++s) {
            float a = partial_acc[(base + s) * HEAD_DIM + d];
            combined += a * weights[s];
        }

        combined *= inv_l_final;

        // Write final output
        Out[sh_idx * HEAD_DIM + d] = static_cast<scalar_t>(combined);
    }
}

// ============================================================================
// Launcher
// ============================================================================

template <int HEAD_DIM, typename scalar_t>
void launch_split_k_reduce(
    const float* partial_acc,
    const float* partial_m,
    const float* partial_l,
    scalar_t* Out,
    int num_seqs,
    int H_Q,
    cudaStream_t stream)
{
    const int grid = num_seqs * H_Q;
    const int block = min(HEAD_DIM, 256);  // Cap at 256 threads

    split_k_reduce_kernel<HEAD_DIM, scalar_t>
        <<<grid, block, 0, stream>>>(
            partial_acc, partial_m, partial_l, Out, num_seqs, H_Q);

    FLASHQUANT_CHECK_LAST_ERROR();
}

// ============================================================================
// C++ wrapper
// ============================================================================

/// Reduce Split-K partial softmax states into final attention output.
///
/// partial_acc: (num_seqs * H_Q * NUM_SPLITS * HEAD_DIM) fp32
/// partial_m:   (num_seqs * H_Q * NUM_SPLITS) fp32
/// partial_l:   (num_seqs * H_Q * NUM_SPLITS) fp32
/// Out:         (num_seqs, H_Q, HEAD_DIM) fp16/bf16
///
/// Output is in rotated space (caller must post-rotate).
void split_k_reduce_cuda(
    const float* partial_acc,
    const float* partial_m,
    const float* partial_l,
    void* Out,
    int num_seqs,
    int H_Q,
    int D,
    bool is_bf16,
    cudaStream_t stream)
{
    #define DISPATCH_REDUCE(HD, scalar_type)                                     \
        launch_split_k_reduce<HD, scalar_type>(                                 \
            partial_acc, partial_m, partial_l,                                    \
            reinterpret_cast<scalar_type*>(Out),                                \
            num_seqs, H_Q, stream)

    #define DISPATCH_REDUCE_DTYPE(HD)                                            \
        if (is_bf16) {                                                           \
            DISPATCH_REDUCE(HD, __nv_bfloat16);                                  \
        } else {                                                                 \
            DISPATCH_REDUCE(HD, __half);                                          \
        }

    switch (D) {
        case 64:  DISPATCH_REDUCE_DTYPE(64);  break;
        case 128: DISPATCH_REDUCE_DTYPE(128); break;
        case 256: DISPATCH_REDUCE_DTYPE(256); break;
        default:
            fprintf(stderr,
                    "split_k_reduce_cuda: unsupported HEAD_DIM=%d.\n", D);
            abort();
    }

    #undef DISPATCH_REDUCE_DTYPE
    #undef DISPATCH_REDUCE
}

/// Returns the number of splits (must match paged_decode).
int split_k_reduce_num_splits() {
    return REDUCE_NUM_SPLITS;
}

}  // namespace flashquant
