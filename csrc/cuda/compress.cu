// Copyright 2026 FlashQuant Authors. Apache-2.0 License.
//
// Fused TQ4 compress kernel: norm + rotate + quantize + nibble-pack.
//
// Single kernel replaces the 6-kernel PyTorch path (norm, divide, matmul,
// bucketize, clamp, pack). Each CTA handles ROWS_PER_BLOCK rows.
//
// Performance fixes over the Triton version:
//   1. Multi-row blocks: ROWS_PER_BLOCK=4 for small M (1-token decode),
//      keeping SMs occupied instead of launching M=4 CTAs on a 128-SM GPU.
//   2. Rotation tile in shared memory: avoids redundant global loads when
//      multiple rows share the same rotation tile.
//   3. Single-pass norm: compute L2 norm from input loaded into shared
//      memory -- no second read from global memory.
//   4. Template on HALF_D: enables compile-time loop unrolling for the
//      inner dimension, critical for the rotation matmul.

#include "compress.cuh"
#include "utils.cuh"

namespace flashquant {

// ============================================================================
// Configuration
// ============================================================================

static constexpr int ROWS_PER_BLOCK = 4;
static constexpr int N_BOUNDARIES   = 15;  // TQ4: 16 levels -> 15 boundaries
static constexpr int BLOCK_K        = 32;  // Tile size for rotation matmul

// ============================================================================
// Kernel
// ============================================================================

/// Fused TQ4 compress kernel templated on HALF_D and input type.
///
/// Grid:  (ceil(M / ROWS_PER_BLOCK),)
/// Block: (HALF_D,) -- one thread per output column
///
/// Shared memory layout:
///   float smem_x[D]                    -- normalized input row (x_hat)
///   float smem_rot[BLOCK_K * HALF_D]   -- rotation tile (even or odd)
///   float smem_boundaries[N_BOUNDARIES] -- quantization boundaries
///   float smem_reduce[num_warps]        -- norm reduction scratch
///
/// Algorithm per row:
///   1. Cooperatively load x into smem, compute L2 norm via reduction
///   2. Normalize x_hat = x / norm (in-place in smem)
///   3. Tiled rotation: result[tid] = sum_k x_hat[k] * Rot[k, tid]
///      Load Rot tiles into smem, read x_hat from smem
///   4. Bucketize against boundaries in smem
///   5. Pack nibbles and store
template <int HALF_D, typename scalar_t>
__global__ void tq4_compress_kernel(
    const scalar_t* __restrict__ x,          // (M, D=2*HALF_D)
    const float* __restrict__ rot_T_even,    // (D, HALF_D)
    const float* __restrict__ rot_T_odd,     // (D, HALF_D)
    const float* __restrict__ boundaries,    // (N_BOUNDARIES,)
    uint8_t* __restrict__ packed,            // (M, HALF_D)
    float* __restrict__ norms_out,           // (M,)
    const int M)
{
    static constexpr int D = HALF_D * 2;

    const int block_row_start = blockIdx.x * ROWS_PER_BLOCK;
    const int tid = threadIdx.x;  // [0, HALF_D)

    // Shared memory layout
    extern __shared__ float smem[];
    float* smem_x          = smem;                                 // D
    float* smem_rot        = smem_x + D;                           // BLOCK_K * HALF_D
    float* smem_boundaries = smem_rot + BLOCK_K * HALF_D;          // N_BOUNDARIES
    float* smem_reduce     = smem_boundaries + N_BOUNDARIES;       // num_warps

    // Load boundaries into shared memory (only first N_BOUNDARIES threads)
    if (tid < N_BOUNDARIES) {
        smem_boundaries[tid] = boundaries[tid];
    }
    __syncthreads();

    // Process ROWS_PER_BLOCK rows per CTA
    for (int local_row = 0; local_row < ROWS_PER_BLOCK; ++local_row) {
        const int row = block_row_start + local_row;
        if (row >= M) return;

        const scalar_t* x_row = x + row * D;

        // ---- Step 1: Load input into shared memory and compute L2 norm ----
        // Each thread loads D/HALF_D = 2 elements and computes partial sum.
        float partial_sq = 0.0f;

        #pragma unroll
        for (int i = tid; i < D; i += HALF_D) {
            float val = static_cast<float>(x_row[i]);
            smem_x[i] = val;
            partial_sq += val * val;
        }

        // Block-level sum reduction for norm^2
        float sum_sq = block_reduce_sum(partial_sq, smem_reduce);

        // Broadcast norm to all threads and normalize x in shared memory
        __shared__ float shared_norm;
        __shared__ float shared_inv_norm;
        if (tid == 0) {
            float n = sqrtf(sum_sq);
            shared_norm = n;
            shared_inv_norm = 1.0f / (n + 1e-10f);
        }
        __syncthreads();

        float inv_norm = shared_inv_norm;

        // Normalize in shared memory: smem_x[i] = x[i] / norm
        #pragma unroll
        for (int i = tid; i < D; i += HALF_D) {
            smem_x[i] *= inv_norm;
        }
        __syncthreads();

        // ---- Step 2: Tiled rotation matmul ----
        // result_even[tid] = sum_k x_hat[k] * R_T_even[k, tid]
        // result_odd[tid]  = sum_k x_hat[k] * R_T_odd[k, tid]
        //
        // Strategy: for each BLOCK_K tile of k:
        //   - Load rotation tile [BLOCK_K, HALF_D] into smem_rot cooperatively
        //   - Each thread reads x_hat[k] from smem_x and rot[k, tid] from smem_rot
        //   - Accumulate dot product
        float rotated_even = 0.0f;
        float rotated_odd  = 0.0f;

        for (int k_start = 0; k_start < D; k_start += BLOCK_K) {

            // -- Load even rotation tile [BLOCK_K, HALF_D] --
            // Cooperative load: each of HALF_D threads loads BLOCK_K values
            #pragma unroll
            for (int k_local = 0; k_local < BLOCK_K; ++k_local) {
                int k = k_start + k_local;
                if (k < D) {
                    smem_rot[k_local * HALF_D + tid] =
                        rot_T_even[k * HALF_D + tid];
                }
            }
            __syncthreads();

            // Accumulate: each thread computes its output column
            #pragma unroll
            for (int k_local = 0; k_local < BLOCK_K; ++k_local) {
                int k = k_start + k_local;
                if (k < D) {
                    float x_hat_k = smem_x[k];
                    rotated_even += x_hat_k * smem_rot[k_local * HALF_D + tid];
                }
            }
            __syncthreads();

            // -- Load odd rotation tile [BLOCK_K, HALF_D] --
            #pragma unroll
            for (int k_local = 0; k_local < BLOCK_K; ++k_local) {
                int k = k_start + k_local;
                if (k < D) {
                    smem_rot[k_local * HALF_D + tid] =
                        rot_T_odd[k * HALF_D + tid];
                }
            }
            __syncthreads();

            #pragma unroll
            for (int k_local = 0; k_local < BLOCK_K; ++k_local) {
                int k = k_start + k_local;
                if (k < D) {
                    float x_hat_k = smem_x[k];
                    rotated_odd += x_hat_k * smem_rot[k_local * HALF_D + tid];
                }
            }
            __syncthreads();
        }

        // ---- Step 3: Bucketize (linear scan over sorted boundaries) ----
        int idx_even = 0;
        int idx_odd  = 0;

        #pragma unroll
        for (int b = 0; b < N_BOUNDARIES; ++b) {
            float boundary = smem_boundaries[b];
            idx_even += (rotated_even >= boundary) ? 1 : 0;
            idx_odd  += (rotated_odd  >= boundary) ? 1 : 0;
        }

        // ---- Step 4: Pack nibbles (high=even, low=odd) ----
        uint8_t packed_byte = static_cast<uint8_t>(
            ((idx_even & 0xF) << 4) | (idx_odd & 0xF));

        // ---- Step 5: Store ----
        packed[row * HALF_D + tid] = packed_byte;

        // One thread writes the norm
        if (tid == 0) {
            norms_out[row] = shared_norm;
        }

        __syncthreads();
    }
}

// ============================================================================
// Launcher
// ============================================================================

/// Dispatch to the correct template instantiation based on D and scalar type.
template <int HALF_D, typename scalar_t>
void launch_compress(
    const scalar_t* x,
    const float* rot_T_even,
    const float* rot_T_odd,
    const float* boundaries,
    uint8_t* packed,
    float* norms,
    int M,
    cudaStream_t stream)
{
    static constexpr int D = HALF_D * 2;
    const int grid = cdiv(M, ROWS_PER_BLOCK);
    const int block = HALF_D;

    // Shared memory: x_hat + rotation tile + boundaries + reduction scratch
    const int num_warps = cdiv(HALF_D, WARP_SIZE);
    const size_t smem_bytes =
        D * sizeof(float) +                      // smem_x (normalized input)
        BLOCK_K * HALF_D * sizeof(float) +        // rotation tile
        N_BOUNDARIES * sizeof(float) +            // boundaries
        num_warps * sizeof(float);                // reduction scratch

    tq4_compress_kernel<HALF_D, scalar_t><<<grid, block, smem_bytes, stream>>>(
        x, rot_T_even, rot_T_odd, boundaries, packed, norms, M);

    FLASHQUANT_CHECK_LAST_ERROR();
}

// ============================================================================
// C++ wrapper
// ============================================================================

void tq4_compress_cuda(
    const void* x,
    const float* rot_T_even,
    const float* rot_T_odd,
    const float* boundaries,
    uint8_t* packed,
    float* norms,
    int M,
    int D,
    bool is_bf16,
    cudaStream_t stream)
{
    const int half_d = D / 2;

    // Dispatch on HALF_D (compile-time specialization) and scalar type.
    // Common head dimensions: 64 (HALF_D=32), 128 (HALF_D=64), 256 (HALF_D=128).
    #define DISPATCH_HALF_D(HD, scalar_type)                                     \
        launch_compress<HD, scalar_type>(                                         \
            reinterpret_cast<const scalar_type*>(x),                             \
            rot_T_even, rot_T_odd, boundaries, packed, norms, M, stream)

    #define DISPATCH_DTYPE(HD)                                                   \
        if (is_bf16) {                                                           \
            DISPATCH_HALF_D(HD, __nv_bfloat16);                                  \
        } else {                                                                 \
            DISPATCH_HALF_D(HD, __half);                                          \
        }

    switch (half_d) {
        case 32:  DISPATCH_DTYPE(32);  break;
        case 64:  DISPATCH_DTYPE(64);  break;
        case 128: DISPATCH_DTYPE(128); break;
        default:
            fprintf(stderr,
                    "tq4_compress_cuda: unsupported HALF_D=%d (D=%d). "
                    "Supported: 32, 64, 128.\n", half_d, D);
            abort();
    }

    #undef DISPATCH_DTYPE
    #undef DISPATCH_HALF_D
}

}  // namespace flashquant
