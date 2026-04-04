// Copyright 2026 FlashQuant Authors. Apache-2.0 License.
//
// Fused TQ4 decompress kernel: unpack + gather + scale.
//
// Performance fixes over the Triton version:
//   1. COALESCED writes: sequential layout (hi block, lo block) instead of
//      interleaved (hi, lo, hi, lo, ...). Eliminates stride-2 writes that
//      double memory transactions.
//   2. Multi-row blocks: ROWS_PER_BLOCK=4 for small M (decode).
//   3. Centroids in shared memory (16 floats = 64 bytes, fits in 1 cache line).
//   4. Template on HALF_D for loop unrolling.

#include "decompress.cuh"
#include "utils.cuh"

namespace flashquant {

// ============================================================================
// Configuration
// ============================================================================

static constexpr int DECOMPRESS_ROWS_PER_BLOCK = 4;
static constexpr int NUM_CENTROIDS = 16;  // TQ4: 4-bit -> 16 levels

// ============================================================================
// Kernel
// ============================================================================

/// Fused TQ4 decompress kernel.
///
/// Grid:  (ceil(M / ROWS_PER_BLOCK),)
/// Block: (HALF_D,) -- one thread per packed byte
///
/// Each thread:
///   1. Loads one packed uint8 byte
///   2. Extracts high and low nibble indices
///   3. Gathers centroids from shared memory
///   4. Multiplies by norm
///   5. Writes two output values (coalesced, sequential layout)
///
/// Output layout per row:
///   [c_hi[0], c_hi[1], ..., c_hi[HALF_D-1], c_lo[0], c_lo[1], ..., c_lo[HALF_D-1]]
template <int HALF_D, typename scalar_t>
__global__ void tq4_decompress_kernel(
    const uint8_t* __restrict__ packed,      // (M, HALF_D)
    const float* __restrict__ norms,         // (M,)
    const float* __restrict__ centroids,     // (NUM_CENTROIDS,)
    scalar_t* __restrict__ out,              // (M, D=2*HALF_D)
    const int M)
{
    static constexpr int D = HALF_D * 2;

    const int block_row_start = blockIdx.x * DECOMPRESS_ROWS_PER_BLOCK;
    const int tid = threadIdx.x;  // [0, HALF_D)

    // Load centroids into shared memory (one cache line, 64 bytes)
    __shared__ float smem_centroids[NUM_CENTROIDS];
    if (tid < NUM_CENTROIDS) {
        smem_centroids[tid] = centroids[tid];
    }
    __syncthreads();

    // Process ROWS_PER_BLOCK rows per CTA
    #pragma unroll
    for (int local_row = 0; local_row < DECOMPRESS_ROWS_PER_BLOCK; ++local_row) {
        const int row = block_row_start + local_row;
        if (row >= M) return;

        // Load packed byte
        uint8_t p = packed[row * HALF_D + tid];

        // Nibble unpack
        int hi_idx = (p >> 4) & 0xF;
        int lo_idx =  p       & 0xF;

        // Centroid gather from shared memory
        float c_hi = smem_centroids[hi_idx];
        float c_lo = smem_centroids[lo_idx];

        // Load norm and scale
        float norm = norms[row];
        c_hi *= norm;
        c_lo *= norm;

        // COALESCED writes: sequential layout
        // First HALF_D positions: high nibble values
        // Next HALF_D positions: low nibble values
        scalar_t* out_row = out + row * D;
        out_row[tid]          = static_cast<scalar_t>(c_hi);
        out_row[HALF_D + tid] = static_cast<scalar_t>(c_lo);
    }
}

// ============================================================================
// Specializations for __half and __nv_bfloat16 static_cast
// ============================================================================

// __half version: use __float2half
template <int HALF_D>
__global__ void tq4_decompress_kernel_fp16(
    const uint8_t* __restrict__ packed,
    const float* __restrict__ norms,
    const float* __restrict__ centroids,
    __half* __restrict__ out,
    const int M)
{
    static constexpr int D = HALF_D * 2;

    const int block_row_start = blockIdx.x * DECOMPRESS_ROWS_PER_BLOCK;
    const int tid = threadIdx.x;

    __shared__ float smem_centroids[NUM_CENTROIDS];
    if (tid < NUM_CENTROIDS) {
        smem_centroids[tid] = centroids[tid];
    }
    __syncthreads();

    #pragma unroll
    for (int local_row = 0; local_row < DECOMPRESS_ROWS_PER_BLOCK; ++local_row) {
        const int row = block_row_start + local_row;
        if (row >= M) return;

        uint8_t p = packed[row * HALF_D + tid];

        int hi_idx = (p >> 4) & 0xF;
        int lo_idx =  p       & 0xF;

        float c_hi = smem_centroids[hi_idx];
        float c_lo = smem_centroids[lo_idx];

        float norm = norms[row];
        c_hi *= norm;
        c_lo *= norm;

        __half* out_row = out + row * D;
        out_row[tid]          = __float2half(c_hi);
        out_row[HALF_D + tid] = __float2half(c_lo);
    }
}

// __nv_bfloat16 version: use __float2bfloat16
template <int HALF_D>
__global__ void tq4_decompress_kernel_bf16(
    const uint8_t* __restrict__ packed,
    const float* __restrict__ norms,
    const float* __restrict__ centroids,
    __nv_bfloat16* __restrict__ out,
    const int M)
{
    static constexpr int D = HALF_D * 2;

    const int block_row_start = blockIdx.x * DECOMPRESS_ROWS_PER_BLOCK;
    const int tid = threadIdx.x;

    __shared__ float smem_centroids[NUM_CENTROIDS];
    if (tid < NUM_CENTROIDS) {
        smem_centroids[tid] = centroids[tid];
    }
    __syncthreads();

    #pragma unroll
    for (int local_row = 0; local_row < DECOMPRESS_ROWS_PER_BLOCK; ++local_row) {
        const int row = block_row_start + local_row;
        if (row >= M) return;

        uint8_t p = packed[row * HALF_D + tid];

        int hi_idx = (p >> 4) & 0xF;
        int lo_idx =  p       & 0xF;

        float c_hi = smem_centroids[hi_idx];
        float c_lo = smem_centroids[lo_idx];

        float norm = norms[row];
        c_hi *= norm;
        c_lo *= norm;

        __nv_bfloat16* out_row = out + row * D;
        out_row[tid]          = __float2bfloat16(c_hi);
        out_row[HALF_D + tid] = __float2bfloat16(c_lo);
    }
}

// ============================================================================
// Launcher
// ============================================================================

template <int HALF_D>
void launch_decompress(
    const uint8_t* packed,
    const float* norms,
    const float* centroids,
    void* out,
    int M,
    bool is_bf16,
    cudaStream_t stream)
{
    const int grid = cdiv(M, DECOMPRESS_ROWS_PER_BLOCK);
    const int block = HALF_D;

    if (is_bf16) {
        tq4_decompress_kernel_bf16<HALF_D><<<grid, block, 0, stream>>>(
            packed, norms, centroids,
            reinterpret_cast<__nv_bfloat16*>(out), M);
    } else {
        tq4_decompress_kernel_fp16<HALF_D><<<grid, block, 0, stream>>>(
            packed, norms, centroids,
            reinterpret_cast<__half*>(out), M);
    }

    FLASHQUANT_CHECK_LAST_ERROR();
}

// ============================================================================
// C++ wrapper
// ============================================================================

void tq4_decompress_cuda(
    const uint8_t* packed,
    const float* norms,
    const float* centroids,
    void* out,
    int M,
    int D,
    bool is_bf16,
    cudaStream_t stream)
{
    const int half_d = D / 2;

    switch (half_d) {
        case 32:
            launch_decompress<32>(packed, norms, centroids, out, M, is_bf16, stream);
            break;
        case 64:
            launch_decompress<64>(packed, norms, centroids, out, M, is_bf16, stream);
            break;
        case 128:
            launch_decompress<128>(packed, norms, centroids, out, M, is_bf16, stream);
            break;
        default:
            fprintf(stderr,
                    "tq4_decompress_cuda: unsupported HALF_D=%d (D=%d). "
                    "Supported: 32, 64, 128.\n", half_d, D);
            abort();
    }
}

}  // namespace flashquant
