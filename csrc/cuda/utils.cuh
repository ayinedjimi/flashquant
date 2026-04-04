// Copyright 2026 FlashQuant Authors. Apache-2.0 License.
//
// Common CUDA helpers for the FlashQuant kernel library.
//
// Provides:
//   - Warp-level reductions (sum, max) via __shfl_xor_sync
//   - Block-level reductions via shared memory
//   - Coalesced load/store helpers
//   - constexpr power-of-2 utilities
//   - CUDA error checking macro
//
// All functions are device-only and header-only for inlining into kernels.

#pragma once

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cfloat>

namespace flashquant {

// ============================================================================
// Error checking
// ============================================================================

#define FLASHQUANT_CUDA_CHECK(expr)                                            \
    do {                                                                        \
        cudaError_t __err = (expr);                                            \
        if (__err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(__err));                                 \
            abort();                                                           \
        }                                                                      \
    } while (0)

#define FLASHQUANT_CHECK_LAST_ERROR() FLASHQUANT_CUDA_CHECK(cudaGetLastError())

// ============================================================================
// Compile-time utilities
// ============================================================================

/// True if N is a power of 2 and N > 0.
template <int N>
struct IsPowerOf2 {
    static constexpr bool value = (N > 0) && ((N & (N - 1)) == 0);
};

/// Ceiling division: cdiv(a, b) = (a + b - 1) / b.
__host__ __device__ __forceinline__ constexpr int cdiv(int a, int b) {
    return (a + b - 1) / b;
}

/// Log2 of a power-of-2 constant (compile-time).
template <int N>
struct Log2 {
    static_assert(IsPowerOf2<N>::value, "N must be a power of 2");
    static constexpr int value = Log2<N / 2>::value + 1;
};

template <>
struct Log2<1> {
    static constexpr int value = 0;
};

// ============================================================================
// Constants
// ============================================================================

static constexpr int WARP_SIZE = 32;

/// log2(e) for the exp2 pre-scaling trick: exp(x) = exp2(x * log2(e)).
static constexpr float LOG2E = 1.4426950408889634f;

// ============================================================================
// Warp-level reductions
// ============================================================================

/// Warp-level sum reduction over all 32 lanes using butterfly shuffle.
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

/// Warp-level max reduction over all 32 lanes using butterfly shuffle.
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

/// Warp-level sum reduction over a partial warp (WIDTH lanes).
template <int WIDTH>
__device__ __forceinline__ float warp_reduce_sum_partial(float val) {
    static_assert(IsPowerOf2<WIDTH>::value && WIDTH <= WARP_SIZE,
                  "WIDTH must be power-of-2 and <= 32");
    #pragma unroll
    for (int offset = WIDTH / 2; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

/// Warp-level max reduction over a partial warp (WIDTH lanes).
template <int WIDTH>
__device__ __forceinline__ float warp_reduce_max_partial(float val) {
    static_assert(IsPowerOf2<WIDTH>::value && WIDTH <= WARP_SIZE,
                  "WIDTH must be power-of-2 and <= 32");
    #pragma unroll
    for (int offset = WIDTH / 2; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

// ============================================================================
// Block-level reductions
// ============================================================================

/// Block-level sum reduction. Requires shared memory of size
/// (blockDim.x / WARP_SIZE) floats.
/// Returns the reduction result in lane 0 of warp 0.
__device__ __forceinline__ float block_reduce_sum(float val, float* smem) {
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int num_warps = blockDim.x / WARP_SIZE;

    // Intra-warp reduce
    val = warp_reduce_sum(val);

    // Write warp results to shared memory
    if (lane == 0) {
        smem[warp_id] = val;
    }
    __syncthreads();

    // First warp reduces across warps
    if (warp_id == 0) {
        val = (lane < num_warps) ? smem[lane] : 0.0f;
        val = warp_reduce_sum(val);
    }
    __syncthreads();

    return val;
}

/// Block-level max reduction. Same shared memory requirements as sum.
__device__ __forceinline__ float block_reduce_max(float val, float* smem) {
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int num_warps = blockDim.x / WARP_SIZE;

    // Intra-warp reduce
    val = warp_reduce_max(val);

    if (lane == 0) {
        smem[warp_id] = val;
    }
    __syncthreads();

    if (warp_id == 0) {
        val = (lane < num_warps) ? smem[lane] : -FLT_MAX;
        val = warp_reduce_max(val);
    }
    __syncthreads();

    return val;
}

// ============================================================================
// Coalesced memory access helpers
// ============================================================================

/// Load a contiguous vector of N floats from global memory.
/// Pointer must be aligned to 16 bytes for float4 path.
template <int N>
__device__ __forceinline__ void coalesced_load_f32(
    float* __restrict__ dst,
    const float* __restrict__ src) {
    static_assert(N % 4 == 0, "N must be divisible by 4 for float4 loads");
    const float4* src4 = reinterpret_cast<const float4*>(src);
    float4* dst4 = reinterpret_cast<float4*>(dst);
    #pragma unroll
    for (int i = 0; i < N / 4; ++i) {
        dst4[i] = src4[i];
    }
}

/// Store a contiguous vector of N floats to global memory.
template <int N>
__device__ __forceinline__ void coalesced_store_f32(
    float* __restrict__ dst,
    const float* __restrict__ src) {
    static_assert(N % 4 == 0, "N must be divisible by 4 for float4 stores");
    float4* dst4 = reinterpret_cast<float4*>(dst);
    const float4* src4 = reinterpret_cast<const float4*>(src);
    #pragma unroll
    for (int i = 0; i < N / 4; ++i) {
        dst4[i] = src4[i];
    }
}

/// Load N half values from global memory into float registers.
template <int N>
__device__ __forceinline__ void coalesced_load_half_to_f32(
    float* __restrict__ dst,
    const __half* __restrict__ src) {
    static_assert(N % 2 == 0, "N must be even for half2 loads");
    const __half2* src2 = reinterpret_cast<const __half2*>(src);
    #pragma unroll
    for (int i = 0; i < N / 2; ++i) {
        __half2 h2 = src2[i];
        dst[2 * i]     = __half2float(h2.x);
        dst[2 * i + 1] = __half2float(h2.y);
    }
}

/// Store N float values to global memory as half.
template <int N>
__device__ __forceinline__ void coalesced_store_f32_to_half(
    __half* __restrict__ dst,
    const float* __restrict__ src) {
    static_assert(N % 2 == 0, "N must be even for half2 stores");
    __half2* dst2 = reinterpret_cast<__half2*>(dst);
    #pragma unroll
    for (int i = 0; i < N / 2; ++i) {
        dst2[i] = __halves2half2(__float2half(src[2 * i]),
                                  __float2half(src[2 * i + 1]));
    }
}

/// Load N uint8 values from global memory using uint32 coalesced reads.
template <int N>
__device__ __forceinline__ void coalesced_load_u8(
    uint8_t* __restrict__ dst,
    const uint8_t* __restrict__ src) {
    static_assert(N % 4 == 0, "N must be divisible by 4 for uint32 loads");
    const uint32_t* src4 = reinterpret_cast<const uint32_t*>(src);
    uint32_t* dst4 = reinterpret_cast<uint32_t*>(dst);
    #pragma unroll
    for (int i = 0; i < N / 4; ++i) {
        dst4[i] = src4[i];
    }
}

// ============================================================================
// Type conversion helpers
// ============================================================================

/// Reinterpret 4 bytes (as stored in uint32) as a float.
/// Used for loading fp32 norms packed as raw bytes in KV cache.
__device__ __forceinline__ float uint32_as_float(uint32_t bits) {
    float result;
    memcpy(&result, &bits, sizeof(float));
    return result;
}

/// Convert __half to float.
__device__ __forceinline__ float half_to_float(__half h) {
    return __half2float(h);
}

/// Convert float to __half.
__device__ __forceinline__ __half float_to_half(float f) {
    return __float2half(f);
}

/// Convert __nv_bfloat16 to float.
__device__ __forceinline__ float bf16_to_float(__nv_bfloat16 h) {
    return __bfloat162float(h);
}

/// Convert float to __nv_bfloat16.
__device__ __forceinline__ __nv_bfloat16 float_to_bf16(float f) {
    return __float2bfloat16(f);
}

}  // namespace flashquant
