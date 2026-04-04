// Copyright 2026 FlashQuant Authors. Apache-2.0 License.
//
// Fused TQ4 decompress kernel: unpack + gather + scale.
//
// CUDA native replacement for turboquant-vllm/triton/tq4_decompress.py.
//
// Key improvement: COALESCED writes. The Triton version interleaves
// even/odd outputs (stride-2 writes), causing 2x more memory transactions.
// This kernel writes sequentially:
//   out[row, 0..HALF_D-1]   = high nibble values * norm
//   out[row, HALF_D..D-1]   = low nibble values * norm
//
// The attention kernels must match this layout (they do -- see
// fused_tq_attention.cu and paged_decode.cu).

#pragma once

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace flashquant {

/// Launch the fused TQ4 decompress kernel.
///
/// Input:
///   packed    - (M, D/2) uint8 nibble-packed indices
///   norms     - (M,) float32 per-vector norms
///   centroids - (16,) float32 centroid table
///
/// Output:
///   out       - (M, D) float16 or bfloat16
///
/// Layout: first D/2 positions = high nibble values, next D/2 = low nibble.
/// This differs from the Triton version (which interleaved) but enables
/// coalesced writes.
///
/// is_bf16: true if output should be bfloat16, false for float16.
void tq4_decompress_cuda(
    const uint8_t* packed,
    const float* norms,
    const float* centroids,
    void* out,
    int M,
    int D,
    bool is_bf16,
    cudaStream_t stream = nullptr);

}  // namespace flashquant
