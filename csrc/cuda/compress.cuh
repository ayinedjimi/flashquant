// Copyright 2026 FlashQuant Authors. Apache-2.0 License.
//
// Fused TQ4 compress kernel: norm + rotate + quantize + nibble-pack.
//
// CUDA native replacement for turboquant-vllm/triton/tq4_compress.py
// with multi-row blocks, shared memory rotation tiles, single-pass norm
// computation, and compile-time loop unrolling.
//
// Key improvements over the Triton version:
//   1. ROWS_PER_BLOCK=4 for small M (fixes 97% SM idle on decode)
//   2. Shared memory for rotation matrix tile (reduces register pressure)
//   3. Single pass for norm computation (no re-read of x)
//   4. Template on HALF_D for full loop unrolling
//   5. Grid: ceil(M / ROWS_PER_BLOCK) -- better occupancy

#pragma once

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace flashquant {

/// Launch the fused TQ4 compress kernel.
///
/// Input:
///   x           - (M, D) float16 or bfloat16 input vectors
///   rot_T_even  - (D, D/2) float32 even columns of rotation.T
///   rot_T_odd   - (D, D/2) float32 odd columns of rotation.T
///   boundaries  - (15,) float32 quantization boundaries
///
/// Output:
///   packed      - (M, D/2) uint8 nibble-packed indices
///   norms       - (M,) float32 per-vector L2 norms
///
/// is_bf16: true if input is bfloat16, false for float16.
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
    cudaStream_t stream = nullptr);

}  // namespace flashquant
