// Copyright 2026 FlashQuant Authors. Apache-2.0 License.
//
// Standard FlashAttention-2 forward kernel (CUDA native).
//
// Implements the online softmax algorithm from Dao (2023) with:
//   - FP32 accumulators (m_i, l_i, acc) for numerical stability
//   - exp2/log2(e) pre-scaling trick (single PTX instruction)
//   - Causal masking with tile-skipping optimization
//   - GQA support (arbitrary Q/KV head ratio)
//   - Template on HEAD_DIM, BLOCK_M, BLOCK_N for compile-time specialization
//   - Separate prefill (BLOCK_M=64/128) and decode (BLOCK_M=1) paths

#pragma once

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace flashquant {

/// Launch FlashAttention-2 forward pass.
///
/// Q:   (B, H_Q, N_Q, D) float16/bfloat16
/// K:   (B, H_KV, N_KV, D) float16/bfloat16
/// V:   (B, H_KV, N_KV, D) float16/bfloat16
/// Out: (B, H_Q, N_Q, D) float16/bfloat16
///
/// All tensors must be contiguous in the last dimension (head_dim).
/// sm_scale: softmax scale factor (typically 1/sqrt(D)).
/// is_causal: apply causal masking.
/// is_bf16: true for bfloat16, false for float16.
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
    cudaStream_t stream = nullptr);

}  // namespace flashquant
