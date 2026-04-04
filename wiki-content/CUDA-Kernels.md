# CUDA Kernels

This page provides a detailed walkthrough of every CUDA kernel in FlashQuant, covering the algorithm, memory layout, performance optimizations, and design rationale for each.

FlashQuant contains **6 CUDA kernel files** totaling approximately 2,575 lines of code, plus a shared header `utils.cuh` with common primitives.

---

## Table of Contents

1. [utils.cuh -- Shared Primitives](#utilscuh----shared-primitives)
2. [compress.cu -- Fused TQ4 Compression](#compresscu----fused-tq4-compression)
3. [decompress.cu -- Coalesced TQ4 Decompression](#decompresscu----coalesced-tq4-decompression)
4. [flash_attention.cu -- Standard FlashAttention-2](#flash_attentioncu----standard-flashattention-2)
5. [fused_tq_attention.cu -- FA2 + Inline TQ4](#fused_tq_attentioncu----fa2--inline-tq4)
6. [paged_decode.cu -- Split-K FlashDecoding](#paged_decodecu----split-k-flashdecoding)
7. [split_k_reduce.cu -- Partial Softmax Reduction](#split_k_reducecu----partial-softmax-reduction)
8. [Performance Optimization Summary](#performance-optimization-summary)

---

## utils.cuh -- Shared Primitives

All kernels include `utils.cuh`, which provides device-only, header-only helpers designed for maximum inlining.

### Warp-Level Reductions

Butterfly shuffle pattern for sum and max across a warp (32 threads):

```cpp
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}
```

This compiles to 5 `SHFL.BFLY` instructions -- the fastest possible reduction on NVIDIA GPUs. No shared memory is needed.

### Block-Level Reductions

Two-phase reduction: intra-warp via shuffles, then inter-warp via shared memory:

```cpp
__device__ __forceinline__ float block_reduce_sum(float val, float* smem) {
    val = warp_reduce_sum(val);              // Phase 1: intra-warp
    if (lane == 0) smem[warp_id] = val;      // Write warp results
    __syncthreads();
    if (warp_id == 0) {                      // Phase 2: first warp reduces
        val = (lane < num_warps) ? smem[lane] : 0.0f;
        val = warp_reduce_sum(val);
    }
    __syncthreads();
    return val;
}
```

### Coalesced Load/Store Helpers

Templated helpers that use `float4` (128-bit) or `uint32` (32-bit) vectorized accesses:

```cpp
template <int N>
__device__ __forceinline__ void coalesced_load_f32(float* dst, const float* src);

template <int N>
__device__ __forceinline__ void coalesced_load_u8(uint8_t* dst, const uint8_t* src);
```

### The exp2/log2(e) Trick

All softmax computations use `exp2f()` instead of `expf()`:

```cpp
static constexpr float LOG2E = 1.4426950408889634f;

// Instead of:  exp(x * sm_scale)
// We compute:  exp2(x * sm_scale * log2(e))
float qk_scale = sm_scale * LOG2E;
float p = exp2f(score - m_new);
```

`exp2f()` compiles to a single PTX instruction (`ex2.approx.ftz.f32`), while `expf()` requires a multi-instruction sequence. This saves ~3 clock cycles per softmax computation.

### Type Conversion: uint32_as_float

For loading fp32 norms that are stored as raw bytes in the paged KV cache:

```cpp
__device__ __forceinline__ float uint32_as_float(uint32_t bits) {
    float result;
    memcpy(&result, &bits, sizeof(float));
    return result;
}
```

This replaces the original 4-separate-byte-load pattern with a single 4-byte load.

---

## compress.cu -- Fused TQ4 Compression

**Purpose:** Fuse the entire TurboQuant compression pipeline into a single kernel launch, replacing 6 separate PyTorch operations.

### What It Replaces

```python
# Original 6-kernel path:
norms = torch.norm(x, dim=-1, keepdim=True)           # Kernel 1: norm
x_hat = x / (norms + 1e-10)                            # Kernel 2: divide
rotated = x_hat @ rotation.T                            # Kernel 3: matmul
indices = torch.bucketize(rotated, boundaries)          # Kernel 4: bucketize
indices = indices.clamp(0, 15)                          # Kernel 5: clamp
packed = (indices[:, 0::2] << 4) | indices[:, 1::2]    # Kernel 6: pack
```

### Kernel Signature

```cpp
template <int HALF_D, typename scalar_t>
__global__ void tq4_compress_kernel(
    const scalar_t* __restrict__ x,          // (M, D=2*HALF_D)
    const float* __restrict__ rot_T_even,    // (D, HALF_D)
    const float* __restrict__ rot_T_odd,     // (D, HALF_D)
    const float* __restrict__ boundaries,    // (15,) for TQ4
    uint8_t* __restrict__ packed,            // (M, HALF_D)
    float* __restrict__ norms_out,           // (M,)
    const int M);
```

### Grid and Block

```
Grid:  (ceil(M / ROWS_PER_BLOCK),)     1D grid over rows
Block: (HALF_D,)                        One thread per output column
```

For D=128: HALF_D=64 threads per block, processing 4 rows per CTA.

### Shared Memory Layout

```
smem_x[D]                            128 floats = 512 bytes   (input row)
smem_rot[BLOCK_K * HALF_D]           32*64 = 8,192 bytes      (rotation tile)
smem_boundaries[15]                   60 bytes                  (boundaries)
smem_reduce[num_warps]                8 bytes                   (reduction)
Total: ~8.8 KB per CTA
```

### Algorithm (Per Row)

**Step 1: Load and Compute Norm**
```
Each thread loads D/HALF_D = 2 elements into smem_x
Compute partial_sq = sum of squares
block_reduce_sum -> total norm^2
norm = sqrt(sum_sq)
```

**Step 2: Normalize In-Place**
```
Each thread: smem_x[i] *= 1/norm  (for its 2 elements)
```

**Step 3: Tiled Rotation**
```
For each tile of BLOCK_K=32 columns of the rotation matrix:
  - Cooperatively load rotation tile [BLOCK_K, HALF_D] into smem_rot
  - Each thread accumulates: rotated[tid] += sum_k smem_x[k] * smem_rot[k, tid]
```

This is the most computationally intensive step. The tiled approach ensures the rotation matrix (D*HALF_D * 4 bytes = 32 KB for D=128) is loaded once into shared memory and reused across all HALF_D threads.

**Step 4: Bucketize**
```
Linear scan over 15 sorted boundaries:
for each boundary b:
    idx += (rotated >= b) ? 1 : 0
```

This compiles to 15 predicated additions -- no branch divergence.

**Step 5: Pack Nibbles**
```
packed_byte = (idx_even << 4) | (idx_odd & 0xF)
```

Thread `tid` writes one byte containing the even and odd nibble indices.

### Performance Characteristics

| Metric | Value (D=128, M=32) |
|--------|---------------------|
| CTAs launched | 8 (32/4 rows per block) |
| Threads per CTA | 64 |
| Shared memory | 8.8 KB |
| Registers | ~40 per thread |
| Arithmetic intensity | High (rotation matmul dominates) |

---

## decompress.cu -- Coalesced TQ4 Decompression

**Purpose:** Unpack nibble-packed indices, look up centroids, and scale by norms. Optimized for coalesced memory writes.

### Kernel Signature

```cpp
template <int HALF_D>
__global__ void tq4_decompress_kernel_fp16(
    const uint8_t* __restrict__ packed,      // (M, HALF_D)
    const float* __restrict__ norms,         // (M,)
    const float* __restrict__ centroids,     // (16,)
    __half* __restrict__ out,                // (M, D=2*HALF_D)
    const int M);
```

### Key Optimization: Sequential Layout

The output writes use **sequential layout**, not interleaved:

```
Output row layout:
  [c_hi[0], c_hi[1], ..., c_hi[63], c_lo[0], c_lo[1], ..., c_lo[63]]
   \___________ HALF_D _____________/ \____________ HALF_D ____________/

Thread tid writes:
  out[tid]            = c_hi * norm    (coalesced: adjacent threads -> adjacent addresses)
  out[HALF_D + tid]   = c_lo * norm    (coalesced: second block, same stride)
```

With interleaved layout (as in the original Triton version):
```
  out[2*tid]     = c_hi * norm    (stride-2: threads 0,1,2... write to 0,2,4...)
  out[2*tid + 1] = c_lo * norm    (stride-2: threads 0,1,2... write to 1,3,5...)
```

Stride-2 writes require **2x memory transactions** because adjacent threads access non-adjacent cache lines.

### Shared Memory: Centroids

```cpp
__shared__ float smem_centroids[16];  // 64 bytes = 1 cache line
```

The entire 16-entry centroid table fits in a single L1 cache line. After one cooperative load, all subsequent centroid lookups are L1 hits with zero bank conflicts.

### Algorithm (Per Row)

```
1. Load one packed byte:       p = packed[row * HALF_D + tid]
2. Extract nibbles:            hi_idx = (p >> 4) & 0xF;  lo_idx = p & 0xF
3. Centroid gather:            c_hi = smem_centroids[hi_idx];  c_lo = smem_centroids[lo_idx]
4. Scale by norm:              c_hi *= norm;  c_lo *= norm
5. Write output (coalesced):   out[tid] = c_hi;  out[HALF_D + tid] = c_lo
```

### Separate FP16/BF16 Specializations

Due to CUDA's lack of `static_cast<__half>()` from float, separate kernel specializations use the correct intrinsics:

```cpp
// FP16 path
out_row[tid] = __float2half(c_hi);

// BF16 path
out_row[tid] = __float2bfloat16(c_hi);
```

---

## flash_attention.cu -- Standard FlashAttention-2

**Purpose:** Standard FlashAttention-2 for uncompressed KV caches. Used as the attention backend when KV cache is already decompressed.

### Two Paths

**Prefill path** (BLOCK_M=64, BLOCK_N=64):
```
Grid: (ceil(N_Q / 64), B * H_Q)
Block: (64,)
```
Full tile matmul via shared memory. Supports causal masking with tile-skipping.

**Decode path** (BLOCK_M=1, specialized):
```
Grid: (B, H_Q)
Block: (128,)
```
Single query vector stays in registers. K/V stream through shared memory. No causal masking needed.

### Online Softmax (Milakov & Gimelshein, 2018)

The key to FlashAttention's memory efficiency is computing softmax in a single pass without materializing the full attention matrix:

```
Initialize: m_i = -inf, l_i = 1, acc = 0

For each KV tile:
    S = Q @ K^T * scale          // Tile of attention scores
    m_new = max(m_i, rowmax(S))  // Update running maximum
    P = exp2(S - m_new)          // Exponentiate (relative to new max)
    acc = acc * exp2(m_i - m_new) + P @ V   // Rescale old accumulator + add new
    l_i = l_i * exp2(m_i - m_new) + rowsum(P)   // Update denominator
    m_i = m_new

Output: acc / l_i
```

### FP32 Accumulators

All intermediate computations (m_i, l_i, acc, score computation) use FP32 accumulators even when inputs are FP16/BF16. This prevents numerical instability that occurs when softmax denominators become very large or very small.

### GQA (Grouped Query Attention) Support

```cpp
const int off_h_kv = off_h_q / (H_Q / H_KV);  // GQA mapping
```

When H_Q > H_KV, multiple query heads share the same KV head. The kernel maps query head indices to KV head indices via integer division.

### Causal Masking with Tile-Skipping

For causal attention, the kernel skips tiles that are entirely above the diagonal:

```cpp
const int hi = IS_CAUSAL ? min((start_m + 1) * BLOCK_M, N_KV) : N_KV;
for (int start_n = 0; start_n < hi; start_n += BLOCK_N) {
    // Only process tiles where some positions are not masked
}
```

This eliminates approximately half the computation for causal attention.

---

## fused_tq_attention.cu -- FA2 + Inline TQ4

**Purpose:** The core fusion kernel that decompresses TQ4-packed K and V **inside** the FlashAttention-2 tile loop, avoiding any intermediate HBM writes.

### Input/Output

```
Inputs:
  Q_rot:         (B, H_Q, N_Q, D)     Pre-rotated queries (fp16/bf16)
  K_packed:      (B, H_KV, N_KV, D/2) Nibble-packed K indices (uint8)
  K_norms:       (B, H_KV, N_KV)      K vector norms (fp32)
  V_packed:      (B, H_KV, N_KV, D/2) Nibble-packed V indices (uint8)
  V_norms:       (B, H_KV, N_KV)      V vector norms (fp32)
  K_centroids:   (16,)                 K centroid table (fp32)
  V_centroids:   (16,)                 V centroid table (fp32)

Output:
  Out:           (B, H_Q, N_Q, D)      Attention output in rotated space
```

Note: Q is **pre-rotated** by Pi^T outside the kernel. The output is in rotated space; the caller applies post-rotation Pi.

### Inner Loop (Prefill)

```
For each KV tile [BLOCK_N, D]:

    // --- Decompress K tile ---
    For each KV position n (1 thread per position):
        k_norm = K_norms[n]
        For each packed byte j:
            hi_idx = (packed[j] >> 4) & 0xF
            lo_idx =  packed[j]       & 0xF
            kv_tile[n, j]          = K_centroids[hi_idx] * k_norm
            kv_tile[n, HALF_D + j] = K_centroids[lo_idx] * k_norm
    __syncthreads()

    // --- Q @ K^T ---
    S[m, n] = dot(q_tile[m, :], kv_tile[n, :]) * qk_scale
    Apply causal + OOB masking
    __syncthreads()

    // --- Decompress V tile (reuse kv_tile) ---
    (same pattern with V_centroids and V_norms)
    __syncthreads()

    // --- Online softmax + P @ V accumulation ---
    m_new = max(m_i, rowmax(S))
    acc *= exp2(m_i - m_new)
    P = exp2(S - m_new)
    acc += P @ kv_tile
    l_i = l_i * exp2(m_i - m_new) + rowsum(P)
    m_i = m_new
```

### Shared Memory Layout

```
k_centroids[16]          64 bytes     K codebook
v_centroids[16]          64 bytes     V codebook
q_tile[64 * 128]         32,768 bytes Q stays resident
kv_tile[64 * 128]        32,768 bytes Reused for K then V
s_tile[64 * 64]          16,384 bytes Attention scores
m_i[64]                  256 bytes    Running max
l_i[64]                  256 bytes    Running denominator
acc[64 * 128]            32,768 bytes Output accumulator
Total: ~115 KB
```

### Decode Path

The decode kernel (BLOCK_M=1) uses a similar pattern but keeps Q in **registers** instead of shared memory, and each thread processes a subset of the HEAD_DIM dimension:

```cpp
float q_reg[HEAD_DIM];  // Q stays in registers for the entire kernel
```

---

## paged_decode.cu -- Split-K FlashDecoding

**Purpose:** The critical production kernel for vLLM integration. Combines Split-K parallelism with TQ4 decompression from a paged KV cache with block tables.

### Grid

```
Grid: (num_seqs, H_Q, NUM_SPLITS=4)
Block: (128,)
```

Each CTA processes 1/4 of the KV sequence for one (sequence, query_head) pair.

### Page Table Addressing

The paged cache uses **bit-shift addressing** instead of division/modulo:

```cpp
// Template parameter BLOCK_SIZE_LOG2 enables bit-shift operations
template <int HEAD_DIM, int BLOCK_N, int BLOCK_SIZE_LOG2, typename scalar_t>

// Inside kernel:
int logical_block = kv_pos >> BLOCK_SIZE_LOG2;        // kv_pos / block_size
int within_block  = kv_pos & (BLOCK_SIZE - 1);        // kv_pos % block_size
int physical_block = block_table[logical_block];
```

Supported block sizes: 8 (log2=3), 16 (log2=4), 32 (log2=5).

### Fused Norm Load

The original Triton version loaded fp32 norms as 4 separate byte loads:

```python
# Original (4 loads, manual reconstruction):
b0, b1, b2, b3 = load_bytes(addr, 4)
norm = reconstruct_float(b0, b1, b2, b3)
```

FlashQuant uses a single 4-byte load with reinterpret:

```cpp
uint32_t k_norm_bits = *reinterpret_cast<const uint32_t*>(k_norm_ptr);
float k_norm = uint32_as_float(k_norm_bits);
```

This reduces load instructions by 4x for norm access.

### KV Cache Byte Layout

```
Per token, per KV head group:
Offset 0:                K indices (HALF_D bytes, nibble-packed)
Offset HALF_D:           K norm (4 bytes, fp32)
Offset HALF_D + 4:       V indices (HALF_D bytes, nibble-packed)
Offset HALF_D + 4 + HALF_D: V norm (4 bytes, fp32)
Total: D + 8 bytes per token per KV head
```

### Partial Output

Each CTA writes its partial softmax state to global memory:

```cpp
partial_acc[part_idx * HEAD_DIM + d] = acc_reg[d];  // Per-dimension accumulator
partial_m[part_idx] = m_i;                           // Running max
partial_l[part_idx] = l_i;                           // Running denominator
```

These are combined by `split_k_reduce.cu`.

### Early Exit for Empty Splits

When a split has no KV tokens to process (sequence shorter than split's range):

```cpp
if (kv_start >= seq_len) {
    partial_m[part_idx] = -FLT_MAX;  // Sentinel: this split contributes nothing
    partial_l[part_idx] = 1.0f;
    for (int d = ...) partial_acc[...] = 0.0f;
    return;
}
```

The sentinel values `m = -FLT_MAX` ensure this split's contribution is zero after the log-sum-exp reduction.

---

## split_k_reduce.cu -- Partial Softmax Reduction

**Purpose:** Combine NUM_SPLITS=4 partial softmax states into one final attention output per (sequence, head) pair.

### Grid

```
Grid: (num_seqs * H_Q,)
Block: (min(HEAD_DIM, 256),)
```

One CTA per (sequence, head) pair. Simple and lightweight.

### Algorithm

```
// Step 1: Find global max across all splits
m_final = max(m_0, m_1, m_2, m_3)

// Step 2: Compute rescaled weights
for each split s:
    weights[s] = l_s * exp2(m_s - m_final)
l_final = sum(weights)

// Step 3: Weighted combination of partial accumulators
for each dimension d:
    combined[d] = 0
    for each split s:
        combined[d] += partial_acc[s][d] * weights[s]
    out[d] = combined[d] / l_final
```

### Numerical Stability

The key insight is that `m_s - m_final <= 0` for all splits, so `exp2(m_s - m_final) <= 1`. This prevents overflow in the exponential. Underflow to zero is handled by the `inv_l_final` guard:

```cpp
float inv_l_final = (l_final > 0.0f) ? (1.0f / l_final) : 0.0f;
```

### Performance

This kernel is extremely lightweight -- it performs only `4 * HEAD_DIM` floating-point multiplications plus 4 `exp2f` calls per thread. The total execution time is typically < 2 microseconds, negligible compared to the main decode kernel.

---

## Performance Optimization Summary

| Optimization | Kernel(s) | Impact |
|-------------|-----------|--------|
| **Multi-row blocks** (ROWS_PER_BLOCK=4) | compress, decompress | 2-4x SM utilization during decode |
| **Split-K** (NUM_SPLITS=4) | paged_decode + split_k_reduce | 4x SM utilization, 100% on A100 |
| **Coalesced writes** (sequential layout) | decompress | 2x memory bandwidth utilization |
| **Centroids in shared memory** | decompress, fused_tq, paged_decode | L1 hit rate ~100% for centroid lookups |
| **exp2/log2(e) prescaling** | all attention kernels | 1 PTX instruction vs. multi-instruction exp() |
| **Fused uint32 norm load** | paged_decode | 4x fewer load instructions for norms |
| **Bit-shift page addressing** | paged_decode | 0-cost division/modulo for page tables |
| **FP32 accumulators** | all attention kernels | Prevents softmax numerical instability |
| **Template on HALF_D** | compress, decompress | Full loop unrolling, optimal register allocation |
| **Warp shuffles** (no smem) | all kernels | Fastest possible intra-warp reductions |
| **Rotation tile in smem** | compress | Amortizes rotation matrix loads across threads |
| **Inline decompression** | fused_tq_attention | Eliminates HBM round-trip for decompressed KV |
| **Causal tile-skipping** | flash_attention, fused_tq | ~50% compute savings for causal attention |

---

*Copyright 2026 Ayi NEDJIMI. Apache License 2.0.*
