# Architecture

This page describes the system architecture of FlashQuant, from the high-level Python API down to the CUDA kernel execution on GPU hardware.

---

## Table of Contents

1. [Layer Diagram](#layer-diagram)
2. [C++ Core Design Decisions](#c-core-design-decisions)
3. [CUDA Kernel Design](#cuda-kernel-design)
4. [Split-K FlashDecoding](#split-k-flashdecoding)
5. [Fused TQ4 Attention](#fused-tq4-attention)
6. [pybind11 Binding Strategy](#pybind11-binding-strategy)
7. [CPU Reference Fallback Chain](#cpu-reference-fallback-chain)
8. [vLLM Backend Architecture](#vllm-backend-architecture)

---

## Layer Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Python API Layer                             │
│                                                                     │
│   FlashQuantConfig    TurboQuantMSE    TurboQuantProd               │
│   CompressedDynamicCache              KeyCompressor / ValueCompressor│
│   register_flashquant_backend()                                     │
├─────────────────────────────────────────────────────────────────────┤
│                        Dispatch Layer                               │
│                                                                     │
│   kernels/compress.py    kernels/attention.py    kernels/decompress.│
│                                                                     │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐ │
│   │ try: _C.xxx  │ -> │ try: CUDA    │ -> │ CPU reference        │ │
│   │   (C++ ext)  │    │   kernel     │    │   (pure PyTorch)     │ │
│   └──────────────┘    └──────────────┘    └──────────────────────┘ │
├─────────────────────────────────────────────────────────────────────┤
│                     C++ Core (flashquant_core)                      │
│                                                                     │
│   codebook.cpp        rotation.cpp        quantizer.cpp             │
│   (Lloyd-Max)         (Haar QR)           (MSE + Prod)              │
│   packing.cpp         types.h                                       │
│   (nibble pack)       (LloydMaxCodebook, QuantizedMSE, etc.)       │
├─────────────────────────────────────────────────────────────────────┤
│                     CUDA Kernels (flashquant_cuda)                  │
│                                                                     │
│   compress.cu         decompress.cu       flash_attention.cu        │
│   fused_tq_attention.cu                   paged_decode.cu           │
│   split_k_reduce.cu                       utils.cuh                 │
├─────────────────────────────────────────────────────────────────────┤
│                          Hardware                                   │
│                                                                     │
│   NVIDIA Ampere+ GPU (SM 80+)                                      │
│   Shared Memory (up to 164 KB/SM)     L2 Cache     HBM2e/HBM3     │
│   Tensor Cores (FP16/BF16)            Warp Shuffles               │
└─────────────────────────────────────────────────────────────────────┘
```

### Data Flow: Compression

```
Input x (FP16/BF16)
    │
    ▼
[Python] TurboQuantMSE.quantize(x)
    │
    ├─ C++ available? ─> [C++] _C.quantize()
    │                         │
    │                         ├─ CUDA available? ─> [CUDA] tq4_compress_kernel
    │                         │                        (fused norm+rot+quant+pack)
    │                         │
    │                         └─ CPU fallback ─> [C++] codebook_quantize()
    │
    └─ Pure Python fallback ─> [PyTorch] torch.bucketize + rotation matmul
    │
    ▼
Output: {packed_indices (uint8), norms (fp32)}
```

### Data Flow: Attention (Decode)

```
Query q (FP16/BF16), Compressed KV cache
    │
    ▼
[Python] FlashQuantAttentionImpl.forward()
    │
    ├─ Fused paged kernel? ─> [CUDA] paged_tq4_decode_split_k_kernel
    │                              + split_k_reduce_kernel
    │
    ├─ Fused TQ4 attention? ─> [CUDA] fused_tq_attention_decode_kernel
    │
    ├─ Decompress + standard? ─> [CUDA] tq4_decompress_kernel
    │                                 + flash_attention_decode_kernel
    │
    └─ CPU fallback ─> [PyTorch] decompress_reference + attention_reference
    │
    ▼
Output: attention output (FP16/BF16)
```

---

## C++ Core Design Decisions

### Why C++17 (Not C++20 or C++23)

FlashQuant uses the **C++17** standard. This was a deliberate choice:

| Factor | C++17 | C++20/23 |
|--------|:-----:|:--------:|
| **NVCC support** | Full | Partial (C++20 limited, C++23 unsupported) |
| **GCC/Clang support** | GCC 7+, Clang 5+ | GCC 10+/13+, Clang 10+/16+ |
| **PyTorch compatibility** | ATen headers require C++17 | C++20 ATen support is fragile |
| **pybind11 ABI** | Stable | C++20 modules break pybind11 |
| **Feature needs** | Sufficient (structured bindings, if constexpr, fold expressions) | Would use concepts, ranges -- nice but not needed |

The C++17 features actively used:

- **`if constexpr`** -- compile-time kernel dispatch based on template parameters
- **Structured bindings** -- clean tuple unpacking from `torch::Tensor` operations
- **`std::optional`** -- nullable returns without raw pointers
- **Fold expressions** -- variadic template parameter validation
- **Guaranteed copy elision** -- zero-copy tensor returns from C++ to Python

### Static Library Architecture

The build produces two static libraries:

```
flashquant_core (C++ only, no CUDA dependency)
    ├── codebook.cpp     Lloyd-Max codebook computation
    ├── rotation.cpp     Haar QR generation
    ├── quantizer.cpp    TurboQuantMSE, TurboQuantProd
    └── packing.cpp      Nibble pack/unpack

flashquant_cuda (CUDA, links flashquant_core)
    ├── compress.cu
    ├── decompress.cu
    ├── flash_attention.cu
    ├── fused_tq_attention.cu
    ├── paged_decode.cu
    └── split_k_reduce.cu
```

This separation ensures that the C++ core can be used on CPU-only machines without any CUDA dependency. The `_C` Python extension links both when CUDA is available, or just `flashquant_core` otherwise.

### Thread-Safe Codebook Registry

The `CodebookRegistry` is a singleton cache that computes codebooks on first access and caches them per (dim, bits, device) key:

```cpp
class CodebookRegistry {
    static CodebookRegistry& instance();
    const LloydMaxCodebook& get(int dim, int bits, torch::Device device);
    void warmup(const vector<int>& dims, const vector<int>& bits_options,
                torch::Device device);
private:
    mutex mutex_;
    unordered_map<CacheKey, LloydMaxCodebook, CacheKeyHash> cache_;
};
```

Key design points:
- **Thread-safe** via `std::mutex` (vLLM uses multiple threads during model init)
- **Device-aware** -- codebooks are computed on CPU, then `.to(device)` once
- **Warmup API** -- pre-computes all needed codebooks to avoid first-request latency

---

## CUDA Kernel Design

### Multi-Row Blocks

A critical performance optimization is processing multiple rows per CUDA thread block (CTA). The original Triton implementation used one row per CTA, which is catastrophically underutilized during decode:

```
Decode: M = num_heads = 32 (typical)
GPU: 128 SMs (A100/H100)

Single-row blocks: 32 CTAs -> 32/128 = 25% SM utilization
Multi-row blocks:  32/4 = 8 CTAs, but each does 4x work -> same utilization
                   However, shared memory is amortized across 4 rows!
```

FlashQuant uses `ROWS_PER_BLOCK = 4` for compress and decompress kernels. For attention kernels, `BLOCK_M = 64` (prefill) and `BLOCK_M = 1` with `BLOCK_N = 32` (decode) provide appropriate tile sizes.

### Shared Memory Layout

Every kernel carefully plans its shared memory to maximize data reuse:

```
compress.cu shared memory (HALF_D=64, D=128):
    smem_x[128]              = 512 bytes   (input row, reused for norm + rotate)
    smem_rot[32 * 64]        = 8,192 bytes (rotation tile)
    smem_boundaries[15]      = 60 bytes    (quantization boundaries)
    smem_reduce[num_warps]   = 8 bytes     (warp reduction scratch)
    Total: ~8.8 KB per CTA

decompress.cu shared memory:
    smem_centroids[16]       = 64 bytes    (one cache line!)
    Total: 64 bytes per CTA

fused_tq_attention.cu shared memory (prefill, D=128, BLOCK_M=64, BLOCK_N=64):
    k_centroids[16]          = 64 bytes
    v_centroids[16]          = 64 bytes
    q_tile[64 * 128]         = 32,768 bytes
    kv_tile[64 * 128]        = 32,768 bytes
    s_tile[64 * 64]          = 16,384 bytes
    m_i[64]                  = 256 bytes
    l_i[64]                  = 256 bytes
    acc[64 * 128]            = 32,768 bytes
    Total: ~115 KB per CTA
```

### Coalesced Memory Access

The decompress kernel writes output in **sequential layout** (all high-nibble values, then all low-nibble values) rather than interleaved (hi, lo, hi, lo):

```
Sequential (FlashQuant):   [c_hi[0] c_hi[1] ... c_hi[63] | c_lo[0] c_lo[1] ... c_lo[63]]
Interleaved (turboquant):  [c_hi[0] c_lo[0] c_hi[1] c_lo[1] ... c_hi[63] c_lo[63]]
```

Sequential layout enables fully coalesced writes where adjacent threads write to adjacent memory addresses. Interleaved layout forces stride-2 writes, doubling memory transactions.

### Template Specialization

All kernels are templated on `HALF_D` (half the head dimension) to enable compile-time loop unrolling:

```cpp
template <int HALF_D, typename scalar_t>
__global__ void tq4_compress_kernel(...);

// Dispatch:
switch (half_d) {
    case 32:  launch_compress<32, scalar_type>(...);  break;  // D=64
    case 64:  launch_compress<64, scalar_type>(...);  break;  // D=128
    case 128: launch_compress<128, scalar_type>(...); break;  // D=256
}
```

This generates three specialized kernel variants where inner loops are fully unrolled by the CUDA compiler, eliminating loop overhead and enabling register allocation optimizations.

---

## Split-K FlashDecoding

### The Problem

During autoregressive decode, each generated token produces exactly **one** query vector. Standard FlashAttention maps one CTA per (batch, head) pair:

```
CTAs = batch_size * num_q_heads
     = 1 * 32 = 32  (single-sequence inference)
```

With 128 SMs on an A100, **75% of SMs sit idle**. The kernel is latency-bound, not throughput-bound.

### The Solution: Split-K

Split-K FlashDecoding (Dao et al., 2023) partitions the KV sequence across multiple CTAs per (batch, head):

```
CTAs = batch_size * num_q_heads * NUM_SPLITS
     = 1 * 32 * 4 = 128  (fully saturates A100)
```

Each CTA processes a `1/NUM_SPLITS` chunk of the KV sequence and produces a **partial softmax state** `{acc_i, m_i, l_i}`:

```
CTA 0: processes KV[0..S/4]        -> {acc_0, m_0, l_0}
CTA 1: processes KV[S/4..S/2]      -> {acc_1, m_1, l_1}
CTA 2: processes KV[S/2..3S/4]     -> {acc_2, m_2, l_2}
CTA 3: processes KV[3S/4..S]       -> {acc_3, m_3, l_3}
```

### Log-Sum-Exp Reduction

A second kernel (`split_k_reduce.cu`) combines the partial states into the final output using the numerically stable log-sum-exp formula:

```
m_final = max(m_0, m_1, m_2, m_3)

l_final = l_0 * exp2(m_0 - m_final)
        + l_1 * exp2(m_1 - m_final)
        + l_2 * exp2(m_2 - m_final)
        + l_3 * exp2(m_3 - m_final)

acc_final = (acc_0 * l_0 * exp2(m_0 - m_final)
           + acc_1 * l_1 * exp2(m_1 - m_final)
           + acc_2 * l_2 * exp2(m_2 - m_final)
           + acc_3 * l_3 * exp2(m_3 - m_final)) / l_final
```

This is exact -- no approximation is introduced by the split.

### Why NUM_SPLITS = 4

| Splits | CTAs (B=1, H=32) | SM util (128 SMs) | Overhead |
|--------|:-----------------:|:-----------------:|:--------:|
| 1 | 32 | 25% | None |
| 2 | 64 | 50% | +1 reduction kernel |
| **4** | **128** | **100%** | **+1 reduction kernel** |
| 8 | 256 | 100% (2 waves) | +1 reduction, more partials |

NUM_SPLITS=4 fully saturates the GPU while minimizing the overhead of the reduction kernel and the size of the partial output buffers.

---

## Fused TQ4 Attention

### Motivation

The standard path for compressed attention is:

```
1. Decompress all K tokens:    K_packed -> K_decompressed   (HBM write: N_KV * D * 2 bytes)
2. Decompress all V tokens:    V_packed -> V_decompressed   (HBM write: N_KV * D * 2 bytes)
3. Run FlashAttention:         Q, K_decompressed, V_decompressed -> Out
```

Steps 1-2 write the full decompressed KV cache to HBM, then step 3 reads it back. This doubles memory bandwidth usage.

### Fused Approach

FlashQuant fuses decompression into the FlashAttention-2 tile loop:

```
For each KV tile [BLOCK_N, D]:
    1. Load K_packed[tile] from HBM
    2. Decompress inline: unpack nibbles -> centroid gather from smem -> scale by norm
       (Result lives in shared memory, never touches HBM)
    3. Compute Q @ K^T (from shared memory)
    4. Online softmax update
    5. Load V_packed[tile] from HBM
    6. Decompress inline (same as step 2)
    7. Accumulate P @ V (from shared memory)
```

### Memory Bandwidth Savings

```
Standard path:
    Read:  K_packed + V_packed + K_decompressed + V_decompressed = 2 * (N_KV * D/2) + 2 * (N_KV * D * 2)
    Write: K_decompressed + V_decompressed + Output             = 2 * (N_KV * D * 2) + N_Q * D * 2

Fused path:
    Read:  K_packed + V_packed + Q                               = 2 * (N_KV * D/2) + N_Q * D * 2
    Write: Output                                                = N_Q * D * 2
```

For N_KV = 4096, D = 128:
- Standard: reads 4 MB, writes 2 MB -> 6 MB total
- Fused: reads 0.5 MB + 0.5 KB, writes 0.25 KB -> **0.5 MB total (12x reduction)**

### Separate K/V Centroids

The fused kernel loads separate centroids for keys and values into shared memory:

```cpp
__shared__ float k_centroids_smem[16];  // 64 bytes
__shared__ float v_centroids_smem[16];  // 64 bytes
// Total: 128 bytes -- fits in a single cache line pair
```

This supports configurations where keys and values use different codebooks (e.g., different bit allocations or seeds).

---

## pybind11 Binding Strategy

### Separate Bind Files per Module

The C++ extension is organized as one bind file per functional module:

```
csrc/bindings/
    module.cpp           # PYBIND11_MODULE(_C, m) entry point
    codebook_bind.cpp    # Codebook registry + quantize/dequantize
    quantizer_bind.cpp   # TurboQuantMSE, TurboQuantProd classes
    packing_bind.cpp     # nibble_pack, nibble_unpack
```

Benefits:
- **Parallel compilation** -- each `.cpp` file compiles independently
- **Encapsulation** -- each module only exposes its public API
- **Incremental builds** -- modifying one bind file recompiles only that translation unit

### Module Entry Point

```cpp
// module.cpp
#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_codebook(py::module&);
void bind_quantizer(py::module&);
void bind_packing(py::module&);

PYBIND11_MODULE(_C, m) {
    m.doc() = "FlashQuant C++/CUDA extension";
    bind_codebook(m);
    bind_quantizer(m);
    bind_packing(m);
}
```

---

## CPU Reference Fallback Chain

Every CUDA kernel has a corresponding pure-PyTorch CPU reference in `src/flashquant/kernels/cpu_reference.py`. The dispatch chain is:

```python
def compress(x, rotation, boundaries):
    if _has_cuda_kernel and x.is_cuda:
        return _C.tq4_compress(x, rotation, boundaries)
    return compress_reference(x, rotation, boundaries)
```

This three-tier fallback ensures FlashQuant works on:
1. **GPU with compiled extension** -- maximum performance via CUDA kernels
2. **CPU with compiled extension** -- C++ codebook operations, PyTorch fallback for kernels
3. **CPU without extension** -- pure Python/PyTorch, no compilation needed at all

All 264 tests pass on the pure-Python path, ensuring correctness is verifiable without GPU access.

---

## vLLM Backend Architecture

### Plugin Registration

FlashQuant registers as a vLLM attention backend via the `vllm.general_plugins` entry point:

```toml
# pyproject.toml
[project.entry-points."vllm.general_plugins"]
flashquant = "flashquant.vllm:register_flashquant_backend"
```

When vLLM starts, it discovers and calls `register_flashquant_backend()`, which injects the FlashQuant backend class into vLLM's attention backend registry.

### Dynamic Class Creation

vLLM base classes are imported lazily to avoid hard dependencies:

```python
def _create_backend_class() -> type:
    # Import at runtime, not at module load
    from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend

    class FlashQuantAttentionBackend(FlashAttentionBackend):
        forward_includes_kv_cache_update = True

        @staticmethod
        def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size, ...):
            total_bytes = num_kv_heads * tq4_bytes_per_token_kv(head_size)
            return (num_blocks, block_size, total_bytes)
        ...

    return FlashQuantAttentionBackend
```

### KV Cache Layout

The vLLM paged cache stores compressed KV data as raw bytes:

```
Per token, per KV head group:
┌──────────────────┬────────┬──────────────────┬────────┐
│ K indices (D/2B) │ K norm │ V indices (D/2B) │ V norm │
│   nibble-packed  │ fp32   │   nibble-packed  │ fp32   │
│   uint8[D/2]     │ 4B     │   uint8[D/2]     │ 4B     │
└──────────────────┴────────┴──────────────────┴────────┘

Total per token per head: D/2 + 4 + D/2 + 4 = D + 8 bytes
For D=128: 136 bytes per token per KV head
```

### CUDA Graph Support

FlashQuant declares `UNIFORM_SINGLE_TOKEN_DECODE` CUDA graph support, enabling vLLM to capture the decode path as a CUDA graph for reduced kernel launch overhead. The prefill path has dynamic shapes and cannot be captured.

---

*Copyright 2026 Ayi NEDJIMI. Apache License 2.0.*
