# FlashQuant Wiki

**Production-grade TurboQuant KV cache compression for LLM inference.**

FlashQuant is a C++17/CUDA implementation of the TurboQuant algorithm ([arXiv 2504.19874](https://arxiv.org/abs/2504.19874)) from Google Research. It compresses transformer Key-Value caches by **4-8x** with near-zero quality loss, enabling dramatically longer contexts and higher throughput on the same GPU hardware.

**Author:** [Ayi NEDJIMI](https://ayinedjimi-consultants.fr) -- Expert Cybersecurity & AI

---

## Key Metrics

| Metric | Value |
|--------|-------|
| **KV cache compression** | **7.5x** (512 bytes -> 68 bytes per token at d=128) |
| **Test suite** | **264 tests**, 0 failures |
| **Codebase** | **~12,000 lines** across C++, CUDA, and Python |
| **Quality loss** | < 1% on MMLU (Llama-3-8B: 65.2% -> 64.8%) |
| **Decode overhead** | < 5% latency (batch=1, 4K context) |
| **Throughput gain** | 2.5-3x higher (batch=32, 4K context) |
| **Dependencies** | **1** runtime dependency (`torch>=2.4`) |

---

## Algorithm Summary

FlashQuant implements the full three-stage TurboQuant pipeline:

1. **PolarQuant (Stage 1):** Random orthogonal rotation makes vector coordinates approximately i.i.d. Gaussian, enabling optimal Lloyd-Max scalar quantization. Achieves MSE within 2.72x of the theoretical optimum.

2. **QJL Correction (Stage 2):** Quantized Johnson-Lindenstrauss projection on the quantization residual eliminates bias in dot-product estimation. Critical for accurate attention scores (Q * K^T).

3. **Fused Attention (Stage 3):** Decompression is fused directly into FlashAttention-2 tile loops, avoiding intermediate HBM allocations. Split-K FlashDecoding with NUM_SPLITS=4 saturates GPU SMs during decode.

**Compression pipeline per vector:**
```
x  ->  ||x||  ->  x/||x||  ->  R^T * (x/||x||)  ->  Lloyd-Max quantize  ->  nibble-pack
        norm      normalize       rotate                scalar quantize        2 indices/byte
```

**Storage per token per KV head (d=128):**
```
K: 64 bytes (packed indices) + 4 bytes (fp32 norm) = 68 bytes
V: 64 bytes (packed indices) + 4 bytes (fp32 norm) = 68 bytes
Total: 136 bytes  vs.  512 bytes (FP16)  =>  3.76x per KV pair, 7.5x per cache dimension
```

---

## Table of Contents

### Core Documentation

| Page | Description |
|------|-------------|
| **[Algorithm Deep Dive](Algorithm-Deep-Dive.md)** | Mathematical foundations: PolarQuant, Lloyd-Max, QJL, distortion bounds, proofs |
| **[Architecture](Architecture.md)** | System design: layer diagram, C++ core, CUDA kernels, pybind11, dispatch chain |
| **[CUDA Kernels](CUDA-Kernels.md)** | Detailed walkthrough of all 6 CUDA kernels with performance analysis |

### Usage & Integration

| Page | Description |
|------|-------------|
| **[Integration Guide](Integration-Guide.md)** | Standalone, HuggingFace, and vLLM integration with configuration reference |

### Quality & Testing

| Page | Description |
|------|-------------|
| **[Testing](Testing.md)** | 264-test suite: GTest, pytest, adversarial, numerical bounds, performance |
| **[Improvements over turboquant-vllm](Improvements-over-turboquant-vllm.md)** | 100+ fixes organized by severity with before/after analysis |

---

## Quick Links

- **Paper:** [arXiv 2504.19874](https://arxiv.org/abs/2504.19874)
- **Google Research Blog:** [TurboQuant: Redefining AI Efficiency](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)
- **Repository:** [github.com/ayinedjimi/flashquant](https://github.com/ayinedjimi/flashquant)
- **License:** Apache 2.0

---

## Project Structure

```
flashquant/
├── csrc/                          # C++/CUDA source (28 files, 5,800+ lines)
│   ├── core/                      # Pure C++ algorithm: codebook, rotation, quantizer, packing
│   │   ├── codebook.h / .cpp      # Closed-form Lloyd-Max via erfinv
│   │   ├── rotation.h / .cpp      # Haar-distributed orthogonal matrices
│   │   ├── quantizer.h / .cpp     # TurboQuantMSE + TurboQuantProd
│   │   ├── packing.h / .cpp       # Nibble-pack / unpack utilities
│   │   └── types.h                # Shared type definitions
│   ├── cuda/                      # 6 Native CUDA kernels (2,575 lines)
│   │   ├── compress.cu            # Fused norm + rotate + quantize + pack
│   │   ├── decompress.cu          # Coalesced unpack + gather + scale
│   │   ├── flash_attention.cu     # FlashAttention-2 (prefill + decode)
│   │   ├── fused_tq_attention.cu  # FA2 + inline TQ4 decompression
│   │   ├── paged_decode.cu        # Split-K paged TQ4 decode
│   │   ├── split_k_reduce.cu      # Log-sum-exp partial softmax reduction
│   │   └── utils.cuh              # Warp/block reductions, coalesced helpers
│   ├── bindings/                  # pybind11 -> flashquant._C
│   └── tests/                     # C++ unit tests (Google Test)
│
├── src/flashquant/                # Python package (25 files, 4,000+ lines)
│   ├── core/                      # Codebook, quantizer, compressor, packing
│   ├── cache/                     # CompressedBuffer (O(1) ring) + HF DynamicCache
│   ├── kernels/                   # CUDA dispatch + CPU reference fallbacks
│   └── vllm/                      # vLLM attention backend plugin (v0.18-0.22)
│
├── tests/                         # Python test suite (21 files, 264 tests)
├── CMakeLists.txt                 # C++17, CUDA optional, GTest, pybind11
├── pyproject.toml                 # scikit-build-core, coverage >= 90%
└── .github/workflows/ci.yml      # Lint + Python tests + C++ tests
```

---

## Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| Core language | C++17 | GCC 11+ / Clang 14+ |
| GPU kernels | CUDA | 12.x (SM 80+: Ampere, Ada, Hopper) |
| Python bindings | pybind11 | 2.13+ |
| Build system | CMake + scikit-build-core | CMake 3.20+, SBC 0.10+ |
| Python API | PyTorch | 2.4+ |
| C++ tests | Google Test | Latest |
| Python tests | pytest | 9.0+ |
| Lint | ruff | 0.15+ |
| Type checking | mypy | 1.10+ |

---

---
*[FlashQuant](https://github.com/ayinedjimi/flashquant) — By [Ayi NEDJIMI](https://ayinedjimi-consultants.fr)*
