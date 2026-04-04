## FlashQuant Wiki

**[Home](Home)**

---

### Core

- [Algorithm Deep Dive](Algorithm-Deep-Dive)
  - PolarQuant
  - Lloyd-Max via erfinv
  - QJL Correction
  - Distortion Bounds
- [Architecture](Architecture)
  - Layer Diagram
  - C++ Core Design
  - Split-K FlashDecoding
  - Fused TQ4 Attention
  - pybind11 Strategy
- [CUDA Kernels](CUDA-Kernels)
  - compress.cu
  - decompress.cu
  - flash_attention.cu
  - fused_tq_attention.cu
  - paged_decode.cu
  - split_k_reduce.cu

---

### Usage

- [Integration Guide](Integration-Guide)
  - Standalone Compression
  - HuggingFace Cache
  - vLLM Backend
  - Configuration
  - Environment Variables

---

### Quality

- [Testing](Testing)
  - 264 Tests, 0 Failures
  - Adversarial Tests
  - Numerical Bounds
  - Long Context Tests
- [Improvements](Improvements-over-turboquant-vllm)
  - P0 Correctness Fixes
  - P1 Performance Fixes
  - C++ vs Python/Triton
  - Dependency Reduction

---

### Links

- [Paper (arXiv 2504.19874)](https://arxiv.org/abs/2504.19874)
- [Repository](https://github.com/ayinedjimi/flashquant)
- [Author](https://ayinedjimi-consultants.fr)
