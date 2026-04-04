> [Back to Documentation Index](index.md) | [Algorithm](Algorithm-Deep-Dive.md) | [Architecture](Architecture.md) | [CUDA Kernels](CUDA-Kernels.md) | [Integration](Integration-Guide.md) | [Testing](Testing.md) | [Improvements](Improvements-over-turboquant-vllm.md)

# Integration Guide

This page covers how to use FlashQuant in three modes: standalone compression, HuggingFace DynamicCache integration, and vLLM backend plugin.

---

## Table of Contents

1. [Installation](#installation)
2. [Standalone Compression](#standalone-compression)
3. [HuggingFace DynamicCache Integration](#huggingface-dynamiccache-integration)
4. [vLLM Backend Plugin](#vllm-backend-plugin)
5. [Configuration Reference](#configuration-reference)
6. [Environment Variables](#environment-variables)
7. [Troubleshooting](#troubleshooting)

---

## Installation

### Prerequisites

| Component | Version | Required? |
|-----------|---------|:---------:|
| Python | 3.10+ | Yes |
| PyTorch | 2.4+ | Yes |
| NVIDIA GPU | Ampere+ (SM 80+) | Optional (CPU fallback available) |
| CUDA Toolkit | 12.x | Optional (only for CUDA kernels) |
| CMake | 3.20+ | Optional (only for C++ build) |
| pybind11 | 2.13+ | Optional (only for C++ build) |

### CPU-Only (No Compilation)

```bash
git clone https://github.com/ayinedjimi/flashquant.git
cd flashquant
pip install -e .
```

This installs the pure-Python package. All features work via PyTorch fallback paths, just without CUDA kernel acceleration.

### With CUDA Kernels

```bash
git clone https://github.com/ayinedjimi/flashquant.git
cd flashquant

pip install -e ".[dev]"

# Build C++ extensions and CUDA kernels
cmake -B build -DFLASHQUANT_CUDA=ON
cmake --build build -j$(nproc)
```

### With vLLM Support

```bash
pip install -e ".[vllm]"
```

This installs `vllm>=0.18,<0.22` as an additional dependency.

---

## Standalone Compression

### TurboQuantMSE (Stage 1 -- MSE-Optimal)

Best for **value cache** compression where reconstruction quality matters.

```python
import torch
from flashquant import TurboQuantMSE

# Create quantizer for head_dim=128, 4-bit
quantizer = TurboQuantMSE(dim=128, bits=4, seed=42)

# Input: any shape ending in dim
x = torch.randn(32, 16, 128)  # [batch, heads, dim]

# Compress
indices, norms = quantizer.quantize(x)
# indices: (32, 16, 128) int64 -- centroid indices
# norms:   (32, 16, 1)   float32 -- vector norms

# Decompress
x_hat = quantizer.dequantize(indices, norms)
# x_hat: (32, 16, 128) float32

# Verify quality
cos_sim = torch.nn.functional.cosine_similarity(
    x.flatten(0, -2), x_hat.flatten(0, -2), dim=-1
).mean()
print(f"Cosine similarity: {cos_sim:.4f}")  # >= 0.95 for 4-bit
```

### TurboQuantProd (Stage 1+2 -- Unbiased Inner Products)

Best for **key cache** compression where attention score accuracy matters.

```python
import torch
from flashquant import TurboQuantProd

# Create quantizer (allocates bits-1=3 for MSE, 1 for QJL)
quantizer = TurboQuantProd(dim=128, bits=4, seed=42)

# Compress keys
keys = torch.randn(32, 16, 128)
indices, norms, qjl_signs, residual_norms = quantizer.quantize(keys)
# indices:        (32, 16, 128) int64
# norms:          (32, 16, 1)   float32
# qjl_signs:      (32, 16, 128) float32 (sign bits: +1 or -1)
# residual_norms: (32, 16, 1)   float32

# Estimate inner products (unbiased!)
queries = torch.randn(32, 16, 128)
scores = quantizer.estimate_inner_product(
    queries, indices, norms, qjl_signs, residual_norms
)
# scores: (32, 16, 1) float32

# Compare with true inner products
true_scores = (queries * keys).sum(dim=-1, keepdim=True)
error = (scores - true_scores).abs().mean()
print(f"Mean absolute error: {error:.6f}")
```

### Compression with Packing

For storage-efficient compression using nibble packing:

```python
from flashquant.core.packing import nibble_pack, nibble_unpack

# After quantization
indices, norms = quantizer.quantize(x)

# Pack: 2 indices per byte (4-bit)
packed = nibble_pack(indices)  # Half the storage

# Unpack
recovered_indices = nibble_unpack(packed)
assert (recovered_indices == indices).all()
```

---

## HuggingFace DynamicCache Integration

FlashQuant provides `CompressedDynamicCache` as a drop-in replacement for HuggingFace's `DynamicCache`.

### Basic Usage

```python
from flashquant.cache import CompressedDynamicCache
from flashquant import FlashQuantConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8B")

# Create compressed cache
config = FlashQuantConfig(bits=4, max_seq_len=32768)
cache = CompressedDynamicCache(
    config,
    num_layers=32,
    num_heads=32,
    head_dim=128,
)

# Generate with compressed cache
inputs = tokenizer("The future of AI is", return_tensors="pt")
outputs = model.generate(
    **inputs,
    past_key_values=cache,
    max_new_tokens=100,
)

print(tokenizer.decode(outputs[0]))
print(f"Cache memory: {cache.vram_bytes() / 1e6:.1f} MB")
```

### O(1) Append

The cache uses pre-allocated ring buffers for O(1) token append during decoding:

```python
# No torch.cat! Pre-allocated buffer with write pointer
cache.update(key_states, value_states, layer_idx)
# Writes to buffer[write_ptr] and increments write_ptr
# Time: O(1) regardless of sequence length
```

### Memory Tracking

```python
# Accurate memory accounting (counts ALL buffers)
total_bytes = cache.vram_bytes()
print(f"Compressed cache: {total_bytes / 1e6:.1f} MB")
print(f"FP16 equivalent:  {total_bytes * 7.5 / 1e6:.1f} MB")
```

---

## vLLM Backend Plugin

FlashQuant integrates with vLLM (v0.18-0.22) as a custom attention backend.

### Setup

**Option 1: Entry Point Registration (Recommended)**

```bash
# Install with vLLM support
pip install -e ".[vllm]"

# FlashQuant auto-registers via pyproject.toml entry point
# Start vLLM normally:
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3-8B \
    --attention-backend CUSTOM \
    --max-model-len 32768
```

**Option 2: Explicit Registration**

```python
from flashquant.vllm import register_flashquant_backend

# Register before creating LLM
register_flashquant_backend()

from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-3-8B",
    attention_backend="flashquant",
    max_model_len=32768,
)
outputs = llm.generate("The future of AI is", max_tokens=100)
```

### How It Works

1. **Registration:** `register_flashquant_backend()` injects `FlashQuantAttentionBackend` into vLLM's backend registry.

2. **Cache Shape:** The backend overrides `get_kv_cache_shape()` to return a 3D `(num_blocks, block_size, bytes_per_token)` layout for the packed TQ4 format.

3. **Forward Pass:** The attention implementation:
   - Compresses new K/V tokens and writes them into the paged cache
   - Runs the fused paged TQ4 decode kernel for decode steps
   - Falls back to decompress-all + standard attention for prefill

4. **CUDA Graphs:** Supports `UNIFORM_SINGLE_TOKEN_DECODE` CUDA graph capture for decode steps.

### Supported vLLM Versions

| vLLM Version | Status | Notes |
|:------------:|:------:|-------|
| 0.18.x | Supported | v1 attention backend API |
| 0.19.x | Supported | |
| 0.20.x | Supported | |
| 0.21.x | Supported | |
| < 0.18 | Not supported | Lacks v1 backend API |
| >= 0.22 | Untested | API may have changed |

---

## Configuration Reference

### FlashQuantConfig

```python
from flashquant import FlashQuantConfig

config = FlashQuantConfig(
    bits=4,              # Quantization bit-width (2, 3, or 4)
    key_seed=42,         # Random seed for key rotation matrix
    value_seed=43,       # Random seed for value rotation matrix (must differ from key_seed)
    use_fused_paged=True,  # Enable fused paged TQ4 decode kernel
    use_cuda_graphs=True,  # Enable CUDA graph capture for decode
    max_seq_len=32768,   # Maximum sequence length (controls buffer allocation)
)

# Validate (raises FlashQuantConfigError if invalid)
config.validate()
```

### Parameter Details

| Parameter | Type | Default | Valid Values | Description |
|-----------|------|---------|-------------|-------------|
| `bits` | int | 4 | 2, 3, 4 | Quantization bits per coordinate |
| `key_seed` | int | 42 | any int | Seed for key rotation matrix |
| `value_seed` | int | 43 | any int | Seed for value rotation matrix (must differ from `key_seed`) |
| `use_fused_paged` | bool | True | True, False | Enable fused paged TQ4 decode kernel |
| `use_cuda_graphs` | bool | True | True, False | Enable CUDA graph capture |
| `max_seq_len` | int | 32768 | 1 - 1,048,576 | Maximum supported sequence length |

### Bit-Width Selection Guide

| Bits | Levels | Compression | Cosine Sim | Use Case |
|:----:|:------:|:-----------:|:----------:|----------|
| 4 | 16 | 7.5x | >= 0.95 | Production (recommended) |
| 3 | 8 | 10x | >= 0.92 | Memory-constrained |
| 2 | 4 | 15x | >= 0.80 | Research / exploration |

---

## Environment Variables

FlashQuantConfig can be configured entirely via environment variables, useful for Docker/Kubernetes deployments:

```bash
# All optional -- defaults apply when not set
export FLASHQUANT_BITS=4
export FLASHQUANT_KEY_SEED=42
export FLASHQUANT_VALUE_SEED=43
export FLASHQUANT_USE_FUSED_PAGED=true    # "1", "true", or "yes"
export FLASHQUANT_USE_CUDA_GRAPHS=true
export FLASHQUANT_MAX_SEQ_LEN=32768
```

Load from environment:

```python
config = FlashQuantConfig.from_env()
```

---

## Troubleshooting

### Common Issues

**Issue: "CUDA error: no kernel image is available for execution on the device"**

Your GPU's compute capability is not in the build targets. FlashQuant targets SM 80, 86, 89, 90 (Ampere, Ada Lovelace, Hopper). If you have an older GPU:

```bash
# Build for your specific SM version (e.g., SM 75 for Turing)
cmake -B build -DFLASHQUANT_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="75"
cmake --build build -j$(nproc)
```

**Issue: "ImportError: cannot import name '_C' from 'flashquant'"**

The C++ extension is not compiled. FlashQuant will fall back to pure Python automatically. To build:

```bash
cmake -B build -DFLASHQUANT_CUDA=ON
cmake --build build -j$(nproc)
```

**Issue: "FlashQuantConfigError: key_seed and value_seed must differ"**

Keys and values must use independent rotation matrices (different seeds). Use the defaults: `key_seed=42, value_seed=43`.

**Issue: "RuntimeError: CUDA out of memory" with large max_seq_len**

`max_seq_len` controls pre-allocated buffer sizes. Reduce it if VRAM is limited:

```python
config = FlashQuantConfig(max_seq_len=8192)  # Instead of 32768
```

**Issue: "vLLM does not recognize flashquant backend"**

Ensure FlashQuant is installed in the same environment as vLLM, and that you have called `register_flashquant_backend()` before creating the LLM instance. Alternatively, install via `pip install -e ".[vllm]"` which enables entry-point auto-registration.

**Issue: Low cosine similarity (< 0.90) with 4-bit quantization**

This may indicate very small head dimensions (< 64). FlashQuant's Gaussian approximation degrades for small dimensions. The C++ core automatically switches to exact Beta-distribution Lloyd-Max for dim < 64:

```cpp
// Automatic dispatch in codebook.cpp
LloydMaxCodebook solve_lloyd_max(int dim, int bits) {
    if (dim >= 64) return gaussian_lloyd_max(dim, bits);
    else           return beta_lloyd_max(dim, bits);
}
```

**Issue: NaN in output with zero-length sequences**

The paged decode kernel handles zero-length sequences gracefully by writing sentinel partial states (`m = -FLT_MAX, l = 1, acc = 0`). If you encounter NaN, check that `seq_lens` contains only non-negative values.

---

---
*[FlashQuant](https://github.com/ayinedjimi/flashquant) — By [Ayi NEDJIMI](https://ayinedjimi-consultants.fr)*
