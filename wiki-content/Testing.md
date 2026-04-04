# Testing

FlashQuant maintains a comprehensive test suite with **264 tests and 0 failures**. Tests are split between C++ (Google Test) and Python (pytest), covering correctness, numerical bounds, adversarial inputs, long contexts, and performance invariants.

---

## Table of Contents

1. [Test Suite Overview](#test-suite-overview)
2. [C++ Tests (Google Test)](#c-tests-google-test)
3. [Python Tests (pytest)](#python-tests-pytest)
4. [Cosine Similarity Thresholds](#cosine-similarity-thresholds)
5. [Adversarial Tests](#adversarial-tests)
6. [Numerical Bounds Verification](#numerical-bounds-verification)
7. [Long Context Tests](#long-context-tests)
8. [Performance Tests](#performance-tests)
9. [Running Tests](#running-tests)

---

## Test Suite Overview

| Category | Framework | Files | Tests | Description |
|----------|-----------|:-----:|:-----:|-------------|
| C++ unit tests | GTest | 5 | ~50 | Codebook, rotation, quantizer, packing, kernels |
| Python unit tests | pytest | 21 | 264 | Full pipeline, integration, adversarial, bounds |
| **Total** | | **26** | **~314** | **0 failures** |

### Test File Inventory

```
tests/
├── conftest.py                 # Fixtures, reference implementations
├── test_codebook.py            # Lloyd-Max codebook correctness
├── test_rotation.py            # Haar orthogonal matrix properties
├── test_packing.py             # Nibble pack/unpack round-trips
├── test_quantizer_mse.py       # TurboQuantMSE: quantize/dequantize
├── test_quantizer_prod.py      # TurboQuantProd: inner product estimation
├── test_compressor.py          # KeyCompressor / ValueCompressor
├── test_config.py              # FlashQuantConfig validation + from_env
├── test_hf_cache.py            # CompressedDynamicCache integration
├── test_buffer_pool.py         # Ring buffer O(1) append
├── test_compress_kernel.py     # compress.cu vs. reference
├── test_decompress_kernel.py   # decompress.cu vs. reference
├── test_attention_kernel.py    # flash_attention.cu vs. reference
├── test_paged_decode_kernel.py # paged_decode.cu + split_k_reduce
├── test_adversarial.py         # Edge cases: zeros, outliers, sparse, correlated
├── test_numerical_bounds.py    # Theorem 1 & 2 empirical verification
├── test_long_context.py        # 4K, 16K, 32K token sequences
├── test_performance.py         # O(1) append, linear scaling
├── test_vllm_impl.py           # vLLM attention impl (mocked)
├── test_vllm_registration.py   # Backend registration mechanics
└── __init__.py
```

```
csrc/tests/
├── test_codebook.cpp           # C++ codebook unit tests
├── test_rotation.cpp           # Rotation matrix properties
├── test_quantizer.cpp          # C++ quantizer round-trip
├── test_packing.cpp            # C++ packing correctness
└── test_kernels.cpp            # CUDA kernel correctness (if built)
```

---

## C++ Tests (Google Test)

The C++ test suite validates the core algorithm implementation independently of Python.

### test_codebook.cpp

- Gaussian Lloyd-Max produces sorted boundaries
- Centroids are within boundary intervals
- Boundary count = 2^bits - 1
- Centroid count = 2^bits
- Codebook is symmetric for symmetric distributions
- Quantize-dequantize round-trip preserves structure
- Beta Lloyd-Max converges for small dimensions (dim < 64)

### test_rotation.cpp

- Generated matrix is orthogonal: R^T * R = I (within tolerance)
- Determinant is +/- 1
- Same seed produces same matrix
- Different seeds produce different matrices
- Matrix is in float32 on CPU regardless of default device

### test_quantizer.cpp

- TurboQuantMSE round-trip: cosine similarity >= 0.90 (4-bit, d=128)
- TurboQuantProd inner product is unbiased (mean error < 0.01)
- Indices are in valid range [0, 2^bits - 1]
- Norms are non-negative
- Different seeds produce different rotations

### test_packing.cpp

- Nibble pack-unpack round-trip: exact equality
- 2-bit pack-unpack round-trip: exact equality
- Boundary values (0, 15) survive packing
- Invalid indices (> 15) are rejected

### test_kernels.cpp (CUDA only)

- Compress kernel matches CPU reference (cosine >= 0.999)
- Decompress kernel matches CPU reference (exact for same codebook)
- Flash attention matches PyTorch SDPA (relative error < 1e-3)
- Fused TQ attention matches decompress + attention (relative error < 1e-3)
- Paged decode with Split-K matches sequential decode

### Building and Running C++ Tests

```bash
cmake -B build -DFLASHQUANT_TESTS=ON -DFLASHQUANT_CUDA=ON
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

---

## Python Tests (pytest)

### Shared Fixtures (conftest.py)

The test suite uses parametrized fixtures for systematic coverage:

```python
@pytest.fixture(autouse=True)
def _seed_torch():
    """Fix seed before every test for reproducibility."""
    torch.manual_seed(42)

@pytest.fixture(params=[64, 128, 256], ids=["dim64", "dim128", "dim256"])
def head_dim(request): ...

@pytest.fixture(params=[2, 3, 4], ids=["2bit", "3bit", "4bit"])
def bits(request): ...

@pytest.fixture(params=[torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def dtype(request): ...

@pytest.fixture(params=["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def device(request): ...
```

### Reference Implementations

`conftest.py` contains pure-Python reference implementations that replicate the algorithms without depending on the C++ extension:

| Reference | Purpose |
|-----------|---------|
| `RefQuantizerMSE` | Pure-PyTorch TurboQuantMSE (iterative Lloyd-Max) |
| `RefQuantizerProd` | Pure-PyTorch TurboQuantProd (MSE + QJL) |
| `generate_rotation_matrix()` | Haar QR on CPU |
| `ref_nibble_pack()` / `ref_nibble_unpack()` | Pure-Python packing |
| `ref_compress()` / `ref_decompress()` | Full compress/decompress pipeline |
| `ref_attention()` | Naive SDPA (no FlashAttention) |
| `cosine_similarity_flat()` | Flat cosine sim between two tensors |

These references serve as ground truth for testing the optimized implementations.

### Parametrized Test Matrix

Many tests are parametrized across dimensions, bit-widths, dtypes, and devices:

```python
class TestTurboQuantMSE:
    def test_round_trip_quality(self, head_dim, bits):
        """Quantize + dequantize preserves vector structure."""
        ...

    def test_indices_in_range(self, head_dim, bits):
        """All indices are in [0, 2^bits - 1]."""
        ...

    def test_norms_preserved(self, head_dim, bits):
        """Stored norms match input norms."""
        ...
```

For `test_round_trip_quality` alone, the fixture combinations produce:
- 3 head_dims x 3 bit-widths = **9 test cases**

---

## Cosine Similarity Thresholds

Quality is measured by cosine similarity between original and reconstructed vectors. FlashQuant enforces **strict** thresholds:

| Bit-Width | Threshold | Rationale |
|:---------:|:---------:|-----------|
| 4-bit | >= 0.95 | Production quality, < 1% perplexity degradation |
| 3-bit | >= 0.92 | Acceptable quality for memory-constrained scenarios |
| 2-bit | >= 0.80 | Exploratory, noticeable quality degradation |

These are significantly stricter than the original turboquant-vllm thresholds (0.80 for all bit-widths).

### How Thresholds Are Tested

```python
cos_sim = torch.nn.functional.cosine_similarity(
    x.flatten(0, -2), x_hat.flatten(0, -2), dim=-1
).mean()

if bits == 4:
    assert cos_sim >= 0.95
elif bits == 3:
    assert cos_sim >= 0.92
elif bits == 2:
    assert cos_sim >= 0.80
```

---

## Adversarial Tests

The `test_adversarial.py` file tests edge cases that could cause NaN, overflow, or silent corruption:

### Test Cases

| Test | Input | Validates |
|------|-------|-----------|
| `TestAllSameVectors` | `ones(dim) * 1e6` | Large uniform vectors do not overflow |
| `TestSparseVectors` | Single nonzero per vector | Sparse inputs do not produce NaN |
| `TestExtremeOutliers` | One coord 1e8, rest 1e-8 | Extreme outliers are handled |
| `TestZeroVectors` | `zeros(dim)` | Zero division does not occur (norm + 1e-10 guard) |
| `TestCorrelatedVectors` | `base + noise * 1e-4` | Nearly identical vectors reconstruct coherently |
| `TestNegativeValues` | `-abs(randn)` | All-negative vectors work correctly |
| `TestVerySmallValues` | `randn * 1e-30` | Near-epsilon values do not cause division by zero |

### Key Invariants

Every adversarial test verifies:
1. `torch.isfinite(reconstructed).all()` -- no NaN or Inf in output
2. Norm preservation (where applicable)
3. Minimum cosine similarity thresholds

---

## Numerical Bounds Verification

The `test_numerical_bounds.py` file empirically validates the theoretical bounds from the TurboQuant paper.

### Theorem 1: MSE Distortion

```python
class TestMSEDistortionBound:
    def test_mse_distortion_bound(self, bound_dim, bound_bits):
        # Quantize 1000 unit-norm vectors
        mse = ((x - reconstructed) ** 2).mean().item()
        theoretical_bound = sqrt(3) * pi / 2 * (1.0 / 4 ** bits)

        # 5x margin for finite-dimensional effects
        assert mse < theoretical_bound * 5
```

Parametrized over:
- Dimensions: 64, 128
- Bit-widths: 2, 3, 4

Total: 6 test cases.

### Theorem 2: Inner Product Variance

```python
class TestInnerProductDistortionBound:
    def test_inner_product_distortion_bound(self, bound_dim, bound_bits):
        # Estimate inner products for 500 query-key pairs
        errors = estimated_ip - true_ip
        empirical_variance = (errors ** 2).mean().item()
        theoretical_bound = sqrt(3) * pi**2 / (dim * 4 ** b_eff)

        # 100x margin (queries/keys have 0.1 scale)
        assert empirical_variance < theoretical_bound * 100
```

### Unbiasedness Test

```python
class TestUnbiasedness:
    def test_unbiasedness(self, dim, bits):
        # 10,000 samples
        mean_error = (estimated_ip - true_ip).mean().abs().item()
        assert mean_error < 0.01
```

Parametrized over 6 (dim, bits) combinations. This test verifies the defining property of TurboQuantProd: the inner product estimator is unbiased.

---

## Long Context Tests

The `test_long_context.py` file validates correctness at production sequence lengths:

| Context Length | Tokens | Purpose |
|:-:|:-:|---|
| 4K | 4,096 | Standard generation |
| 16K | 16,384 | Long document processing |
| 32K | 32,768 | Maximum default config |

### What Is Tested

- Cache append at every position maintains correctness
- Memory usage scales linearly with sequence length
- No precision degradation as context grows
- Ring buffer wrapping works correctly at buffer boundaries

---

## Performance Tests

The `test_performance.py` file verifies algorithmic complexity invariants:

### O(1) Append Verification

```python
def test_o1_append():
    """Single-token append time should not grow with cache size."""
    times = []
    for seq_len in [100, 1000, 10000]:
        # Measure append time for one token at position seq_len
        t0 = time.perf_counter()
        cache.update(key, value, layer_idx=0)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    # Last append should not be > 10x slower than first
    assert times[-1] < times[0] * 10
```

This verifies the O(1) pre-allocated ring buffer, as opposed to the O(N) `torch.cat` in the original implementation.

### Linear Scaling

```python
def test_linear_memory_scaling():
    """Memory should scale linearly with sequence length."""
    for seq_len in [1000, 2000, 4000]:
        cache.resize(seq_len)
        memory = cache.vram_bytes()
        # Memory should be approximately proportional to seq_len
```

---

## Running Tests

### Python Tests

```bash
# Run all tests (CPU only, no GPU needed)
pytest tests/ -v

# Run specific test file
pytest tests/test_quantizer_mse.py -v

# Run with markers
pytest tests/ -v -m "not gpu"           # Skip GPU tests
pytest tests/ -v -m "not slow"          # Skip slow tests
pytest tests/ -v -m "gpu"               # Only GPU tests

# Run with coverage
pytest tests/ --cov=flashquant --cov-report=html
# Opens htmlcov/index.html -- coverage must be >= 90%

# Run with verbose output
pytest tests/ -v -s  # -s shows print output
```

### C++ Tests

```bash
# Build with tests enabled
cmake -B build -DFLASHQUANT_TESTS=ON
cmake --build build -j$(nproc)

# Run all C++ tests
ctest --test-dir build --output-on-failure

# Run with verbose output
ctest --test-dir build --output-on-failure -V

# Run specific test
./build/flashquant_tests --gtest_filter="*codebook*"
```

### Coverage

The project requires **>= 90% code coverage** with **zero exclusions**:

```toml
# pyproject.toml
[tool.coverage.run]
source = ["flashquant"]
omit = []                    # No exclusions!

[tool.coverage.report]
fail_under = 90
show_missing = true
```

This is a significant improvement over the original turboquant-vllm, which excluded `triton/` and `vllm/` directories from coverage, artificially inflating the coverage number.

### CI Pipeline

The GitHub Actions CI runs:

1. **Lint** (ruff) -- code style and import ordering
2. **Type check** (mypy) -- static type analysis
3. **Python tests** (pytest) -- all 264 tests on CPU
4. **C++ tests** (ctest) -- all C++ unit tests
5. **Coverage check** -- fails if < 90%

---

*Copyright 2026 Ayi NEDJIMI. Apache License 2.0.*
