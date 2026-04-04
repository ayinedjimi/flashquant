"""Tests for key/value compressor wrappers."""
from __future__ import annotations

import pytest
import torch

from .conftest import RefQuantizerMSE, cosine_similarity_flat, ref_nibble_pack


# ---------------------------------------------------------------------------
# Simple compressor wrappers for testing
# ---------------------------------------------------------------------------


class RefValueCompressor:
    """Reference value compressor using MSE quantization."""

    def __init__(self, head_dim: int, bits: int = 3, seed: int = 43) -> None:
        self.head_dim = head_dim
        self.bits = bits
        self.quantizer = RefQuantizerMSE(head_dim, bits, seed=seed)

    def compress(self, values: torch.Tensor) -> dict:
        indices, norms = self.quantizer.quantize(values.float())
        return {
            "indices": indices,
            "norms": norms,
            "original_dtype": values.dtype,
        }

    def decompress(self, compressed: dict) -> torch.Tensor:
        result = self.quantizer.dequantize(compressed["indices"], compressed["norms"])
        return result.to(compressed["original_dtype"])


class RefKeyCompressor:
    """Reference key compressor using MSE (drop-in mode)."""

    def __init__(self, head_dim: int, bits: int = 3, seed: int = 42) -> None:
        self.head_dim = head_dim
        self.bits = bits
        self.quantizer = RefQuantizerMSE(head_dim, bits, seed=seed)

    def compress(self, keys: torch.Tensor) -> dict:
        indices, norms = self.quantizer.quantize(keys.float())
        return {
            "indices": indices,
            "norms": norms,
            "original_dtype": keys.dtype,
        }

    def decompress(self, compressed: dict) -> torch.Tensor:
        result = self.quantizer.dequantize(compressed["indices"], compressed["norms"])
        return result.to(compressed["original_dtype"])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestKeyValueDifferentCodebooks:
    """Key and value compressors should use different seeds."""

    def test_key_value_different_codebooks(self) -> None:
        key_comp = RefKeyCompressor(128, bits=4, seed=42)
        val_comp = RefValueCompressor(128, bits=4, seed=43)

        # Different seeds should produce different rotation matrices
        assert not torch.allclose(
            key_comp.quantizer.rotation,
            val_comp.quantizer.rotation,
            atol=1e-3,
        ), "Key and value compressors should use different rotation matrices"


class TestCompressDecompressRoundtrip:
    """Compress then decompress should have good cosine similarity."""

    @pytest.mark.parametrize("bits", [3, 4])
    @pytest.mark.parametrize("dim", [64, 128])
    def test_compress_decompress_roundtrip(
        self, bits: int, dim: int
    ) -> None:
        comp = RefValueCompressor(dim, bits=bits)
        x = torch.randn(2, 4, 16, dim, dtype=torch.float32)
        compressed = comp.compress(x)
        reconstructed = comp.decompress(compressed)

        cos = cosine_similarity_flat(x, reconstructed)
        threshold = {4: 0.95, 3: 0.92}[bits]
        assert cos >= threshold, (
            f"Cosine similarity {cos:.4f} < {threshold}"
        )


class TestCompressedShapes:
    """Packed indices should be half the dim for 4-bit."""

    def test_compressed_shapes(self) -> None:
        dim = 128
        comp = RefValueCompressor(dim, bits=4)
        x = torch.randn(2, 4, 8, dim)
        compressed = comp.compress(x)

        # Indices shape should match input
        assert compressed["indices"].shape == (2, 4, 8, dim)
        assert compressed["norms"].shape == (2, 4, 8, 1)

        # After nibble packing
        packed = ref_nibble_pack(compressed["indices"].clamp(0, 15).long())
        assert packed.shape == (2, 4, 8, dim // 2)


class TestNormsDtype:
    """Norms should always be float32."""

    @pytest.mark.parametrize("input_dtype", [torch.float16, torch.float32])
    def test_norms_dtype(self, input_dtype: torch.dtype) -> None:
        comp = RefValueCompressor(128, bits=4)
        x = torch.randn(1, 2, 4, 128).to(input_dtype)
        compressed = comp.compress(x)
        assert compressed["norms"].dtype == torch.float32
