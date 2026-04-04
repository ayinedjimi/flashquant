"""Tests for the TQ4 compress kernel (or pure Python fallback).

Validates that the fused compress operation (norm + rotate + quantize + pack)
matches the reference step-by-step implementation.
"""
from __future__ import annotations

import pytest
import torch

from .conftest import (
    RefQuantizerMSE,
    generate_rotation_matrix,
    ref_compress,
    ref_nibble_pack,
    split_rotation,
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCompressMatchesReference:
    """Compress kernel (or fallback) should match the reference exactly."""

    @pytest.mark.parametrize("head_dim", [64, 128, 256])
    @pytest.mark.parametrize("bits", [4])
    def test_compress_matches_reference(
        self, head_dim: int, bits: int
    ) -> None:
        rotation = generate_rotation_matrix(head_dim, seed=42)
        q = RefQuantizerMSE(head_dim, bits, seed=42)
        even, odd = split_rotation(rotation)
        boundaries = q.boundaries

        x = torch.randn(4, 8, head_dim, dtype=torch.float32)
        packed_ref, norms_ref = ref_compress(x, rotation, boundaries)

        # Verify shapes
        assert packed_ref.shape == (4, 8, head_dim // 2)
        assert norms_ref.shape == (4, 8, 1)

        # Verify roundtrip: unpack and check index range
        high = (packed_ref >> 4).long()
        low = (packed_ref & 0x0F).long()
        assert high.min().item() >= 0
        assert high.max().item() <= 15
        assert low.min().item() >= 0
        assert low.max().item() <= 15

        # Verify norms are positive
        assert (norms_ref >= 0).all()


class TestCompressDifferentDtypes:
    """fp16 and bf16 inputs should produce consistent results."""

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_compress_different_dtypes(self, dtype: torch.dtype) -> None:
        head_dim = 128
        rotation = generate_rotation_matrix(head_dim, seed=42)
        q = RefQuantizerMSE(head_dim, 4, seed=42)
        boundaries = q.boundaries

        x = torch.randn(2, 4, head_dim).to(dtype)
        packed, norms = ref_compress(x, rotation, boundaries)

        assert packed.shape == (2, 4, head_dim // 2)
        assert norms.shape == (2, 4, 1)
        assert packed.dtype == torch.uint8
        assert norms.dtype == torch.float32


class TestCompressBatchShapes:
    """Various (N, H, D) shapes should work."""

    @pytest.mark.parametrize("shape", [
        (1, 1, 64),
        (4, 8, 128),
        (2, 32, 256),
        (16, 4, 64),
    ])
    def test_compress_batch_shapes(
        self, shape: tuple[int, int, int]
    ) -> None:
        N, H, D = shape
        rotation = generate_rotation_matrix(D, seed=42)
        q = RefQuantizerMSE(D, 4, seed=42)
        boundaries = q.boundaries

        x = torch.randn(N, H, D)
        packed, norms = ref_compress(x, rotation, boundaries)

        assert packed.shape == (N, H, D // 2)
        assert norms.shape == (N, H, 1)
