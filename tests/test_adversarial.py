"""Adversarial input tests for the quantization pipeline.

Tests edge cases that could cause NaN, overflow, or silent corruption.
"""
from __future__ import annotations

import pytest
import torch

from .conftest import RefQuantizerMSE, cosine_similarity_flat


class TestAllSameVectors:
    """x = ones * 1e6 -- all coordinates identical, large magnitude."""

    def test_all_same_vectors(self) -> None:
        dim = 128
        q = RefQuantizerMSE(dim, 4, seed=42)
        x = torch.ones(10, dim) * 1e6

        indices, norms = q.quantize(x)
        reconstructed = q.dequantize(indices, norms)

        # Should not produce NaN
        assert torch.isfinite(reconstructed).all(), (
            "NaN or Inf in reconstruction of all-same vectors"
        )

        # Norms should be large
        expected_norm = 1e6 * (dim ** 0.5)
        actual_norms = norms.squeeze(-1)
        rel_error = ((actual_norms - expected_norm).abs() / expected_norm).mean()
        assert rel_error < 0.01


class TestSparseVectors:
    """Only 1 nonzero coordinate."""

    def test_sparse_vectors(self) -> None:
        dim = 128
        q = RefQuantizerMSE(dim, 4, seed=42)

        x = torch.zeros(10, dim)
        for i in range(10):
            x[i, i % dim] = 1.0

        indices, norms = q.quantize(x)
        reconstructed = q.dequantize(indices, norms)

        assert torch.isfinite(reconstructed).all()
        # Norms should be 1.0
        assert torch.allclose(norms.squeeze(-1), torch.ones(10), atol=1e-5)


class TestExtremeOutliers:
    """One coordinate 1e8, rest 1e-8."""

    def test_extreme_outliers(self) -> None:
        dim = 128
        q = RefQuantizerMSE(dim, 4, seed=42)

        x = torch.full((10, dim), 1e-8)
        x[:, 0] = 1e8

        indices, norms = q.quantize(x)
        reconstructed = q.dequantize(indices, norms)

        assert torch.isfinite(reconstructed).all(), (
            "NaN or Inf in reconstruction of extreme outlier vectors"
        )

        # The dominant coordinate should be approximately preserved
        # After rotation the structure is distributed, so check norm instead
        original_norms = torch.norm(x, dim=-1)
        recon_norms = torch.norm(reconstructed, dim=-1)
        rel_error = ((original_norms - recon_norms).abs() / original_norms).mean()
        assert rel_error < 0.05


class TestZeroVectors:
    """All zeros should not produce NaN."""

    def test_zero_vectors(self) -> None:
        dim = 128
        q = RefQuantizerMSE(dim, 4, seed=42)

        x = torch.zeros(10, dim)
        indices, norms = q.quantize(x)
        reconstructed = q.dequantize(indices, norms)

        assert torch.isfinite(reconstructed).all(), (
            "NaN or Inf in reconstruction of zero vectors"
        )
        # Reconstructed should be near zero (norm * centroid ~ 0)
        assert reconstructed.abs().max().item() < 1e-5


class TestCorrelatedVectors:
    """x_i = x_0 + small_noise -- nearly identical vectors."""

    def test_correlated_vectors(self) -> None:
        dim = 128
        q = RefQuantizerMSE(dim, 4, seed=42)

        base = torch.randn(1, dim)
        noise = torch.randn(100, dim) * 1e-4
        x = base + noise

        indices, norms = q.quantize(x)
        reconstructed = q.dequantize(indices, norms)

        assert torch.isfinite(reconstructed).all()

        # All reconstructed vectors should be similar to each other
        cos = cosine_similarity_flat(x, reconstructed)
        assert cos > 0.85, (
            f"Correlated vector reconstruction cosine {cos:.4f} < 0.85"
        )


class TestNegativeValues:
    """Vectors with all negative values should work correctly."""

    def test_negative_values(self) -> None:
        dim = 128
        q = RefQuantizerMSE(dim, 4, seed=42)

        x = -torch.abs(torch.randn(50, dim))

        indices, norms = q.quantize(x)
        reconstructed = q.dequantize(indices, norms)

        assert torch.isfinite(reconstructed).all()
        cos = cosine_similarity_flat(x, reconstructed)
        assert cos > 0.90


class TestVerySmallValues:
    """Values near machine epsilon should not cause division by zero."""

    def test_very_small_values(self) -> None:
        dim = 128
        q = RefQuantizerMSE(dim, 4, seed=42)

        x = torch.randn(20, dim) * 1e-30

        indices, norms = q.quantize(x)
        reconstructed = q.dequantize(indices, norms)

        assert torch.isfinite(reconstructed).all(), (
            "NaN or Inf in reconstruction of very small vectors"
        )
