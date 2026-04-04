"""Tests for Lloyd-Max codebook computation and quantize/dequantize."""
from __future__ import annotations

import pytest
import torch

from flashquant.core.codebook import CodebookRegistry, _gaussian_centroids_boundaries

from .conftest import _gaussian_centroids_boundaries as ref_centroids_boundaries


# ---------------------------------------------------------------------------
# Parametrized over dim and bits
# ---------------------------------------------------------------------------


@pytest.fixture(params=[16, 64, 128, 256], ids=["d16", "d64", "d128", "d256"])
def cb_dim(request: pytest.FixtureRequest) -> int:
    return request.param


@pytest.fixture(params=[1, 2, 3, 4], ids=["1b", "2b", "3b", "4b"])
def cb_bits(request: pytest.FixtureRequest) -> int:
    return request.param


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGaussianCentroidsMonotone:
    """Centroids must be strictly increasing."""

    def test_gaussian_centroids_monotone(self, cb_dim: int, cb_bits: int) -> None:
        centroids, _ = _gaussian_centroids_boundaries(cb_dim, cb_bits)
        n_levels = 1 << cb_bits
        assert centroids.shape == (n_levels,)
        for i in range(n_levels - 1):
            assert centroids[i].item() < centroids[i + 1].item(), (
                f"Centroids not strictly increasing at index {i}: "
                f"{centroids[i].item()} >= {centroids[i + 1].item()}"
            )


class TestGaussianBoundariesBetweenCentroids:
    """Each boundary must lie between its adjacent centroids."""

    def test_gaussian_boundaries_between_centroids(
        self, cb_dim: int, cb_bits: int
    ) -> None:
        centroids, boundaries = _gaussian_centroids_boundaries(cb_dim, cb_bits)
        n_levels = 1 << cb_bits
        assert boundaries.shape == (n_levels - 1,)
        for i in range(n_levels - 1):
            assert centroids[i].item() < boundaries[i].item() < centroids[i + 1].item(), (
                f"Boundary {i} = {boundaries[i].item()} not between "
                f"centroids {centroids[i].item()} and {centroids[i + 1].item()}"
            )


class TestGaussianSymmetry:
    """For even n_levels, centroids should be symmetric around 0."""

    @pytest.mark.parametrize("dim", [64, 128, 256])
    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_gaussian_symmetry(self, dim: int, bits: int) -> None:
        centroids, _ = _gaussian_centroids_boundaries(dim, bits)
        n_levels = 1 << bits
        if n_levels % 2 != 0:
            pytest.skip("Symmetry test only applies to even n_levels")
        for i in range(n_levels // 2):
            c_low = centroids[i].item()
            c_high = centroids[n_levels - 1 - i].item()
            assert abs(c_low + c_high) < 1e-5, (
                f"Centroid pair ({i}, {n_levels - 1 - i}) not symmetric: "
                f"{c_low} + {c_high} = {c_low + c_high}"
            )


class TestQuantizeDequantizeRoundtrip:
    """Quantize then dequantize gives values close to centroids."""

    @pytest.mark.parametrize("dim", [64, 128])
    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_quantize_dequantize_roundtrip(self, dim: int, bits: int) -> None:
        registry = CodebookRegistry()
        cb = registry.get(dim, bits)
        # Generate values that cover the centroid range
        x = torch.randn(100, dim) / (dim ** 0.5)
        indices = cb.quantize(x)
        reconstructed = cb.dequantize(indices)
        # Each reconstructed value should equal its centroid
        for i in range(cb.centroids.shape[0]):
            mask = indices == i
            if mask.any():
                vals = reconstructed[mask]
                expected = cb.centroids[i].item()
                assert torch.allclose(
                    vals, torch.full_like(vals, expected), atol=1e-6
                )


class TestBitsRange:
    """bits=1,2,3,4 should all work without errors."""

    @pytest.mark.parametrize("bits", [1, 2, 3, 4])
    def test_bits_range(self, bits: int) -> None:
        centroids, boundaries = _gaussian_centroids_boundaries(128, bits)
        assert centroids.shape[0] == (1 << bits)
        assert boundaries.shape[0] == (1 << bits) - 1


class TestInvalidBits:
    """bits=0 produces a degenerate codebook (1 level, no boundaries).

    The pure-Python solver does not validate bits; it produces whatever
    2^bits levels are requested. bits=0 gives 1 centroid and 0 boundaries.
    bits=9 gives 512 centroids (expensive but valid).
    """

    def test_bits_zero_degenerate(self) -> None:
        centroids, boundaries = _gaussian_centroids_boundaries(128, 0)
        assert centroids.shape[0] == 1  # 2^0 = 1 level
        assert boundaries.shape[0] == 0  # no partition boundaries

    def test_bits_nine_large(self) -> None:
        # bits=9 -> 512 centroids: valid but slow
        centroids, boundaries = _gaussian_centroids_boundaries(128, 9)
        assert centroids.shape[0] == 512
        assert boundaries.shape[0] == 511


class TestInvalidDim:
    """dim=0 should raise an error (ZeroDivisionError from 1/sqrt(0))."""

    def test_invalid_dim_zero(self) -> None:
        with pytest.raises((ValueError, ZeroDivisionError, RuntimeError)):
            centroids, _ = _gaussian_centroids_boundaries(0, 4)
            # If it somehow doesn't raise, check for NaN
            assert torch.isfinite(centroids).all()


class TestCodebookRegistry:
    """Validate the registry caching behavior."""

    def test_get_returns_codebook(self) -> None:
        registry = CodebookRegistry()
        cb = registry.get(128, 4)
        assert cb.centroids.shape == (16,)
        assert cb.boundaries.shape == (15,)
        assert cb.bits == 4
        assert cb.dim == 128

    def test_get_cached(self) -> None:
        registry = CodebookRegistry()
        cb1 = registry.get(128, 4)
        cb2 = registry.get(128, 4)
        assert cb1 is cb2

    def test_clear(self) -> None:
        registry = CodebookRegistry()
        registry.get(128, 4)
        registry.clear()
        # After clear, next call should recompute
        cb = registry.get(128, 4)
        assert cb.centroids.shape == (16,)
