"""Tests for Haar-distributed random orthogonal rotation matrices."""
from __future__ import annotations

import pytest
import torch

from .conftest import generate_rotation_matrix, split_rotation


class TestOrthogonality:
    """R^T @ R should approximate identity."""

    @pytest.mark.parametrize("dim", [64, 128, 256])
    def test_orthogonality(self, dim: int) -> None:
        R = generate_rotation_matrix(dim, seed=42)
        product = R.T @ R
        identity = torch.eye(dim, dtype=torch.float32)
        assert torch.allclose(product, identity, atol=1e-5), (
            f"R^T @ R deviates from I: max error = "
            f"{(product - identity).abs().max().item():.2e}"
        )


class TestDeterminism:
    """Same seed should produce the same matrix."""

    @pytest.mark.parametrize("dim", [64, 128, 256])
    def test_determinism(self, dim: int) -> None:
        R1 = generate_rotation_matrix(dim, seed=123)
        R2 = generate_rotation_matrix(dim, seed=123)
        assert torch.allclose(R1, R2, atol=0.0), (
            "Same seed produced different rotation matrices"
        )


class TestDifferentSeeds:
    """Different seeds should produce different matrices."""

    @pytest.mark.parametrize("dim", [64, 128, 256])
    def test_different_seeds(self, dim: int) -> None:
        R1 = generate_rotation_matrix(dim, seed=42)
        R2 = generate_rotation_matrix(dim, seed=99)
        assert not torch.allclose(R1, R2, atol=1e-3), (
            "Different seeds produced the same rotation matrix"
        )


class TestDimensions:
    """Various dimensions should work."""

    @pytest.mark.parametrize("dim", [64, 128, 256])
    def test_dimensions(self, dim: int) -> None:
        R = generate_rotation_matrix(dim)
        assert R.shape == (dim, dim)
        assert R.dtype == torch.float32


class TestSquare:
    """Output should be (dim, dim)."""

    @pytest.mark.parametrize("dim", [32, 64, 128, 256])
    def test_square(self, dim: int) -> None:
        R = generate_rotation_matrix(dim)
        assert R.shape[0] == R.shape[1] == dim


class TestSplitRotationShapes:
    """Even/odd halves should have correct shapes."""

    @pytest.mark.parametrize("dim", [64, 128, 256])
    def test_split_rotation_shapes(self, dim: int) -> None:
        R = generate_rotation_matrix(dim)
        even, odd = split_rotation(R)
        half = dim // 2
        assert even.shape == (dim, half), f"Even half shape: {even.shape}"
        assert odd.shape == (dim, half), f"Odd half shape: {odd.shape}"


class TestSplitRotationReconstructs:
    """Interleaving even/odd halves should reconstruct rotation.T."""

    @pytest.mark.parametrize("dim", [64, 128, 256])
    def test_split_rotation_reconstructs(self, dim: int) -> None:
        R = generate_rotation_matrix(dim)
        even, odd = split_rotation(R)
        half = dim // 2

        # Reconstruct rotation.T from even/odd halves
        reconstructed = torch.empty(dim, dim, dtype=torch.float32)
        reconstructed[:, 0::2] = even
        reconstructed[:, 1::2] = odd

        expected = R.T.contiguous()
        assert torch.allclose(reconstructed, expected, atol=1e-6), (
            f"Reconstructed rotation.T differs: max error = "
            f"{(reconstructed - expected).abs().max().item():.2e}"
        )
