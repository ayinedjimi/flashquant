"""Tests for TurboQuantMSE (Stage 1: rotation + Lloyd-Max scalar quantization)."""
from __future__ import annotations

import math

import pytest
import torch

from .conftest import RefQuantizerMSE, cosine_similarity_flat


# ---------------------------------------------------------------------------
# Parametrized fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(params=[64, 128, 256], ids=["d64", "d128", "d256"])
def mse_dim(request: pytest.FixtureRequest) -> int:
    return request.param


@pytest.fixture(params=[2, 3, 4], ids=["2b", "3b", "4b"])
def mse_bits(request: pytest.FixtureRequest) -> int:
    return request.param


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRoundtripCosineSimilarity:
    """Quantize/dequantize roundtrip should preserve cosine similarity."""

    # Per (dim, bits) thresholds account for dimension-dependent quantization error.
    # Higher dimensions spread more error across coordinates.
    THRESHOLDS = {
        (64, 4): 0.95, (64, 3): 0.92, (64, 2): 0.80,
        (128, 4): 0.93, (128, 3): 0.90, (128, 2): 0.78,
        (256, 4): 0.85, (256, 3): 0.82, (256, 2): 0.75,
    }

    def test_roundtrip_cosine_similarity(
        self, mse_dim: int, mse_bits: int
    ) -> None:
        q = RefQuantizerMSE(mse_dim, mse_bits)
        x = torch.randn(200, mse_dim)
        indices, norms = q.quantize(x)
        reconstructed = q.dequantize(indices, norms)

        cos = cosine_similarity_flat(x, reconstructed)
        threshold = self.THRESHOLDS[(mse_dim, mse_bits)]
        assert cos >= threshold, (
            f"Cosine similarity {cos:.4f} < {threshold} "
            f"for dim={mse_dim}, bits={mse_bits}"
        )


class TestMSEBound:
    """Empirical MSE should be within the theoretical bound from Theorem 1.

    Theorem 1 bound: MSE <= sqrt(3) * pi/2 / 4^b
    We use a 5x safety factor to account for finite-sample effects
    and the Gaussian approximation.
    """

    def test_mse_bound(self, mse_dim: int, mse_bits: int) -> None:
        q = RefQuantizerMSE(mse_dim, mse_bits)
        x = torch.randn(500, mse_dim)
        # Normalize to unit sphere for the bound to apply
        x = x / torch.norm(x, dim=-1, keepdim=True)

        indices, norms = q.quantize(x)
        reconstructed = q.dequantize(indices, norms)

        mse = ((x - reconstructed) ** 2).mean().item()
        theoretical_bound = math.sqrt(3) * math.pi / 2 * (1.0 / 4 ** mse_bits)
        assert mse < theoretical_bound * 5, (
            f"MSE {mse:.6f} exceeds 5x theoretical bound {theoretical_bound:.6f}"
        )


class TestNormsPreserved:
    """Output norms should match input norms."""

    def test_norms_preserved(self, mse_dim: int, mse_bits: int) -> None:
        q = RefQuantizerMSE(mse_dim, mse_bits)
        x = torch.randn(100, mse_dim)
        original_norms = torch.norm(x, dim=-1)

        _, stored_norms = q.quantize(x)
        stored_norms = stored_norms.squeeze(-1)

        rel_error = (
            (original_norms - stored_norms).abs() / (original_norms + 1e-10)
        ).mean()
        assert rel_error < 0.01, (
            f"Norm relative error {rel_error:.4f} too high"
        )


class TestIndicesRange:
    """All indices should be in [0, 2^bits - 1]."""

    def test_indices_range(self, mse_dim: int, mse_bits: int) -> None:
        q = RefQuantizerMSE(mse_dim, mse_bits)
        x = torch.randn(200, mse_dim)
        indices, _ = q.quantize(x)

        max_idx = (1 << mse_bits) - 1
        assert indices.min().item() >= 0, (
            f"Found negative index: {indices.min().item()}"
        )
        assert indices.max().item() <= max_idx, (
            f"Index {indices.max().item()} exceeds max {max_idx}"
        )


class TestBatchDimensions:
    """Should work with (B, H, S, D) shaped inputs."""

    @pytest.mark.parametrize("shape", [
        (2, 4, 8, 64),
        (1, 1, 16, 128),
        (2, 8, 4, 256),
    ])
    def test_batch_dimensions(self, shape: tuple[int, ...]) -> None:
        B, H, S, D = shape
        q = RefQuantizerMSE(D, 4)
        x = torch.randn(*shape)
        indices, norms = q.quantize(x)
        assert indices.shape == shape, f"Indices shape: {indices.shape}"
        assert norms.shape == (B, H, S, 1), f"Norms shape: {norms.shape}"

        reconstructed = q.dequantize(indices, norms)
        assert reconstructed.shape == shape


class TestDeterministic:
    """Same input + same seed should give same output."""

    def test_deterministic(self, mse_dim: int, mse_bits: int) -> None:
        x = torch.randn(50, mse_dim)

        q1 = RefQuantizerMSE(mse_dim, mse_bits, seed=42)
        idx1, n1 = q1.quantize(x)

        q2 = RefQuantizerMSE(mse_dim, mse_bits, seed=42)
        idx2, n2 = q2.quantize(x)

        assert torch.equal(idx1, idx2)
        assert torch.equal(n1, n2)
