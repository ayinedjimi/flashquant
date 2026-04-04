"""Tests for TurboQuantProd (Stage 2: MSE + QJL inner product correction)."""
from __future__ import annotations

import math

import pytest
import torch

from .conftest import RefQuantizerProd


# ---------------------------------------------------------------------------
# Parametrized fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(params=[64, 128], ids=["d64", "d128"])
def prod_dim(request: pytest.FixtureRequest) -> int:
    return request.param


@pytest.fixture(params=[2, 3, 4], ids=["2b", "3b", "4b"])
def prod_bits(request: pytest.FixtureRequest) -> int:
    return request.param


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestUnbiasedInnerProduct:
    """E[<q, x_hat>] should approximate <q, x> (mean over many samples)."""

    def test_unbiased_inner_product(
        self, prod_dim: int, prod_bits: int
    ) -> None:
        q = RefQuantizerProd(prod_dim, prod_bits)
        n_samples = 1000

        queries = torch.randn(n_samples, prod_dim) * 0.1
        keys = torch.randn(n_samples, prod_dim) * 0.1

        true_ip = (queries * keys).sum(dim=-1)

        indices, norms, qjl_signs, res_norms = q.quantize(keys)
        estimated_ip = q.estimate_inner_product(
            queries, indices, norms, qjl_signs, res_norms
        ).squeeze(-1)

        bias = (estimated_ip - true_ip).mean().item()
        signal = true_ip.abs().mean().item()
        relative_bias = abs(bias) / (signal + 1e-10)

        assert relative_bias < 0.05, (
            f"Relative bias {relative_bias:.4f} >= 0.05 "
            f"(bias={bias:.6f}, signal={signal:.6f})"
        )


class TestVarianceBound:
    """Var(<q, x_hat>) should be bounded by Theorem 2.

    Theorem 2 bound: Var <= sqrt(3) * pi^2 * ||y||^2 / (d * 4^b)
    where y is the normalized key after rotation.
    """

    def test_variance_bound(self, prod_dim: int, prod_bits: int) -> None:
        q = RefQuantizerProd(prod_dim, prod_bits)
        n_samples = 500

        queries = torch.randn(n_samples, prod_dim) * 0.1
        keys = torch.randn(n_samples, prod_dim) * 0.1

        true_ip = (queries * keys).sum(dim=-1)

        indices, norms, qjl_signs, res_norms = q.quantize(keys)
        estimated_ip = q.estimate_inner_product(
            queries, indices, norms, qjl_signs, res_norms
        ).squeeze(-1)

        empirical_variance = ((estimated_ip - true_ip) ** 2).mean().item()

        # Compute expected y norm: after rotation of unit-norm vector,
        # ||y|| = 1 on average
        avg_y_norm_sq = 1.0  # unit sphere normalization

        # Use bits - 1 for the MSE part (Prod uses bits-1 for MSE)
        b_eff = prod_bits - 1
        theoretical_bound = (
            math.sqrt(3) * math.pi ** 2 * avg_y_norm_sq
            / (prod_dim * 4 ** b_eff)
        )
        # 50x safety margin since these are small-scale estimates
        # and the keys have non-trivial norms (0.1 scale)
        assert empirical_variance < theoretical_bound * 50, (
            f"Variance {empirical_variance:.6f} exceeds 50x theoretical "
            f"bound {theoretical_bound:.6f}"
        )


class TestQJLSignsAreInt8:
    """QJL signs should be stored as int8 (or float with +/-1 values)."""

    def test_qjl_signs_values(self, prod_dim: int, prod_bits: int) -> None:
        q = RefQuantizerProd(prod_dim, prod_bits)
        x = torch.randn(20, prod_dim)
        _, _, qjl_signs, _ = q.quantize(x)

        # Signs should be +/-1 only
        unique_vals = qjl_signs.unique()
        for v in unique_vals:
            assert v.item() in (-1.0, 1.0), (
                f"QJL sign value {v.item()} not in {{-1, +1}}"
            )


class TestResidualNormsPositive:
    """All residual norms should be >= 0."""

    def test_residual_norms_positive(
        self, prod_dim: int, prod_bits: int
    ) -> None:
        q = RefQuantizerProd(prod_dim, prod_bits)
        x = torch.randn(50, prod_dim)
        _, _, _, res_norms = q.quantize(x)

        assert (res_norms >= 0).all(), (
            f"Found negative residual norms: min={res_norms.min().item()}"
        )


class TestBitsAtLeast2:
    """bits=1 should raise ValueError."""

    def test_bits_at_least_2(self) -> None:
        with pytest.raises(ValueError, match="bits >= 2"):
            RefQuantizerProd(64, bits=1)
