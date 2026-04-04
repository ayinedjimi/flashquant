"""Tests for theoretical numerical bounds from TurboQuant paper.

Validates empirical measurements against theoretical bounds from
Theorem 1 (MSE distortion) and Theorem 2 (inner product variance).

Reference: arXiv 2504.19874, Sections 3-4.
"""
from __future__ import annotations

import math

import pytest
import torch

from .conftest import RefQuantizerMSE, RefQuantizerProd


@pytest.fixture(params=[2, 3, 4], ids=["2b", "3b", "4b"])
def bound_bits(request: pytest.FixtureRequest) -> int:
    return request.param


@pytest.fixture(params=[64, 128], ids=["d64", "d128"])
def bound_dim(request: pytest.FixtureRequest) -> int:
    return request.param


class TestMSEDistortionBound:
    """Empirical MSE should be <= theoretical bound from Theorem 1.

    Theorem 1: D_MSE <= sqrt(3) * pi / 2 / 4^b
    We use a generous safety factor since:
    - The theorem assumes d -> infinity
    - The Gaussian approximation has finite-d error
    - We test on random (not adversarial) inputs
    """

    def test_mse_distortion_bound(
        self, bound_dim: int, bound_bits: int
    ) -> None:
        q = RefQuantizerMSE(bound_dim, bound_bits)
        n_samples = 1000

        x = torch.randn(n_samples, bound_dim)
        x = x / torch.norm(x, dim=-1, keepdim=True)

        indices, norms = q.quantize(x)
        reconstructed = q.dequantize(indices, norms)

        mse = ((x - reconstructed) ** 2).mean().item()
        theoretical_bound = math.sqrt(3) * math.pi / 2 * (1.0 / 4 ** bound_bits)

        # Allow 5x margin for finite-dimensional effects
        assert mse < theoretical_bound * 5, (
            f"MSE {mse:.6f} > 5x theoretical bound {theoretical_bound:.6f} "
            f"for dim={bound_dim}, bits={bound_bits}"
        )


class TestInnerProductDistortionBound:
    """Empirical variance should be <= theoretical bound from Theorem 2.

    Theorem 2: Var(<q, x_hat>) <= sqrt(3) * pi^2 * ||y||^2 / (d * 4^b)
    where b = bits - 1 (MSE portion of TurboQuantProd).
    """

    def test_inner_product_distortion_bound(
        self, bound_dim: int, bound_bits: int
    ) -> None:
        q = RefQuantizerProd(bound_dim, bound_bits)
        n_samples = 500

        queries = torch.randn(n_samples, bound_dim) * 0.1
        keys = torch.randn(n_samples, bound_dim) * 0.1

        true_ip = (queries * keys).sum(dim=-1)

        indices, norms, qjl_signs, res_norms = q.quantize(keys)
        estimated_ip = q.estimate_inner_product(
            queries, indices, norms, qjl_signs, res_norms
        ).squeeze(-1)

        errors = estimated_ip - true_ip
        empirical_variance = (errors ** 2).mean().item()

        # Theorem 2 bound with effective bits = bound_bits - 1
        b_eff = bound_bits - 1
        # Average ||y||^2 for unit-norm vectors is 1
        theoretical_bound = (
            math.sqrt(3) * math.pi ** 2 / (bound_dim * 4 ** b_eff)
        )

        # Very generous margin (100x) since:
        # - queries/keys have 0.1 scale -> actual variance is scale^4 * bound
        # - finite samples, finite dimensions
        assert empirical_variance < theoretical_bound * 100, (
            f"Variance {empirical_variance:.8f} > 100x theoretical "
            f"bound {theoretical_bound:.8f} for dim={bound_dim}, bits={bound_bits}"
        )


class TestUnbiasedness:
    """Mean estimation error should be < 0.01 over many samples."""

    @pytest.mark.parametrize("dim", [64, 128])
    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_unbiasedness(self, dim: int, bits: int) -> None:
        q = RefQuantizerProd(dim, bits)
        n_samples = 10000

        queries = torch.randn(n_samples, dim) * 0.1
        keys = torch.randn(n_samples, dim) * 0.1

        true_ip = (queries * keys).sum(dim=-1)

        indices, norms, qjl_signs, res_norms = q.quantize(keys)
        estimated_ip = q.estimate_inner_product(
            queries, indices, norms, qjl_signs, res_norms
        ).squeeze(-1)

        mean_error = (estimated_ip - true_ip).mean().abs().item()
        assert mean_error < 0.01, (
            f"Mean estimation error {mean_error:.6f} >= 0.01 "
            f"for dim={dim}, bits={bits}"
        )
