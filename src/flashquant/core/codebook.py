"""Lloyd-Max codebook registry with device-aware caching.

Provides :class:`CodebookRegistry`, a singleton-style cache that
computes Lloyd-Max optimal codebooks once and moves them to the
requested device on demand. Avoids recomputing codebooks across layers
(e.g., 32 layers x 2 K/V = 64 lookups pay the cost only once).

The codebook computation tries the C++ extension ``_C.codebook`` first,
falling back to a pure Python implementation using
``torch.special.erfinv`` for the Gaussian CDF inverse.

The pure Python path avoids the scipy dependency that the original
turboquant-vllm ``lloyd_max.py`` requires. The Gaussian approximation
(accurate for dim >= 64) computes optimal Lloyd-Max centroids and
boundaries analytically via erfinv.

Reference: Section 3.1 of arXiv 2504.19874.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Tuple

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try C++ extension first
# ---------------------------------------------------------------------------

_USE_C_CODEBOOK = False
try:
    from flashquant import _C  # type: ignore[attr-defined]

    if hasattr(_C, "codebook"):
        _USE_C_CODEBOOK = True
        logger.debug("Using _C.codebook backend for Lloyd-Max computation")
except ImportError:
    pass

if not _USE_C_CODEBOOK:
    logger.debug(
        "C++ codebook extension not available; using pure Python "
        "(torch.special.erfinv) fallback"
    )


# ---------------------------------------------------------------------------
# Pure Python Lloyd-Max solver (no scipy)
# ---------------------------------------------------------------------------


def _gaussian_centroids_boundaries(
    dim: int, bits: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Lloyd-Max centroids and boundaries for the Gaussian approximation.

    After random orthogonal rotation, each coordinate of a d-dimensional
    unit vector follows approximately N(0, 1/d) for d >= 64. The optimal
    Lloyd-Max quantizer for a Gaussian source can be computed via the
    inverse CDF (erfinv).

    This is an iterative Lloyd-Max solver that uses torch operations
    for the Gaussian PDF/CDF, avoiding the scipy dependency.

    Args:
        dim: Vector dimension (controls the variance 1/d).
        bits: Number of quantization bits (produces 2^bits centroids).

    Returns:
        Tuple of (centroids, boundaries) as 1-D float32 tensors.
        Centroids has length 2^bits, boundaries has length 2^bits - 1.
    """
    n_levels = 1 << bits
    sigma = 1.0 / math.sqrt(dim)
    lo, hi = -3.5 * sigma, 3.5 * sigma

    # Initialize centroids uniformly in the support
    centroids = torch.tensor(
        [lo + (hi - lo) * (i + 0.5) / n_levels for i in range(n_levels)],
        dtype=torch.float64,
    )

    # Precompute Gaussian PDF and CDF helpers
    inv_sigma = 1.0 / sigma
    inv_sigma_sq2 = inv_sigma / math.sqrt(2.0)
    norm_const = 1.0 / (sigma * math.sqrt(2.0 * math.pi))

    def _gaussian_pdf(x: torch.Tensor) -> torch.Tensor:
        return norm_const * torch.exp(-0.5 * (x * inv_sigma) ** 2)

    def _gaussian_cdf(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * (1.0 + torch.erf(x * inv_sigma_sq2))

    def _conditional_expectation(
        a: torch.Tensor, b: torch.Tensor
    ) -> torch.Tensor:
        """E[X | a <= X <= b] for X ~ N(0, sigma^2).

        Uses the identity: E[X | a<=X<=b] = sigma^2 * (pdf(a) - pdf(b)) / (cdf(b) - cdf(a))
        """
        pdf_a = _gaussian_pdf(a)
        pdf_b = _gaussian_pdf(b)
        cdf_a = _gaussian_cdf(a)
        cdf_b = _gaussian_cdf(b)
        denom = cdf_b - cdf_a
        # Avoid division by zero for empty partitions
        safe_denom = torch.where(denom > 1e-15, denom, torch.ones_like(denom))
        result = (sigma**2) * (pdf_a - pdf_b) / safe_denom
        # For empty partitions, use midpoint
        return torch.where(denom > 1e-15, result, (a + b) / 2.0)

    # Iterate Lloyd-Max
    max_iter = 200
    tol = 1e-10
    for _ in range(max_iter):
        # Boundaries = midpoints between adjacent centroids
        boundaries = (centroids[:-1] + centroids[1:]) / 2.0

        # Edges including -inf and +inf (approximated by lo/hi)
        lo_t = torch.tensor([lo], dtype=torch.float64)
        hi_t = torch.tensor([hi], dtype=torch.float64)
        edges = torch.cat([lo_t, boundaries, hi_t])

        # Update centroids as conditional expectations
        new_centroids = _conditional_expectation(edges[:-1], edges[1:])

        # Check convergence
        max_shift = (new_centroids - centroids).abs().max().item()
        centroids = new_centroids
        if max_shift < tol:
            break

    boundaries = (centroids[:-1] + centroids[1:]) / 2.0
    return centroids.float(), boundaries.float()


@lru_cache(maxsize=32)
def _solve_codebook_cached(
    dim: int, bits: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cached codebook solver. Returns (centroids, boundaries) on CPU."""
    if _USE_C_CODEBOOK:
        return _C.codebook.solve_lloyd_max(dim, bits)  # type: ignore[attr-defined]
    return _gaussian_centroids_boundaries(dim, bits)


# ---------------------------------------------------------------------------
# Codebook dataclass
# ---------------------------------------------------------------------------


@dataclass
class Codebook:
    """A precomputed Lloyd-Max codebook for a given dimension and bit-width.

    Attributes:
        centroids: Reconstruction values, shape ``(2^bits,)``.
        boundaries: Partition boundaries, shape ``(2^bits - 1,)``.
        bits: Number of quantization bits.
        dim: Vector dimension used to compute the codebook.
    """

    centroids: torch.Tensor
    boundaries: torch.Tensor
    bits: int
    dim: int

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Map continuous values to nearest centroid indices.

        Args:
            x: Input tensor of any shape.

        Returns:
            Long tensor of same shape with centroid indices in
            [0, 2^bits - 1].
        """
        bounds = self.boundaries.to(x.device)
        return torch.bucketize(x, bounds)

    def dequantize(self, indices: torch.Tensor) -> torch.Tensor:
        """Reconstruct continuous values from centroid indices.

        Args:
            indices: Integer tensor of centroid indices.

        Returns:
            Float tensor of reconstructed values.
        """
        cents = self.centroids.to(indices.device)
        return cents[indices]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class CodebookRegistry:
    """Device-aware cache for Lloyd-Max codebooks.

    Computes codebooks once (on CPU) and caches device-specific copies.
    Thread-safe for read-only access after initial computation (the
    ``lru_cache`` on ``_solve_codebook_cached`` handles locking).

    Examples:
        Get a codebook for dim=128, bits=4 on CUDA::

            registry = CodebookRegistry()
            cb = registry.get(128, 4, device=torch.device("cuda:0"))
            cb.centroids.device  # device(type='cuda', index=0)

        Same codebook, different device (no recompute)::

            cb_cpu = registry.get(128, 4, device=torch.device("cpu"))
    """

    def __init__(self) -> None:
        self._device_cache: Dict[
            Tuple[int, int, torch.device], Codebook
        ] = {}

    def get(
        self,
        dim: int,
        bits: int,
        device: torch.device | None = None,
    ) -> Codebook:
        """Get or create a codebook for the given parameters.

        Args:
            dim: Vector dimension.
            bits: Quantization bits.
            device: Target device. Defaults to CPU.

        Returns:
            A Codebook with tensors on the requested device.
        """
        if device is None:
            device = torch.device("cpu")

        key = (dim, bits, device)
        if key in self._device_cache:
            return self._device_cache[key]

        # Get CPU codebook (cached computation)
        centroids_cpu, boundaries_cpu = _solve_codebook_cached(dim, bits)

        codebook = Codebook(
            centroids=centroids_cpu.to(device),
            boundaries=boundaries_cpu.to(device),
            bits=bits,
            dim=dim,
        )
        self._device_cache[key] = codebook
        return codebook

    def clear(self) -> None:
        """Clear all cached device copies."""
        self._device_cache.clear()


# Module-level singleton for convenience
_default_registry = CodebookRegistry()


def get_codebook(
    dim: int,
    bits: int,
    device: torch.device | None = None,
) -> Codebook:
    """Get a codebook from the default global registry.

    Convenience function that uses a module-level singleton.

    Args:
        dim: Vector dimension.
        bits: Quantization bits.
        device: Target device.

    Returns:
        A Codebook with tensors on the requested device.
    """
    return _default_registry.get(dim, bits, device)
