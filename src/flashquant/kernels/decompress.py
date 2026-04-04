"""TQ4 decompression dispatch: tries CUDA kernel via _C, falls back to CPU reference.

Provides a single entry point ``tq4_decompress()`` that selects the best
available backend. Decompression does NOT apply the rotation matrix --
output stays in rotated space. The caller must pre-rotate Q and
post-rotate the output if needed.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from flashquant.kernels.cpu_reference import decompress_reference
from flashquant.profiling import trace_decompress

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try C++ extension
# ---------------------------------------------------------------------------

_USE_C_DECOMPRESS = False
_c_tq4_decompress = None
try:
    from flashquant import _C  # type: ignore[attr-defined]

    if hasattr(_C, "tq4_decompress"):
        _c_tq4_decompress = _C.tq4_decompress
        _USE_C_DECOMPRESS = True
        logger.debug("Using _C.tq4_decompress CUDA kernel")
except ImportError:
    pass

if not _USE_C_DECOMPRESS:
    logger.debug(
        "CUDA decompress kernel not available; using PyTorch fallback"
    )


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


@trace_decompress
def tq4_decompress(
    packed: torch.Tensor,
    norms: torch.Tensor,
    centroids: torch.Tensor,
    dtype: torch.dtype = torch.float16,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Decompress TQ4 nibble-packed data to full-precision vectors.

    Does NOT apply the rotation matrix -- output remains in rotated
    space. The caller must apply post-rotation if needed.

    Dispatches to the CUDA kernel when available, otherwise falls
    back to pure PyTorch.

    Args:
        packed: ``(N, H, D//2)`` uint8 nibble-packed centroid indices.
        norms: ``(N, H, 1)`` fp32 per-vector norms.
        centroids: ``(C,)`` fp32 centroid table (C=16 for TQ4).
        dtype: Output dtype (default: torch.float16).
        out: Optional pre-allocated ``(N, H, D)`` output tensor.

    Returns:
        Tensor of shape ``(N, H, D)`` in ``dtype``, still in rotated
        space.
    """
    if _USE_C_DECOMPRESS and packed.is_cuda:
        assert _c_tq4_decompress is not None
        return _c_tq4_decompress(packed, norms, centroids, dtype, out)

    # CPU/fallback path
    result = decompress_reference(packed, norms, centroids)
    result = result.to(dtype)

    if out is not None:
        out.copy_(result)
        return out
    return result
