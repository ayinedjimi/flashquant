"""TQ4 compression dispatch: tries CUDA kernel via _C, falls back to CPU reference.

Provides a single entry point ``tq4_compress()`` that selects the best
available backend:

1. ``_C.tq4_compress`` -- C++/CUDA fused kernel (fastest).
2. ``cpu_reference.compress_reference`` -- Pure PyTorch fallback.

The dispatch is transparent to callers; the function signature is
identical regardless of the backend.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch

from flashquant.kernels.cpu_reference import compress_reference
from flashquant.profiling import trace_compress

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try C++ extension
# ---------------------------------------------------------------------------

_USE_C_COMPRESS = False
_c_tq4_compress = None
try:
    from flashquant import _C  # type: ignore[attr-defined]

    if hasattr(_C, "tq4_compress"):
        _c_tq4_compress = _C.tq4_compress
        _USE_C_COMPRESS = True
        logger.debug("Using _C.tq4_compress CUDA kernel")
except ImportError:
    pass

if not _USE_C_COMPRESS:
    logger.debug("CUDA compress kernel not available; using PyTorch fallback")


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


@trace_compress
def tq4_compress(
    x: torch.Tensor,
    rotation: torch.Tensor,
    boundaries: torch.Tensor,
    *,
    rotation_T_even: Optional[torch.Tensor] = None,
    rotation_T_odd: Optional[torch.Tensor] = None,
    out: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compress vectors to TQ4 nibble-packed format.

    Dispatches to the CUDA kernel when available, otherwise falls
    back to pure PyTorch.

    Args:
        x: Input vectors ``(N, H, D)`` in any float dtype.
        rotation: Orthogonal rotation matrix ``(D, D)`` fp32.
        boundaries: Quantization boundaries ``(2^bits - 1,)`` fp32.
        rotation_T_even: Optional pre-split even columns of rotation.T,
            shape ``(D, D//2)`` fp32. Used by the CUDA kernel for
            contiguous loads. If None and CUDA is used, computed
            on the fly.
        rotation_T_odd: Optional pre-split odd columns of rotation.T,
            shape ``(D, D//2)`` fp32.
        out: Optional pre-allocated ``(packed, norms)`` buffers for
            the CUDA kernel. Follows PyTorch ``out`` convention.

    Returns:
        Tuple of (packed, norms):
            - packed: ``(N, H, D//2)`` uint8 nibble-packed indices.
            - norms: ``(N, H, 1)`` fp32 per-vector norms.
    """
    if _USE_C_COMPRESS and x.is_cuda:
        assert _c_tq4_compress is not None
        # The C kernel expects pre-split rotation columns
        if rotation_T_even is None or rotation_T_odd is None:
            rot_t = rotation.T.contiguous()
            rotation_T_even = rot_t[:, 0::2].contiguous()
            rotation_T_odd = rot_t[:, 1::2].contiguous()
        return _c_tq4_compress(
            x, rotation_T_even, rotation_T_odd, boundaries, out
        )

    # CPU/fallback path
    return compress_reference(x, rotation, boundaries)
