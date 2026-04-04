"""Fused TQ4 Flash Attention dispatch.

Dispatches to the CUDA/Triton fused kernel when available, otherwise
falls back to the pure PyTorch reference implementation that
decompresses K/V and runs standard attention.

The fused kernel operates in rotated space: Q is pre-rotated by Pi^T,
the kernel decompresses K/V inline (without HBM writes), computes
attention, and the output is post-rotated by Pi.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch

from flashquant.kernels.cpu_reference import fused_tq_attention_reference
from flashquant.profiling import trace_attention

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try C++ / Triton extension
# ---------------------------------------------------------------------------

_USE_C_ATTENTION = False
_c_fused_tq_attention = None
try:
    from flashquant import _C  # type: ignore[attr-defined]

    if hasattr(_C, "fused_tq_attention"):
        _c_fused_tq_attention = _C.fused_tq_attention
        _USE_C_ATTENTION = True
        logger.debug("Using _C.fused_tq_attention kernel")
except ImportError:
    pass

if not _USE_C_ATTENTION:
    logger.debug(
        "Fused TQ4 attention kernel not available; using PyTorch fallback"
    )


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


@trace_attention
def fused_tq_attention(
    q: torch.Tensor,
    k_packed: torch.Tensor,
    k_norms: torch.Tensor,
    v_packed: torch.Tensor,
    v_norms: torch.Tensor,
    centroids_k: torch.Tensor,
    centroids_v: torch.Tensor,
    rotation: torch.Tensor,
    sm_scale: Optional[float] = None,
    is_causal: bool = False,
) -> torch.Tensor:
    """Fused TQ4 Flash Attention with both K and V compressed.

    Pre-rotates Q by rotation^T, launches the kernel that decompresses
    both K and V inline, then post-rotates the output by rotation to
    return to the original coordinate space.

    Args:
        q: Query ``(B, H_Q, N_Q, D)`` fp16/bf16.
        k_packed: Nibble-packed key indices ``(B, H_KV, N_KV, D//2)`` uint8.
        k_norms: Key norms ``(B, H_KV, N_KV, 1)`` or ``(..., 1)`` fp32.
        v_packed: Nibble-packed value indices ``(B, H_KV, N_KV, D//2)`` uint8.
        v_norms: Value norms ``(B, H_KV, N_KV, 1)`` fp32.
        centroids_k: Key codebook ``(C,)`` fp32.
        centroids_v: Value codebook ``(C,)`` fp32.
        rotation: Shared orthogonal rotation ``(D, D)`` fp32.
        sm_scale: Softmax scale. Defaults to ``1 / sqrt(D)``.
        is_causal: Apply causal masking.

    Returns:
        Attention output ``(B, H_Q, N_Q, D)`` in original space.
    """
    B, H_Q, N_Q, D = q.shape

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    if _USE_C_ATTENTION and q.is_cuda:
        assert _c_fused_tq_attention is not None
        return _c_fused_tq_attention(
            q, k_packed, k_norms, v_packed, v_norms,
            centroids_k, centroids_v, rotation,
            sm_scale, is_causal,
        )

    # Pre-rotate Q for the reference implementation
    rot = rotation.to(q.device)
    q_rot = torch.matmul(q.float(), rot.T).to(q.dtype)

    return fused_tq_attention_reference(
        q_rot, k_packed, k_norms, v_packed, v_norms,
        centroids_k, centroids_v, rotation,
        sm_scale=sm_scale, is_causal=is_causal,
    )
