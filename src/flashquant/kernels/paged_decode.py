"""Split-K paged decode attention dispatch.

Dispatches to the CUDA/Triton fused paged decode kernel when available,
otherwise falls back to the pure PyTorch reference. The fused kernel
reads TQ4-compressed blocks directly from the page table, decompresses
in SRAM, and computes attention in a single fused pass.

Split-K partitioning is the default strategy: the KV sequence is split
across multiple thread blocks (splits), each computing a partial
softmax. The partial results are then reduced to produce the final
attention output. This improves GPU utilization for long sequences
during decode (single-token queries).
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch

from flashquant.kernels.cpu_reference import paged_decode_reference
from flashquant.profiling import trace_attention

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try C++ / Triton extension
# ---------------------------------------------------------------------------

_USE_C_PAGED_DECODE = False
_c_paged_decode = None
try:
    from flashquant import _C  # type: ignore[attr-defined]

    if hasattr(_C, "split_k_paged_decode"):
        _c_paged_decode = _C.split_k_paged_decode
        _USE_C_PAGED_DECODE = True
        logger.debug("Using _C.split_k_paged_decode kernel")
except ImportError:
    pass

if not _USE_C_PAGED_DECODE:
    logger.debug(
        "Split-K paged decode kernel not available; using PyTorch fallback"
    )


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


@trace_attention
def split_k_paged_decode(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    centroids_k: torch.Tensor,
    centroids_v: torch.Tensor,
    rotation: torch.Tensor,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    sm_scale: Optional[float] = None,
    out: Optional[torch.Tensor] = None,
    num_splits: int = 0,
) -> torch.Tensor:
    """Split-K paged decode attention from packed TQ4 cache.

    Pre-rotates Q by rotation^T, launches the fused paged kernel that
    decompresses TQ4 blocks in-tile from the page table, then
    post-rotates the output by rotation to return to original space.

    Args:
        q: Query ``(num_seqs, H_Q, D)`` fp16/bf16.
        kv_cache: Packed paged cache ``(num_blocks, block_size, total_bytes)``
            uint8.
        block_table: Page table ``(num_seqs, max_blocks_per_seq)`` int32.
        seq_lens: Sequence lengths ``(num_seqs,)`` int32.
        centroids_k: Key codebook ``(C,)`` fp32.
        centroids_v: Value codebook ``(C,)`` fp32.
        rotation: Orthogonal rotation ``(D, D)`` fp32.
        num_kv_heads: Number of KV heads.
        head_dim: Head dimension.
        block_size: Tokens per block.
        sm_scale: Softmax scale. Defaults to ``1 / sqrt(D)``.
        out: Optional pre-allocated output ``(num_seqs, H_Q, D)``.
        num_splits: Number of Split-K partitions. 0 = auto.

    Returns:
        Attention output ``(num_seqs, H_Q, D)`` in original space.
    """
    num_seqs, H_Q, D = q.shape

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    rot = rotation.to(q.device)

    if _USE_C_PAGED_DECODE and q.is_cuda:
        assert _c_paged_decode is not None
        return _c_paged_decode(
            q, kv_cache, block_table, seq_lens,
            centroids_k, centroids_v, rotation,
            num_kv_heads, head_dim, block_size,
            sm_scale, out, num_splits,
        )

    # CPU/fallback path: pre-rotate Q, run reference, post-rotate
    q_rot = torch.matmul(q.float(), rot.T).to(q.dtype)

    out_rot = paged_decode_reference(
        q_rot, kv_cache, block_table, seq_lens,
        centroids_k, centroids_v,
        num_kv_heads, head_dim, block_size,
        sm_scale=sm_scale,
    )

    # Post-rotate output back to original space
    result = torch.matmul(out_rot.float(), rot).to(q.dtype)

    if out is not None:
        out.copy_(result)
        return out
    return result
