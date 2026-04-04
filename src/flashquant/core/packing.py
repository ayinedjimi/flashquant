"""Nibble pack/unpack utilities for 4-bit index storage.

Packs pairs of 4-bit indices into single uint8 bytes to achieve 2x
storage reduction over one-byte-per-index representations. The packing
convention is: high nibble = even index, low nibble = odd index.

Tries the C++ extension ``_C`` first, falling back to pure PyTorch.

Examples:
    Pack and unpack round-trip::

        >>> indices = torch.randint(0, 16, (4, 128), dtype=torch.uint8)
        >>> packed = nibble_pack(indices)
        >>> packed.shape
        torch.Size([4, 64])
        >>> unpacked = nibble_unpack(packed)
        >>> (unpacked == indices.long()).all()
        True
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try C++ extension
# ---------------------------------------------------------------------------

_USE_C_PACKING = False
try:
    from flashquant import _C  # type: ignore[attr-defined]

    if hasattr(_C, "nibble_pack") and hasattr(_C, "nibble_unpack"):
        _USE_C_PACKING = True
        logger.debug("Using _C nibble pack/unpack backend")
except ImportError:
    pass

if not _USE_C_PACKING:
    logger.debug(
        "C++ packing extension not available; using pure PyTorch fallback"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def nibble_pack(indices: torch.Tensor) -> torch.Tensor:
    """Pack pairs of 4-bit indices into single uint8 bytes.

    Args:
        indices: uint8 tensor with values in [0, 15], last dimension
            must be even. Shape ``(..., D)``.

    Returns:
        uint8 tensor of shape ``(..., D // 2)`` with two indices
        per byte (high nibble = even index, low nibble = odd index).

    Raises:
        ValueError: If the last dimension is odd.
    """
    if indices.shape[-1] % 2 != 0:
        raise ValueError(
            f"Last dimension must be even for nibble packing, "
            f"got {indices.shape[-1]}"
        )

    if _USE_C_PACKING:
        return _C.nibble_pack(indices)  # type: ignore[attr-defined]

    even = indices[..., 0::2]
    odd = indices[..., 1::2]
    return ((even.to(torch.uint8) << 4) | odd.to(torch.uint8)).to(torch.uint8)


def nibble_unpack(packed: torch.Tensor) -> torch.Tensor:
    """Unpack nibble-packed uint8 bytes into pairs of 4-bit indices.

    Args:
        packed: uint8 tensor of shape ``(..., D // 2)`` with two
            indices per byte.

    Returns:
        Long tensor of shape ``(..., D)`` with individual indices
        suitable for centroid lookup.
    """
    if _USE_C_PACKING:
        return _C.nibble_unpack(packed)  # type: ignore[attr-defined]

    high = (packed >> 4).long()
    low = (packed & 0x0F).long()
    return torch.stack([high, low], dim=-1).flatten(-2)
