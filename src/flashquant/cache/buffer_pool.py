"""Pre-allocated ring buffer for compressed KV cache storage.

Replaces the O(N^2) ``torch.cat`` append pattern from the original
turboquant-vllm ``kv_cache.py`` with O(1) writes into a pre-allocated
buffer. At decode time, each step appends exactly 1 token to each
layer's buffer -- with torch.cat this requires copying the entire
previous sequence on every step (O(N^2) total for N tokens). The
CompressedBuffer pre-allocates for ``max_seq`` tokens and writes at
the current position, avoiding all copies.

The buffer provides zero-copy ``get()`` views via slicing, so
downstream consumers see only the valid portion without memory
duplication.
"""

from __future__ import annotations

import torch


class CompressedBuffer:
    """Pre-allocated ring buffer for compressed KV cache indices and norms.

    Allocates contiguous tensors for ``max_seq`` tokens upfront. The
    ``append()`` method writes at position ``self._len`` and increments
    the length counter. The ``get()`` method returns a zero-copy view
    of the valid portion.

    Attributes:
        max_seq: Maximum sequence length (buffer capacity).
        batch: Batch size.
        heads: Number of KV heads.
        half_dim: Half the head dimension (for nibble-packed indices).
        dtype: Data type for index storage (typically uint8).
        device: Device for tensor allocation.

    Examples:
        >>> buf = CompressedBuffer(
        ...     max_seq=1024, batch=1, heads=8, half_dim=64,
        ...     dtype=torch.uint8, device=torch.device("cpu"),
        ... )
        >>> packed = torch.zeros(1, 8, 1, 64, dtype=torch.uint8)
        >>> norms = torch.ones(1, 8, 1, 1, dtype=torch.float32)
        >>> buf.append(packed, norms)
        >>> buf.length
        1
        >>> idx, n = buf.get()
        >>> idx.shape
        torch.Size([1, 8, 1, 64])
    """

    def __init__(
        self,
        max_seq: int,
        batch: int,
        heads: int,
        half_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """Pre-allocate index and norm buffers.

        Args:
            max_seq: Maximum tokens to store (buffer capacity).
            batch: Batch size dimension.
            heads: Number of KV heads.
            half_dim: D // 2 for nibble-packed 4-bit indices.
            dtype: Index tensor dtype (typically torch.uint8).
            device: Target device for allocation.
        """
        self.max_seq = max_seq
        self.batch = batch
        self.heads = heads
        self.half_dim = half_dim

        # Shape: (batch, heads, max_seq, half_dim) for indices
        self._indices = torch.empty(
            batch, heads, max_seq, half_dim, dtype=dtype, device=device
        )
        # Shape: (batch, heads, max_seq, 1) for norms (always fp32)
        self._norms = torch.empty(
            batch, heads, max_seq, 1, dtype=torch.float32, device=device
        )
        self._len = 0

    def append(
        self, indices: torch.Tensor, norms: torch.Tensor
    ) -> None:
        """Write new tokens at the current position.

        Args:
            indices: Packed index tensor, shape
                ``(batch, heads, new_tokens, half_dim)``.
            norms: Norm tensor, shape
                ``(batch, heads, new_tokens, 1)`` in float32.

        Raises:
            RuntimeError: If appending would exceed the buffer capacity.
        """
        new_tokens = indices.shape[2]
        end = self._len + new_tokens
        if end > self.max_seq:
            raise RuntimeError(
                f"CompressedBuffer overflow: appending {new_tokens} tokens "
                f"at position {self._len} would exceed max_seq={self.max_seq}. "
                f"Increase FlashQuantConfig.max_seq_len."
            )
        self._indices[:, :, self._len : end, :] = indices
        self._norms[:, :, self._len : end, :] = norms
        self._len = end

    def get(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return zero-copy views of the valid buffer portion.

        Returns:
            Tuple of (indices, norms) where:
                - indices: shape ``(batch, heads, length, half_dim)``
                - norms: shape ``(batch, heads, length, 1)``
        """
        return (
            self._indices[:, :, : self._len, :],
            self._norms[:, :, : self._len, :],
        )

    @property
    def length(self) -> int:
        """Number of tokens currently stored in the buffer."""
        return self._len

    def reset(self) -> None:
        """Reset the buffer to empty (does not deallocate memory)."""
        self._len = 0
