"""Tests for a pre-allocated buffer pool for incremental KV cache appends.

The buffer pool pre-allocates a fixed-size buffer and tracks a write cursor.
Appends are O(1) slice-writes, not O(N) torch.cat operations.
"""
from __future__ import annotations

import time

import pytest
import torch


# ---------------------------------------------------------------------------
# Reference buffer pool implementation
# ---------------------------------------------------------------------------


class RefBufferPool:
    """Pre-allocated buffer pool for O(1) append operations.

    Pre-allocates a tensor of shape (max_seq, ...) and tracks a cursor.
    append() writes into the next slice; get() returns the valid prefix.
    """

    def __init__(
        self, max_seq: int, *shape: int, dtype: torch.dtype = torch.float32
    ) -> None:
        self.max_seq = max_seq
        self.buffer = torch.zeros(max_seq, *shape, dtype=dtype)
        self.length = 0

    def append(self, data: torch.Tensor) -> None:
        """Append data to the buffer. data.shape[0] is the number of tokens."""
        n = data.shape[0]
        if self.length + n > self.max_seq:
            raise RuntimeError(
                f"Buffer overflow: {self.length} + {n} > {self.max_seq}"
            )
        self.buffer[self.length : self.length + n] = data
        self.length += n

    def get(self) -> torch.Tensor:
        """Return the valid portion of the buffer."""
        return self.buffer[: self.length]

    def reset(self) -> None:
        """Reset the cursor to 0 (does not zero the buffer)."""
        self.length = 0


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAppendGetRoundtrip:
    """Append data, get it back unchanged."""

    def test_append_get_roundtrip(self) -> None:
        pool = RefBufferPool(100, 64)
        data = torch.randn(10, 64)
        pool.append(data)
        result = pool.get()
        assert result.shape == (10, 64)
        assert torch.equal(result, data)


class TestSequentialAppend:
    """Append N times, get returns all N."""

    def test_sequential_append(self) -> None:
        pool = RefBufferPool(100, 32)
        chunks = [torch.randn(5, 32) for _ in range(10)]
        for chunk in chunks:
            pool.append(chunk)

        result = pool.get()
        assert result.shape == (50, 32)
        expected = torch.cat(chunks, dim=0)
        assert torch.equal(result, expected)


class TestNoCopyOnAppend:
    """Append should be O(1), not O(N) (timing test)."""

    def test_no_copy_on_append(self) -> None:
        max_seq = 10000
        pool = RefBufferPool(max_seq, 128)

        # Fill most of the buffer
        big_chunk = torch.randn(max_seq - 100, 128)
        pool.append(big_chunk)

        # Time 100 small appends
        times = []
        for _ in range(100):
            data = torch.randn(1, 128)
            start = time.perf_counter()
            pool.append(data)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            pool.length -= 1  # undo for next iteration

        avg_time = sum(times) / len(times)
        # Should be very fast (sub-millisecond on any hardware)
        assert avg_time < 0.01, (
            f"Append average time {avg_time * 1000:.2f}ms -- "
            f"expected < 10ms for O(1) operation"
        )


class TestOverflowRaises:
    """Appending beyond max_seq should raise."""

    def test_overflow_raises(self) -> None:
        pool = RefBufferPool(10, 8)
        pool.append(torch.randn(10, 8))
        with pytest.raises(RuntimeError, match="overflow"):
            pool.append(torch.randn(1, 8))


class TestReset:
    """After reset, length should be 0."""

    def test_reset(self) -> None:
        pool = RefBufferPool(100, 16)
        pool.append(torch.randn(50, 16))
        assert pool.length == 50
        pool.reset()
        assert pool.length == 0
        result = pool.get()
        assert result.shape == (0, 16)


class TestLargeSequence:
    """Should work up to 32768 tokens."""

    @pytest.mark.slow
    def test_32k_tokens(self) -> None:
        max_seq = 32768
        pool = RefBufferPool(max_seq, 128)

        # Append in chunks
        chunk_size = 512
        for i in range(0, max_seq, chunk_size):
            n = min(chunk_size, max_seq - pool.length)
            pool.append(torch.randn(n, 128))

        assert pool.length == max_seq
        result = pool.get()
        assert result.shape == (max_seq, 128)
