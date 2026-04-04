"""Performance and scaling tests.

Validates that key operations have the expected computational complexity.
All tests use wall-clock timing and generous margins to avoid flakiness.
"""
from __future__ import annotations

import time

import pytest
import torch

from .conftest import RefQuantizerMSE


class TestCompressLinearScaling:
    """10x input should take < 15x time (linear scaling, not quadratic)."""

    def test_compress_linear_scaling(self) -> None:
        dim = 128
        q = RefQuantizerMSE(dim, 4, seed=42)

        # Small input
        x_small = torch.randn(100, dim)

        # Warmup
        q.quantize(x_small)

        # Time small
        start = time.perf_counter()
        for _ in range(10):
            q.quantize(x_small)
        t_small = (time.perf_counter() - start) / 10

        # Large input (10x)
        x_large = torch.randn(1000, dim)

        # Time large
        start = time.perf_counter()
        for _ in range(10):
            q.quantize(x_large)
        t_large = (time.perf_counter() - start) / 10

        # 10x input should take < 15x time (generous margin for overhead)
        ratio = t_large / (t_small + 1e-9)
        assert ratio < 15, (
            f"Compression scaling ratio {ratio:.1f}x for 10x input -- "
            f"expected < 15x for linear scaling (t_small={t_small*1000:.2f}ms, "
            f"t_large={t_large*1000:.2f}ms)"
        )


class TestBufferAppendConstantTime:
    """Late appends should not be slower than early appends."""

    def test_buffer_append_constant_time(self) -> None:
        max_seq = 10000
        dim = 128
        buffer = torch.zeros(max_seq, dim)
        cursor = 0

        # Early appends
        early_times = []
        for _ in range(100):
            data = torch.randn(1, dim)
            start = time.perf_counter()
            buffer[cursor : cursor + 1] = data
            elapsed = time.perf_counter() - start
            early_times.append(elapsed)
            cursor += 1

        # Fill most of the buffer
        buffer[cursor : max_seq - 100] = torch.randn(max_seq - 200, dim)
        cursor = max_seq - 100

        # Late appends
        late_times = []
        for _ in range(100):
            data = torch.randn(1, dim)
            start = time.perf_counter()
            buffer[cursor : cursor + 1] = data
            elapsed = time.perf_counter() - start
            late_times.append(elapsed)
            cursor += 1

        avg_early = sum(early_times) / len(early_times)
        avg_late = sum(late_times) / len(late_times)

        # Late appends should not be more than 5x slower than early
        ratio = avg_late / (avg_early + 1e-9)
        assert ratio < 5, (
            f"Late appends {ratio:.1f}x slower than early "
            f"(early={avg_early*1e6:.1f}us, late={avg_late*1e6:.1f}us)"
        )


class TestNoQuadraticCat:
    """Total time for N appends should be O(N), not O(N^2).

    Compares pre-allocated buffer (O(N)) vs torch.cat (O(N^2)).
    The buffer approach should be significantly faster for large N.
    """

    def test_no_quadratic_cat(self) -> None:
        N = 500
        dim = 128

        # Method 1: Pre-allocated buffer (O(N))
        buffer = torch.zeros(N, dim)
        start = time.perf_counter()
        for i in range(N):
            buffer[i] = torch.randn(dim)
        t_buffer = time.perf_counter() - start

        # Method 2: torch.cat chain (O(N^2))
        start = time.perf_counter()
        result = torch.empty(0, dim)
        for _ in range(N):
            new_row = torch.randn(1, dim)
            result = torch.cat([result, new_row], dim=0)
        t_cat = time.perf_counter() - start

        # Buffer should be faster (at least 2x for N=500)
        # Both have the same data volume; difference is allocation pattern
        speedup = t_cat / (t_buffer + 1e-9)
        assert speedup > 1.5, (
            f"Buffer append only {speedup:.1f}x faster than cat -- "
            f"expected > 1.5x (buffer={t_buffer*1000:.1f}ms, "
            f"cat={t_cat*1000:.1f}ms)"
        )
