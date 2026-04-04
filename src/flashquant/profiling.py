"""Profiling decorators for FlashQuant operations.

Wraps functions with ``torch.profiler.record_function`` so that
compress, decompress, and attention operations appear as named regions
in PyTorch profiler traces (chrome://tracing, TensorBoard, etc.).

Usage::

    @trace_compress
    def my_compress(x, rotation, boundaries):
        ...

    @trace_attention
    def my_attention(q, k, v, sm_scale):
        ...

The decorators are zero-overhead when profiling is not active --
``record_function`` is a no-op context manager in that case.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, TypeVar

import torch

F = TypeVar("F", bound=Callable[..., Any])


def trace_compress(fn: F) -> F:
    """Wrap a function with a ``flashquant::compress`` profiler region.

    Args:
        fn: Function to wrap.

    Returns:
        Wrapped function that records a profiler region on each call.
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with torch.profiler.record_function("flashquant::compress"):
            return fn(*args, **kwargs)

    return wrapper  # type: ignore[return-value]


def trace_decompress(fn: F) -> F:
    """Wrap a function with a ``flashquant::decompress`` profiler region.

    Args:
        fn: Function to wrap.

    Returns:
        Wrapped function that records a profiler region on each call.
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with torch.profiler.record_function("flashquant::decompress"):
            return fn(*args, **kwargs)

    return wrapper  # type: ignore[return-value]


def trace_attention(fn: F) -> F:
    """Wrap a function with a ``flashquant::attention`` profiler region.

    Args:
        fn: Function to wrap.

    Returns:
        Wrapped function that records a profiler region on each call.
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with torch.profiler.record_function("flashquant::attention"):
            return fn(*args, **kwargs)

    return wrapper  # type: ignore[return-value]
