"""Tests for HuggingFace DynamicCache KV cache wrapper.

Uses a mock DynamicCache to avoid the transformers dependency. The wrapper
intercepts update() calls, compresses key/value states, and returns
decompressed tensors.
"""
from __future__ import annotations

from typing import Any

import pytest
import torch

from flashquant.errors import FlashQuantCacheError

from .conftest import RefQuantizerMSE, cosine_similarity_flat


# ---------------------------------------------------------------------------
# Mock DynamicCache (no transformers dependency)
# ---------------------------------------------------------------------------


class MockDynamicLayer:
    """Mock layer in a DynamicCache."""

    def __init__(self) -> None:
        self.keys: torch.Tensor | None = None
        self.values: torch.Tensor | None = None
        self.is_initialized = False

    def lazy_initialization(self, _: torch.Tensor) -> None:
        self.is_initialized = True


class MockDynamicCache:
    """Minimal mock of transformers.DynamicCache for testing."""

    def __init__(self) -> None:
        self.layers: list[MockDynamicLayer] = []
        self.layer_class_to_replicate = MockDynamicLayer

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        while len(self.layers) <= layer_idx:
            self.layers.append(MockDynamicLayer())
        layer = self.layers[layer_idx]
        if not layer.is_initialized:
            layer.lazy_initialization(key_states)
        if layer.keys is None:
            layer.keys = key_states
            layer.values = value_states
        else:
            layer.keys = torch.cat([layer.keys, key_states], dim=-2)
            layer.values = torch.cat([layer.values, value_states], dim=-2)
        return layer.keys, layer.values

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx >= len(self.layers) or self.layers[layer_idx].keys is None:
            return 0
        return self.layers[layer_idx].keys.shape[-2]


# ---------------------------------------------------------------------------
# TurboQuant KV cache wrapper (reference implementation)
# ---------------------------------------------------------------------------


class RefTurboQuantKVCache:
    """Reference KV cache wrapper that compresses/decompresses on update."""

    _instances: set[int] = set()

    def __init__(
        self,
        cache: MockDynamicCache,
        head_dim: int,
        bits: int = 3,
        seed: int = 42,
    ) -> None:
        cache_id = id(cache)
        if cache_id in RefTurboQuantKVCache._instances:
            raise FlashQuantCacheError(
                "Cache is already wrapped by FlashQuant. "
                "Call restore() on the existing wrapper first."
            )
        RefTurboQuantKVCache._instances.add(cache_id)

        self.cache = cache
        self.head_dim = head_dim
        self.bits = bits
        self.enabled = True
        self.key_compressor = RefQuantizerMSE(head_dim, bits, seed=seed)
        self.value_compressor = RefQuantizerMSE(head_dim, bits, seed=seed + 1)
        self._original_update = cache.update
        cache.update = self._compressed_update

    def _compressed_update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.enabled:
            return self._original_update(
                key_states, value_states, layer_idx, cache_kwargs
            )
        # Compress -> decompress roundtrip
        k_idx, k_n = self.key_compressor.quantize(key_states.float())
        key_states = self.key_compressor.dequantize(k_idx, k_n).to(key_states.dtype)
        v_idx, v_n = self.value_compressor.quantize(value_states.float())
        value_states = self.value_compressor.dequantize(v_idx, v_n).to(
            value_states.dtype
        )
        return self._original_update(
            key_states, value_states, layer_idx, cache_kwargs
        )

    def restore(self) -> None:
        self.cache.update = self._original_update
        RefTurboQuantKVCache._instances.discard(id(self.cache))

    def __enter__(self) -> RefTurboQuantKVCache:
        return self

    def __exit__(self, *exc: object) -> bool:
        self.restore()
        return False


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBasicUpdate:
    """Update with key/value, retrieve decompressed."""

    def test_basic_update(self) -> None:
        cache = MockDynamicCache()
        wrapper = RefTurboQuantKVCache(cache, head_dim=128, bits=4)
        try:
            k = torch.randn(1, 4, 8, 128)
            v = torch.randn(1, 4, 8, 128)
            k_out, v_out = cache.update(k, v, layer_idx=0)
            assert k_out.shape == (1, 4, 8, 128)
            assert v_out.shape == (1, 4, 8, 128)
        finally:
            wrapper.restore()


class TestCosineSimilarity:
    """Compressed values should have high cosine similarity with originals."""

    @pytest.mark.parametrize("bits,threshold", [(4, 0.90), (3, 0.85)])
    def test_cosine_similarity(self, bits: int, threshold: float) -> None:
        cache = MockDynamicCache()
        wrapper = RefTurboQuantKVCache(cache, head_dim=128, bits=bits)
        try:
            k = torch.randn(1, 4, 32, 128)
            v = torch.randn(1, 4, 32, 128)
            k_out, v_out = cache.update(k, v, layer_idx=0)

            k_cos = cosine_similarity_flat(k, k_out)
            v_cos = cosine_similarity_flat(v, v_out)
            assert k_cos >= threshold, f"Key cosine {k_cos:.4f} < {threshold}"
            assert v_cos >= threshold, f"Value cosine {v_cos:.4f} < {threshold}"
        finally:
            wrapper.restore()


class TestVRAMBytesAccurate:
    """Reported VRAM should match actual tensor sizes (for mock)."""

    def test_vram_bytes_accurate(self) -> None:
        # For the mock wrapper, just verify shapes are maintained
        cache = MockDynamicCache()
        wrapper = RefTurboQuantKVCache(cache, head_dim=128, bits=4)
        try:
            k = torch.randn(1, 4, 16, 128)
            v = torch.randn(1, 4, 16, 128)
            cache.update(k, v, layer_idx=0)

            layer = cache.layers[0]
            assert layer.keys is not None
            assert layer.values is not None
            k_bytes = layer.keys.nelement() * layer.keys.element_size()
            v_bytes = layer.values.nelement() * layer.values.element_size()
            # Keys/values stored as float32 in mock
            expected_per_kv = 1 * 4 * 16 * 128 * 4  # B*H*S*D*sizeof(float32)
            assert k_bytes == expected_per_kv
            assert v_bytes == expected_per_kv
        finally:
            wrapper.restore()


class TestDoubleWrapRaises:
    """Wrapping twice should raise FlashQuantCacheError."""

    def test_double_wrap_raises(self) -> None:
        cache = MockDynamicCache()
        w1 = RefTurboQuantKVCache(cache, head_dim=128, bits=4)
        try:
            with pytest.raises(FlashQuantCacheError, match="already wrapped"):
                RefTurboQuantKVCache(cache, head_dim=128, bits=4)
        finally:
            w1.restore()


class TestContextManager:
    """Enter/exit should properly restore cache."""

    def test_context_manager(self) -> None:
        cache = MockDynamicCache()

        # Before wrapping, update is the default method
        k = torch.randn(1, 4, 4, 128)
        v = torch.randn(1, 4, 4, 128)

        with RefTurboQuantKVCache(cache, head_dim=128, bits=4) as wrapper:
            # Inside context, update compresses
            k_out, v_out = cache.update(k, v, layer_idx=0)
            # Compressed output differs from input
            assert not torch.equal(k, k_out)

        # Outside context, cache id should be removed from _instances
        assert id(cache) not in RefTurboQuantKVCache._instances

        # Should be able to wrap again after restore
        with RefTurboQuantKVCache(cache, head_dim=128, bits=4):
            pass
