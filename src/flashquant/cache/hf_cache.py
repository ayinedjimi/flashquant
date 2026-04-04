"""CompressedDynamicCache: HuggingFace DynamicCache integration.

Adapted from turboquant-vllm ``kv_cache.py`` with the following
improvements:

1. **CompressedBuffer** -- Pre-allocated ring buffers replace O(N^2)
   torch.cat with O(1) append. Each decode step writes 1 token without
   copying the entire sequence.

2. **Separate K/V codebooks** -- Keys use seed=42, values use seed=43
   for independent rotation matrices and codebooks.

3. **vram_bytes()** counts EVERYTHING -- compressed indices, norms,
   AND decompressed buffers (when present).

4. **FlashQuantCacheError** for double-wrap detection instead of a
   plain UserWarning.

5. **Context manager** support for automatic restore() on scope exit.

Storage per token per head (head_dim=128):

============  =======  =====  ===========
Mode          Dtype    Bytes  Compression
============  =======  =====  ===========
FP16 baseline fp16     256    1.0x
TQ4 (4-bit)   nibble   68     3.76x
============  =======  =====  ===========
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from flashquant.cache.buffer_pool import CompressedBuffer
from flashquant.core.compressor import CompressedValues, ValueCompressor
from flashquant.core.packing import nibble_pack, nibble_unpack
from flashquant.errors import FlashQuantCacheError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-layer compressed storage
# ---------------------------------------------------------------------------


@dataclass
class _CompressedLayer:
    """Storage for one compressed cache layer (indices + norms).

    Attributes:
        buffer: Pre-allocated CompressedBuffer with O(1) append.
        packed: Whether indices are nibble-packed (4-bit mode).
    """

    buffer: CompressedBuffer
    packed: bool = False


# ---------------------------------------------------------------------------
# CompressedDynamicCache
# ---------------------------------------------------------------------------


class CompressedDynamicCache:
    """KV cache with real VRAM savings via compressed index storage.

    Stores TurboQuant-compressed representations using pre-allocated
    CompressedBuffers and dequantizes lazily on each cache read.
    Uses separate codebooks for K (seed=42) and V (seed=43).

    Supports the context manager protocol for automatic ``restore()``
    on scope exit::

        cache = DynamicCache()
        with CompressedDynamicCache(cache, head_dim=128, bits=4) as compressed:
            # cache.update is patched inside the block
            ...
        # cache.update is restored here

    Attributes:
        cache: The wrapped DynamicCache instance.
        key_compressor: Compressor for key tensors (seed=42).
        value_compressor: Compressor for value tensors (seed=43).
        bits: Quantization bits per coordinate.
        head_dim: Model head dimension.
        enabled: Whether compression is active.
        fused_mode: When True, skip decompression in update().
    """

    def __init__(
        self,
        cache: Any,
        head_dim: int,
        bits: int = 4,
        *,
        key_seed: int = 42,
        value_seed: int = 43,
        max_seq_len: int = 32768,
    ) -> None:
        """Initialize the compressed KV cache wrapper.

        Args:
            cache: A HuggingFace DynamicCache instance to wrap.
            head_dim: Dimension of each attention head. Must be even
                when bits=4 (required for nibble packing).
            bits: Quantization bits per coordinate (default 4).
            key_seed: Random seed for key codebook (default 42).
            value_seed: Random seed for value codebook (default 43).
            max_seq_len: Maximum sequence length for buffer pre-allocation.

        Raises:
            ValueError: If bits=4 and head_dim is odd.
            FlashQuantCacheError: If cache is already wrapped.
        """
        if bits == 4 and head_dim % 2 != 0:
            raise ValueError(
                f"bits=4 requires even head_dim for nibble packing, "
                f"got {head_dim}"
            )

        self.cache = cache
        self.head_dim = head_dim
        self.bits = bits
        self._nibble_packed = bits == 4
        self.enabled = True
        self.fused_mode = False
        self._max_seq_len = max_seq_len

        # Separate codebooks: K uses key_seed, V uses value_seed
        self.key_compressor = ValueCompressor(
            head_dim, bits, seed=key_seed
        )
        self.value_compressor = ValueCompressor(
            head_dim, bits, seed=value_seed
        )

        # Per-layer compressed storage
        self._compressed_keys: List[Optional[_CompressedLayer]] = []
        self._compressed_values: List[Optional[_CompressedLayer]] = []

        # Per-layer decompressed buffers (incremental)
        self._decompressed_k: List[Optional[torch.Tensor]] = []
        self._decompressed_v: List[Optional[torch.Tensor]] = []
        self._original_dtype: torch.dtype = torch.bfloat16

        # Detect double-wrapping
        if hasattr(cache, "_flashquant_wrapped") and cache._flashquant_wrapped:
            raise FlashQuantCacheError(
                "Cache is already wrapped by FlashQuant. "
                "Call restore() on the existing wrapper first."
            )

        # Mark the cache as wrapped
        cache._flashquant_wrapped = True

        # Patch cache methods
        self._original_update = cache.update
        self._original_get_seq_length = cache.get_seq_length
        cache.update = self._compressed_update
        cache.get_seq_length = self._compressed_get_seq_length

    # ----- compression helpers -----

    def _compress_tensor(
        self,
        compressor: ValueCompressor,
        tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compress a tensor to packed/unpacked indices + float32 norms.

        Args:
            compressor: The compressor instance.
            tensor: Input tensor, shape (batch, heads, seq_len, head_dim).

        Returns:
            Tuple of (indices_uint8, norms_fp32).
            If nibble-packed, indices shape is (..., head_dim // 2).
        """
        compressed = compressor.compress(tensor)
        indices = compressed.indices.to(torch.uint8)

        if self._nibble_packed:
            indices = nibble_pack(indices)

        return indices, compressed.norms.float()

    def _dequantize_tensor(
        self,
        compressor: ValueCompressor,
        indices: torch.Tensor,
        norms: torch.Tensor,
    ) -> torch.Tensor:
        """Dequantize indices + norms back to the original dtype.

        Args:
            compressor: The compressor instance.
            indices: uint8 indices (possibly nibble-packed).
            norms: float32 norms.

        Returns:
            Reconstructed tensor in the original dtype.
        """
        if self._nibble_packed:
            idx_long = nibble_unpack(indices)
        else:
            idx_long = indices.long()

        compressed = CompressedValues(
            indices=idx_long,
            norms=norms,
            original_dtype=self._original_dtype,
        )
        return compressor.decompress(compressed)

    def _ensure_layer(self, layer_idx: int) -> None:
        """Extend layer lists to accommodate layer_idx."""
        while len(self._compressed_keys) <= layer_idx:
            self._compressed_keys.append(None)
            self._compressed_values.append(None)
            self._decompressed_k.append(None)
            self._decompressed_v.append(None)

    def _get_or_create_buffer(
        self,
        layer_idx: int,
        is_key: bool,
        batch: int,
        heads: int,
        half_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> _CompressedLayer:
        """Get existing or create new CompressedBuffer for a layer."""
        storage = (
            self._compressed_keys if is_key else self._compressed_values
        )
        if storage[layer_idx] is not None:
            return storage[layer_idx]  # type: ignore[return-value]

        buf = CompressedBuffer(
            max_seq=self._max_seq_len,
            batch=batch,
            heads=heads,
            half_dim=half_dim,
            dtype=dtype,
            device=device,
        )
        layer = _CompressedLayer(buffer=buf, packed=self._nibble_packed)
        storage[layer_idx] = layer
        return layer

    # ----- patched methods -----

    def _compressed_update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compress new tokens and optionally dequantize.

        Stores compressed representations in pre-allocated buffers.
        In normal mode, uses incremental dequantization (only NEW tokens
        decompressed). In fused_mode, skips decompression entirely.

        Args:
            key_states: Key tensor, shape (batch, heads, seq_len, head_dim).
            value_states: Value tensor, same shape.
            layer_idx: Transformer layer index.
            cache_kwargs: Additional cache arguments (unused).

        Returns:
            Tuple of (keys, values) decompressed for attention use.
        """
        if not self.enabled:
            return self._original_update(
                key_states, value_states, layer_idx, cache_kwargs
            )

        self._original_dtype = key_states.dtype
        self._ensure_layer(layer_idx)

        # Determine buffer dimensions
        batch, heads, new_seq, dim = key_states.shape
        half_dim = dim // 2 if self._nibble_packed else dim

        # Compress new tokens
        k_indices, k_norms = self._compress_tensor(
            self.key_compressor, key_states
        )
        v_indices, v_norms = self._compress_tensor(
            self.value_compressor, value_states
        )

        # Append to compressed buffers (O(1), no torch.cat)
        k_layer = self._get_or_create_buffer(
            layer_idx, True, batch, heads, half_dim,
            k_indices.dtype, k_indices.device,
        )
        v_layer = self._get_or_create_buffer(
            layer_idx, False, batch, heads, half_dim,
            v_indices.dtype, v_indices.device,
        )
        k_layer.buffer.append(k_indices, k_norms)
        v_layer.buffer.append(v_indices, v_norms)

        # Ensure DynamicCache has created layers up to layer_idx
        if hasattr(self.cache, "layer_class_to_replicate") and \
                self.cache.layer_class_to_replicate is not None:
            while len(self.cache.layers) <= layer_idx:
                self.cache.layers.append(
                    self.cache.layer_class_to_replicate()
                )

        # Fused mode: skip decompression entirely
        if self.fused_mode:
            layer = self.cache.layers[layer_idx]
            if hasattr(layer, "is_initialized") and not layer.is_initialized:
                layer.lazy_initialization(key_states)
            layer.keys = key_states
            layer.values = value_states
            return key_states, value_states

        # Incremental dequantization: only decompress NEW tokens
        new_k = self._dequantize_tensor(
            self.key_compressor, k_indices, k_norms
        )
        new_v = self._dequantize_tensor(
            self.value_compressor, v_indices, v_norms
        )

        # Append to decompressed running buffers
        if self._decompressed_k[layer_idx] is None:
            self._decompressed_k[layer_idx] = new_k
            self._decompressed_v[layer_idx] = new_v
        else:
            self._decompressed_k[layer_idx] = torch.cat(
                [self._decompressed_k[layer_idx], new_k], dim=-2
            )
            self._decompressed_v[layer_idx] = torch.cat(
                [self._decompressed_v[layer_idx], new_v], dim=-2
            )

        decompressed_k = self._decompressed_k[layer_idx]
        decompressed_v = self._decompressed_v[layer_idx]
        assert decompressed_k is not None
        assert decompressed_v is not None

        # Store in DynamicLayer for compatibility
        if hasattr(self.cache, "layers") and layer_idx < len(self.cache.layers):
            layer = self.cache.layers[layer_idx]
            if hasattr(layer, "is_initialized") and not layer.is_initialized:
                layer.lazy_initialization(key_states)
            if hasattr(layer, "keys"):
                layer.keys = decompressed_k
                layer.values = decompressed_v

        return decompressed_k, decompressed_v

    def _compressed_get_seq_length(self, layer_idx: int = 0) -> int:
        """Return cached sequence length from compressed storage.

        Args:
            layer_idx: Layer to query (default 0).

        Returns:
            Number of cached tokens for the given layer.
        """
        if not self.enabled:
            return self._original_get_seq_length(layer_idx)
        if layer_idx >= len(self._compressed_keys):
            return 0
        k_layer = self._compressed_keys[layer_idx]
        if k_layer is None:
            return 0
        return k_layer.buffer.length

    # ----- fused kernel API -----

    def get_compressed(
        self, layer_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return compressed K and V for a layer (fused kernel API).

        Args:
            layer_idx: Transformer layer index.

        Returns:
            (k_packed, k_norms, v_packed, v_norms) where packed tensors
            are uint8 and norms are fp32.

        Raises:
            FlashQuantCacheError: If the layer has not been initialized.
        """
        if layer_idx >= len(self._compressed_keys):
            raise FlashQuantCacheError(
                f"Layer {layer_idx} has not been initialized"
            )
        k_layer = self._compressed_keys[layer_idx]
        v_layer = self._compressed_values[layer_idx]
        if k_layer is None or v_layer is None:
            raise FlashQuantCacheError(
                f"Layer {layer_idx} has not been initialized"
            )
        k_indices, k_norms = k_layer.buffer.get()
        v_indices, v_norms = v_layer.buffer.get()
        return k_indices, k_norms, v_indices, v_norms

    # ----- properties -----

    @property
    def rotation_k(self) -> torch.Tensor:
        """Key rotation matrix [head_dim, head_dim] fp32."""
        return self.key_compressor.quantizer.rotation

    @property
    def rotation_v(self) -> torch.Tensor:
        """Value rotation matrix [head_dim, head_dim] fp32."""
        return self.value_compressor.quantizer.rotation

    @property
    def centroids_k(self) -> torch.Tensor:
        """Key codebook centroids [2^bits] fp32."""
        return self.key_compressor.quantizer.codebook.centroids

    @property
    def centroids_v(self) -> torch.Tensor:
        """Value codebook centroids [2^bits] fp32."""
        return self.value_compressor.quantizer.codebook.centroids

    # ----- VRAM accounting -----

    def vram_bytes(self) -> int:
        """Calculate total VRAM used by ALL storage.

        Counts compressed indices, compressed norms, AND decompressed
        buffers (when present). This gives the true VRAM footprint,
        not just the compressed portion.

        Returns:
            Total bytes across all layers.
        """
        total = 0

        # Compressed storage
        for k_layer, v_layer in zip(
            self._compressed_keys, self._compressed_values
        ):
            if k_layer is not None:
                k_idx, k_norms = k_layer.buffer.get()
                total += k_idx.nelement() * k_idx.element_size()
                total += k_norms.nelement() * k_norms.element_size()
            if v_layer is not None:
                v_idx, v_norms = v_layer.buffer.get()
                total += v_idx.nelement() * v_idx.element_size()
                total += v_norms.nelement() * v_norms.element_size()

        # Decompressed buffers
        for dk in self._decompressed_k:
            if dk is not None:
                total += dk.nelement() * dk.element_size()
        for dv in self._decompressed_v:
            if dv is not None:
                total += dv.nelement() * dv.element_size()

        return total

    def compressed_vram_bytes(self) -> int:
        """Calculate VRAM used by compressed storage only.

        Returns:
            Total bytes for compressed indices and norms.
        """
        total = 0
        for k_layer, v_layer in zip(
            self._compressed_keys, self._compressed_values
        ):
            if k_layer is not None:
                k_idx, k_norms = k_layer.buffer.get()
                total += k_idx.nelement() * k_idx.element_size()
                total += k_norms.nelement() * k_norms.element_size()
            if v_layer is not None:
                v_idx, v_norms = v_layer.buffer.get()
                total += v_idx.nelement() * v_idx.element_size()
                total += v_norms.nelement() * v_norms.element_size()
        return total

    def baseline_vram_bytes(self) -> int:
        """Estimate FP16 VRAM that would be used without compression.

        Returns:
            Total bytes if keys and values were stored as FP16 tensors.
        """
        total = 0
        for k_layer, v_layer in zip(
            self._compressed_keys, self._compressed_values
        ):
            for layer in (k_layer, v_layer):
                if layer is None:
                    continue
                idx, _ = layer.buffer.get()
                b, h, s, d = idx.shape
                if layer.packed:
                    d = d * 2  # Nibble-packed: d is head_dim // 2
                total += b * h * s * d * 2  # FP16 = 2 bytes
        return total

    def compression_stats(self) -> Dict[str, Any]:
        """Return compression statistics for reporting.

        Returns:
            Dict with layer count, sizes, compression ratio, etc.
        """
        num_layers = sum(
            1 for k in self._compressed_keys if k is not None
        )
        if num_layers == 0:
            return {}

        compressed_bytes = self.compressed_vram_bytes()
        baseline_bytes = self.baseline_vram_bytes()
        ratio = (
            baseline_bytes / compressed_bytes if compressed_bytes > 0 else 0.0
        )

        first_k = next(
            k for k in self._compressed_keys if k is not None
        )
        idx, _ = first_k.buffer.get()
        b, h, s, _ = idx.shape

        return {
            "num_layers": num_layers,
            "seq_len": s,
            "batch_size": b,
            "num_heads": h,
            "head_dim": self.head_dim,
            "bits": self.bits,
            "nibble_packed": self._nibble_packed,
            "compressed_mib": compressed_bytes / (1024 * 1024),
            "baseline_mib": baseline_bytes / (1024 * 1024),
            "total_vram_mib": self.vram_bytes() / (1024 * 1024),
            "compression_ratio": round(ratio, 2),
            "savings_mib": (baseline_bytes - compressed_bytes) / (1024 * 1024),
        }

    # ----- lifecycle -----

    def disable(self) -> None:
        """Disable compression, passing through to original update."""
        self.enabled = False

    def enable(self) -> None:
        """Re-enable compression after disable()."""
        self.enabled = True

    def restore(self) -> None:
        """Restore original methods on the wrapped cache.

        Call this to fully unwrap the cache and remove all FlashQuant
        interception.
        """
        self.cache.update = self._original_update
        self.cache.get_seq_length = self._original_get_seq_length
        if hasattr(self.cache, "_flashquant_wrapped"):
            self.cache._flashquant_wrapped = False

    def __enter__(self) -> CompressedDynamicCache:
        """Enter the context manager."""
        return self

    def __exit__(self, *exc: object) -> bool:
        """Exit the context manager, restoring the original cache methods."""
        self.restore()
        return False
