"""FlashQuant attention implementation for vLLM.

The forward pass handles three distinct paths:

1. **Prefill**: Compress incoming K/V tokens, store in packed cache,
   pre-rotate Q, decompress full cache, run Flash Attention, post-rotate.

2. **Decode (decompress-all)**: Same as prefill but uses pre-allocated
   CUDA graph buffers for zero-allocation decode.

3. **Decode (fused paged)**: Compress incoming token, then run the
   fused paged TQ4 kernel that decompresses blocks in-tile from the
   page table. Eliminates all HBM writes of decompressed cache.
   Split-K partitioning is the default strategy.

Adapted from turboquant-vllm ``vllm/tq4_backend.py`` with fixes:
- Proper error handling via FlashQuantError hierarchy
- Profiling decorators on hot paths
- Separate K/V codebook support (future)
- Cleaner separation of prefill/decode/speculative paths
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch

from flashquant.config import FlashQuantConfig
from flashquant.core.quantizer import TurboQuantMSE
from flashquant.kernels.compress import tq4_compress
from flashquant.kernels.decompress import tq4_decompress
from flashquant.profiling import trace_attention
from flashquant.vllm.spec import TQ4_NORM_BYTES

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TQ4_BITS = 4
TQ4_KEY_SEED = 42
TQ4_VALUE_SEED = 43

# ---------------------------------------------------------------------------
# Fused paged decode feature gate
# ---------------------------------------------------------------------------

_fused_paged_kernel_available = False
_fused_paged_fn = None

try:
    from flashquant import _C  # type: ignore[attr-defined]
    if hasattr(_C, "fused_paged_tq4_decode"):
        _fused_paged_fn = _C.fused_paged_tq4_decode
        _fused_paged_kernel_available = True
except ImportError:
    pass

if not _fused_paged_kernel_available:
    # Try Triton kernel as second attempt
    try:
        from flashquant.kernels.paged_decode import split_k_paged_decode as _fused_paged_fn  # type: ignore[assignment]
        _fused_paged_kernel_available = True
    except ImportError:
        pass


def _parse_fused_paged_env() -> bool:
    """Parse TQ4_USE_FUSED_PAGED environment variable."""
    return os.environ.get("TQ4_USE_FUSED_PAGED", "").lower() in (
        "1", "true", "yes"
    )


# ---------------------------------------------------------------------------
# Implementation class factory
# ---------------------------------------------------------------------------


def _create_impl_class() -> type:
    """Create FlashQuantAttentionImpl that extends FlashAttentionImpl."""
    from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl

    class FlashQuantAttentionImpl(FlashAttentionImpl):
        """FlashQuant attention: compress -> store -> decompress -> FA -> post-rotate.

        Stores packed TQ4 bytes in a uint8 cache for real VRAM savings.
        Each forward() call:

        1. Compresses incoming K/V tokens to TQ4 packed bytes.
        2. Scatter-writes packed bytes to the uint8 cache via slot_mapping.
        3. Decompresses the full cache (or uses fused paged kernel).
        4. Runs Flash Attention with rotated Q and KV.
        5. Post-rotates the output by Pi.
        """

        def __init__(self, *args, **kwargs) -> None:
            """Initialize with TQ4 compression primitives."""
            super().__init__(*args, **kwargs)

            head_size = self.head_size
            num_kv_heads = self.num_kv_heads

            # TQ4 compression primitives (shared across layers)
            quantizer = TurboQuantMSE(
                head_size, TQ4_BITS, seed=TQ4_KEY_SEED
            )

            # Get device from vLLM config
            from vllm.config import get_current_vllm_config_or_none
            vllm_config = get_current_vllm_config_or_none()
            device = (
                vllm_config.device_config.device
                if vllm_config is not None
                else torch.device("cpu")
            )

            # Move compression primitives to target device
            self._tq4_rotation = quantizer.rotation.to(device)
            self._tq4_centroids = quantizer.codebook.centroids.to(device)
            self._tq4_boundaries = quantizer.codebook.boundaries.to(device)

            # Pre-split rotation.T for fused compress kernel
            rot_t = quantizer.rotation.T.contiguous()
            self._tq4_rot_T_even = rot_t[:, 0::2].contiguous().to(device)
            self._tq4_rot_T_odd = rot_t[:, 1::2].contiguous().to(device)

            # Byte layout offsets
            half_D = head_size // 2
            self._half_D = half_D
            self._k_idx_end = num_kv_heads * half_D
            self._k_norm_end = self._k_idx_end + num_kv_heads * TQ4_NORM_BYTES
            self._v_idx_end = self._k_norm_end + num_kv_heads * half_D
            self._total_bytes = self._v_idx_end + num_kv_heads * TQ4_NORM_BYTES

            # CUDA graph scratch buffers (lazy-allocated)
            self._cg_buffers_ready = False

            # Fused paged decode feature gate
            self._fused_paged_available = (
                _parse_fused_paged_env() and _fused_paged_kernel_available
            )

            # Max prefill length for buffer sizing
            self._max_prefill_len = (
                vllm_config.scheduler_config.max_num_batched_tokens
                if vllm_config is not None
                else 2048
            )

            logger.info(
                "FlashQuantAttentionImpl: %d KV heads, head_size=%d, "
                "%d bytes/token (%.2fx compression vs FP16), "
                "fused_paged=%s",
                num_kv_heads, head_size, self._total_bytes,
                (2 * num_kv_heads * head_size * 2) / self._total_bytes,
                self._fused_paged_available,
            )

        def _init_cg_buffers(
            self, kv_cache: torch.Tensor, compute_dtype: torch.dtype
        ) -> None:
            """Pre-allocate CUDA graph scratch buffers.

            Called once during warmup (first forward), before CUDA graph
            capture. Uses max-size + slicing pattern.
            """
            num_blocks, block_size, _ = kv_cache.shape
            max_tokens = num_blocks * block_size
            device = kv_cache.device
            H = self.num_kv_heads
            D = self.head_size
            half_D = self._half_D

            # Decompress buffer sizing
            if self._fused_paged_available:
                decompress_tokens = min(self._max_prefill_len, max_tokens)
            else:
                decompress_tokens = max_tokens

            self._cg_decompress_k = torch.empty(
                decompress_tokens, H, D, dtype=compute_dtype, device=device
            )
            self._cg_decompress_v = torch.empty_like(self._cg_decompress_k)

            # Compress output buffers (single token decode)
            self._cg_compress_packed = torch.empty(
                1, H, half_D, dtype=torch.uint8, device=device
            )
            self._cg_compress_norms = torch.empty(
                1, H, 1, dtype=torch.float32, device=device
            )

            # Q rotation buffers
            self._cg_q_rot = torch.empty(
                1, self.num_heads, D, dtype=torch.float32, device=device
            )
            self._cg_q_rot_cast = torch.empty(
                1, self.num_heads, D, dtype=compute_dtype, device=device
            )

            # Row assembly buffer for compress+store
            self._cg_compress_row = torch.empty(
                1, self._total_bytes, dtype=torch.uint8, device=device
            )

            self._cg_buffers_ready = True
            logger.info(
                "FlashQuant CUDA graph buffers allocated: "
                "decompress=%s, tokens=%d",
                self._cg_decompress_k.shape, decompress_tokens,
            )

        # ----- packed cache operations -----

        def _compress_and_store(
            self, key, value, kv_cache, slot_mapping, *,
            compress_out=None, row_out=None,
        ) -> None:
            """Compress K/V and scatter-write TQ4 bytes to packed cache."""
            N = key.shape[0]
            H = self.num_kv_heads

            row = (
                row_out[:N] if row_out is not None
                else torch.empty(
                    N, self._total_bytes, dtype=torch.uint8,
                    device=key.device,
                )
            )

            # Compress K
            k_packed, k_norms = tq4_compress(
                key, self._tq4_rotation, self._tq4_boundaries,
                rotation_T_even=self._tq4_rot_T_even,
                rotation_T_odd=self._tq4_rot_T_odd,
                out=compress_out,
            )
            row[:, :self._k_idx_end] = k_packed.reshape(N, -1)
            row[:, self._k_idx_end:self._k_norm_end] = (
                k_norms.reshape(N, H).contiguous().view(torch.uint8)
            )

            # Compress V
            v_packed, v_norms = tq4_compress(
                value, self._tq4_rotation, self._tq4_boundaries,
                rotation_T_even=self._tq4_rot_T_even,
                rotation_T_odd=self._tq4_rot_T_odd,
                out=compress_out,
            )
            row[:, self._k_norm_end:self._v_idx_end] = v_packed.reshape(N, -1)
            row[:, self._v_idx_end:] = (
                v_norms.reshape(N, H).contiguous().view(torch.uint8)
            )

            # Scatter-write to flat cache
            num_actual = slot_mapping.shape[0]
            flat_cache = kv_cache.view(-1, self._total_bytes)
            flat_cache[slot_mapping[:num_actual]] = row[:num_actual]

        def _decompress_cache(
            self, kv_cache, compute_dtype, *,
            apply_rotation=True, out_k=None, out_v=None,
        ):
            """Decompress packed uint8 cache -> key_cache, value_cache."""
            NB, BS, _ = kv_cache.shape
            H = self.num_kv_heads
            half_D = self._half_D
            D = self.head_size

            flat = kv_cache.reshape(NB * BS, self._total_bytes)

            # Extract K regions
            k_packed = flat[:, :self._k_idx_end].contiguous().reshape(
                -1, H, half_D
            )
            k_norms = (
                flat[:, self._k_idx_end:self._k_norm_end]
                .contiguous().view(torch.float32).reshape(-1, H, 1)
            )

            # Extract V regions
            v_packed = (
                flat[:, self._k_norm_end:self._v_idx_end]
                .contiguous().reshape(-1, H, half_D)
            )
            v_norms = (
                flat[:, self._v_idx_end:]
                .contiguous().view(torch.float32).reshape(-1, H, 1)
            )

            # Decompress (no rotation applied)
            key_out = tq4_decompress(
                k_packed, k_norms, self._tq4_centroids, compute_dtype,
                out=out_k,
            )
            value_out = tq4_decompress(
                v_packed, v_norms, self._tq4_centroids, compute_dtype,
                out=out_v,
            )

            # Optionally unrotate
            if apply_rotation:
                key_out = (
                    key_out.float() @ self._tq4_rotation
                ).to(compute_dtype)
                value_out = (
                    value_out.float() @ self._tq4_rotation
                ).to(compute_dtype)

            return (
                key_out.reshape(NB, BS, H, D),
                value_out.reshape(NB, BS, H, D),
            )

        # ----- prefill / decode paths -----

        def _tq4_prefill(self, query, key, value, kv_cache, attn_metadata):
            """Prefill path: compress, rotate Q, decompress with dynamic alloc."""
            num_actual = attn_metadata.num_actual_tokens
            if kv_cache is not None and key is not None and value is not None:
                self._compress_and_store(
                    key, value, kv_cache, attn_metadata.slot_mapping
                )

            q_slice = query[:num_actual]
            q_rot = (q_slice.float() @ self._tq4_rotation.T).to(q_slice.dtype)

            key_cache, value_cache = self._decompress_cache(
                kv_cache, query.dtype, apply_rotation=False,
            )
            return q_rot, key_cache, value_cache

        def _tq4_decode(self, query, key, value, kv_cache, attn_metadata):
            """Decode path: compress, rotate Q, decompress with CG buffers."""
            if key is not None and value is not None:
                self._compress_and_store(
                    key, value, kv_cache, attn_metadata.slot_mapping,
                    compress_out=(
                        self._cg_compress_packed,
                        self._cg_compress_norms,
                    ),
                    row_out=self._cg_compress_row,
                )

            q_slice = query[:attn_metadata.num_actual_tokens]
            q_rot_buf = self._cg_q_rot[:1]
            torch.matmul(q_slice.float(), self._tq4_rotation.T, out=q_rot_buf)
            self._cg_q_rot_cast[:1].copy_(q_rot_buf)

            key_cache, value_cache = self._decompress_cache(
                kv_cache, query.dtype, apply_rotation=False,
                out_k=self._cg_decompress_k,
                out_v=self._cg_decompress_v,
            )
            return self._cg_q_rot_cast[:1], key_cache, value_cache

        def _fused_decode_path(
            self, query, key, value, kv_cache, attn_metadata, output,
        ):
            """Fused paged decode: compress -> fused kernel."""
            num_actual = attn_metadata.num_actual_tokens

            if key is not None and value is not None:
                self._compress_and_store(
                    key, value, kv_cache, attn_metadata.slot_mapping,
                    compress_out=(
                        self._cg_compress_packed,
                        self._cg_compress_norms,
                    ),
                    row_out=self._cg_compress_row,
                )

            q_slice = query[:num_actual]
            assert _fused_paged_fn is not None
            _fused_paged_fn(
                q_slice,
                kv_cache,
                attn_metadata.block_table,
                attn_metadata.seq_lens,
                self._tq4_centroids,
                self._tq4_centroids,  # V centroids (same for now)
                self._tq4_rotation,
                self.num_kv_heads,
                self.head_size,
                kv_cache.shape[1],  # block_size
                sm_scale=self.scale,
                out=output[:num_actual],
            )
            return output

        # ----- forward -----

        @trace_attention
        def forward(
            self, layer, query, key, value, kv_cache, attn_metadata,
            output=None, output_scale=None, output_block_scale=None,
        ):
            """FlashQuant attention forward pass.

            Compress -> store -> pre-rotate Q -> decompress -> FA -> post-rotate.
            """
            assert output is not None, (
                "FlashQuantAttentionImpl requires pre-allocated output"
            )

            if output_scale is not None or output_block_scale is not None:
                raise NotImplementedError(
                    "Fused output quantization is not supported with "
                    "FlashQuant backend"
                )

            # Profiling mode: no metadata
            if attn_metadata is None:
                output.zero_()
                return output

            # Encoder attention: delegate to parent
            from vllm.v1.attention.backend import AttentionType
            if self.attn_type in (
                AttentionType.ENCODER_ONLY, AttentionType.ENCODER
            ):
                return self._forward_encoder_attention(
                    query[:attn_metadata.num_actual_tokens],
                    key[:attn_metadata.num_actual_tokens],
                    value[:attn_metadata.num_actual_tokens],
                    output[:attn_metadata.num_actual_tokens],
                    attn_metadata, layer,
                )

            num_actual = attn_metadata.num_actual_tokens

            # Lazy-init CUDA graph buffers
            if not self._cg_buffers_ready and kv_cache is not None:
                self._init_cg_buffers(kv_cache, compute_dtype=query.dtype)

            is_decode = self._cg_buffers_ready and num_actual == 1

            # === Fused paged decode path ===
            if self._fused_paged_available and is_decode:
                return self._fused_decode_path(
                    query, key, value, kv_cache, attn_metadata, output,
                )

            # === Decompress-all path ===
            if is_decode:
                q_rot, key_cache, value_cache = self._tq4_decode(
                    query, key, value, kv_cache, attn_metadata,
                )
            else:
                q_rot, key_cache, value_cache = self._tq4_prefill(
                    query, key, value, kv_cache, attn_metadata,
                )

            # Flash Attention with rotated Q and KV
            from vllm.v1.attention.backends.fa_utils import (
                flash_attn_varlen_func,
            )

            if attn_metadata.use_cascade:
                raise NotImplementedError(
                    "FlashQuant does not yet support cascade attention"
                )

            descale_shape = (
                attn_metadata.query_start_loc.shape[0] - 1,
                self.num_kv_heads,
            )
            q_descale = layer._q_scale.expand(descale_shape)
            k_descale = layer._k_scale.expand(descale_shape)
            v_descale = layer._v_scale.expand(descale_shape)

            flash_attn_varlen_func(
                q=q_rot,
                k=key_cache,
                v=value_cache,
                out=output[:num_actual],
                cu_seqlens_q=attn_metadata.query_start_loc,
                max_seqlen_q=attn_metadata.max_query_len,
                seqused_k=attn_metadata.seq_lens,
                max_seqlen_k=attn_metadata.max_seq_len,
                softmax_scale=self.scale,
                causal=attn_metadata.causal,
                alibi_slopes=self.alibi_slopes,
                window_size=(
                    list(self.sliding_window)
                    if self.sliding_window is not None
                    else None
                ),
                block_table=attn_metadata.block_table,
                softcap=self.logits_soft_cap,
                scheduler_metadata=attn_metadata.scheduler_metadata,
                fa_version=self.vllm_flash_attn_version,
                q_descale=q_descale,
                k_descale=k_descale,
                v_descale=v_descale,
                num_splits=attn_metadata.max_num_splits,
                s_aux=self.sinks,
            )

            # Post-rotate output by Pi
            out_slice = output[:num_actual]
            output[:num_actual] = (
                out_slice.float() @ self._tq4_rotation
            ).to(out_slice.dtype)

            return output

    return FlashQuantAttentionImpl


# Module-level cache
_FlashQuantAttentionImpl = None


def get_flashquant_impl_class() -> type:
    """Get the FlashQuantAttentionImpl class (lazy creation).

    Returns:
        The FlashQuantAttentionImpl class.
    """
    global _FlashQuantAttentionImpl
    if _FlashQuantAttentionImpl is None:
        _FlashQuantAttentionImpl = _create_impl_class()
    return _FlashQuantAttentionImpl
