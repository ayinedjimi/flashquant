"""FlashQuant attention backend for vLLM.

Provides the backend class that vLLM's attention backend registry uses
to instantiate FlashQuant attention. The backend handles:

- KV cache shape calculation for the packed TQ4 format
- CUDA graph support declaration
- Implementation class routing
- Metadata builder delegation
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from flashquant.vllm.spec import tq4_bytes_per_token_kv

if TYPE_CHECKING:
    from vllm.v1.attention.backend import (
        AttentionCGSupport,
        AttentionImplBase,
        AttentionMetadataBuilder,
    )

logger = logging.getLogger(__name__)


def _get_backend_classes() -> tuple[type, type]:
    """Lazily import vLLM backend base classes.

    Returns:
        Tuple of (FlashAttentionBackend, FlashAttentionMetadataBuilder).
    """
    from vllm.v1.attention.backends.flash_attn import (
        FlashAttentionBackend,
        FlashAttentionMetadataBuilder,
    )
    return FlashAttentionBackend, FlashAttentionMetadataBuilder


class FlashQuantMetadataBuilder:
    """Metadata builder for FlashQuant with CUDA graph support.

    FlashQuant supports CUDA graphs for single-token decode. The
    prefill path has dynamic allocations and cannot be captured.

    This class is created dynamically to inherit from the correct
    vLLM base class at runtime.
    """

    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: object,
        kv_cache_spec: object,
    ) -> AttentionCGSupport:
        """Report single-token-decode CUDA graph support."""
        from vllm.v1.attention.backend import AttentionCGSupport
        return AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE


def _create_metadata_builder_class() -> type:
    """Create a metadata builder that inherits from FlashAttentionMetadataBuilder."""
    _, FlashAttentionMetadataBuilder = _get_backend_classes()

    class _FlashQuantMetadataBuilder(FlashAttentionMetadataBuilder):
        """Metadata builder for FlashQuant with CUDA graph support."""

        @classmethod
        def get_cudagraph_support(
            cls,
            vllm_config: object,
            kv_cache_spec: object,
        ) -> AttentionCGSupport:
            from vllm.v1.attention.backend import AttentionCGSupport
            return AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE

    return _FlashQuantMetadataBuilder


def _create_backend_class() -> type:
    """Create the FlashQuantAttentionBackend class dynamically.

    This avoids import-time dependency on vLLM while creating a proper
    class that inherits from FlashAttentionBackend.
    """
    FlashAttentionBackend, _ = _get_backend_classes()
    MetadataBuilder = _create_metadata_builder_class()

    class FlashQuantAttentionBackend(FlashAttentionBackend):
        """FlashQuant compressed KV cache attention backend.

        Stores nibble-packed TQ4 indices + fp32 norms as raw bytes.
        ``get_kv_cache_shape()`` returns a 3D ``(NB, BS, bytes_per_token)``
        layout matching the packed format.
        """

        forward_includes_kv_cache_update = True

        @classmethod
        def supports_mm_prefix(cls) -> bool:
            """Required for VLMs with bidirectional visual tokens."""
            return True

        @staticmethod
        def get_name() -> str:
            """Must return 'CUSTOM' to match AttentionBackendEnum.CUSTOM."""
            return "CUSTOM"

        @staticmethod
        def get_impl_cls() -> type[AttentionImplBase]:
            """Return FlashQuantAttentionImpl."""
            from flashquant.vllm.impl import get_flashquant_impl_class
            return get_flashquant_impl_class()

        @staticmethod
        def get_builder_cls() -> type[AttentionMetadataBuilder]:
            """Return FlashQuantMetadataBuilder."""
            return MetadataBuilder

        @staticmethod
        def get_kv_cache_shape(
            num_blocks: int,
            block_size: int,
            num_kv_heads: int,
            head_size: int,
            cache_dtype_str: str = "auto",
        ) -> tuple[int, ...]:
            """Packed TQ4 cache: (num_blocks, block_size, total_bytes).

            The last dimension packs K and V data for all heads as raw bytes:
            [K_indices | K_norms | V_indices | V_norms].
            """
            total_bytes = num_kv_heads * tq4_bytes_per_token_kv(head_size)
            return (num_blocks, block_size, total_bytes)

        @staticmethod
        def get_kv_cache_stride_order(
            include_num_layers_dimension: bool = False,
        ) -> tuple[int, ...]:
            """Raise to trigger identity fallback in reshape.

            Our 3D packed layout needs identity ordering. Raising
            NotImplementedError triggers the fallback.
            """
            raise NotImplementedError

    return FlashQuantAttentionBackend


# Module-level caches
_FlashQuantAttentionBackend = None


def get_flashquant_backend_class() -> type:
    """Get the FlashQuantAttentionBackend class (lazy creation).

    Returns:
        The FlashQuantAttentionBackend class.
    """
    global _FlashQuantAttentionBackend
    if _FlashQuantAttentionBackend is None:
        _FlashQuantAttentionBackend = _create_backend_class()
    return _FlashQuantAttentionBackend
