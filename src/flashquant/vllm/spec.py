"""FlashQuant KV cache spec for vLLM page allocation.

Extends vLLM's ``FullAttentionSpec`` with TQ4-specific page sizing.
The ``real_page_size_bytes`` property tells vLLM's block allocator to
provision buffers sized for the packed TQ4 format (3.76x smaller than
FP16), allowing 3.76x more tokens to fit in the same VRAM budget.

This follows the same pattern as vLLM's ``MLAAttentionSpec`` which
overrides page size for FlashMLA's 656-byte format.
"""

from __future__ import annotations

from dataclasses import dataclass

# Lazy imports to avoid requiring vllm at module load time
_FullAttentionSpec = None


def _get_full_attention_spec() -> type:
    """Lazily import FullAttentionSpec from vLLM."""
    global _FullAttentionSpec
    if _FullAttentionSpec is None:
        from vllm.v1.kv_cache_interface import FullAttentionSpec
        _FullAttentionSpec = FullAttentionSpec
    return _FullAttentionSpec


# TQ4 constants
TQ4_NORM_BYTES = 4  # fp32 norm per head


def tq4_bytes_per_token(head_dim: int) -> int:
    """Packed byte count for one token, one KV head, one of K or V.

    Returns:
        head_dim // 2 (nibble-packed indices) + 4 (fp32 norm).
    """
    return head_dim // 2 + TQ4_NORM_BYTES


def tq4_bytes_per_token_kv(head_dim: int) -> int:
    """Total packed bytes per token per KV head (K + V combined)."""
    return 2 * tq4_bytes_per_token(head_dim)


def _create_spec_class() -> type:
    """Dynamically create FlashQuantAttentionSpec that extends FullAttentionSpec.

    This avoids import-time dependency on vLLM while still creating a
    proper dataclass that inherits from FullAttentionSpec.
    """
    FullAttentionSpec = _get_full_attention_spec()

    @dataclass(frozen=True, kw_only=True)
    class FlashQuantAttentionSpec(FullAttentionSpec):  # type: ignore[misc]
        """KV cache spec with TQ4 packed page size.

        Overrides ``real_page_size_bytes`` so the block allocator provisions
        buffers sized for the packed TQ4 format (3.76x smaller than FP16).
        """

        @property
        def real_page_size_bytes(self) -> int:
            return (
                self.block_size
                * self.num_kv_heads
                * tq4_bytes_per_token_kv(self.head_size)
            )

    return FlashQuantAttentionSpec


# Module-level cache for the dynamically created class
_FlashQuantAttentionSpec = None


def get_flashquant_attention_spec() -> type:
    """Get the FlashQuantAttentionSpec class (lazy creation).

    Returns:
        The FlashQuantAttentionSpec dataclass type.
    """
    global _FlashQuantAttentionSpec
    if _FlashQuantAttentionSpec is None:
        _FlashQuantAttentionSpec = _create_spec_class()
    return _FlashQuantAttentionSpec
