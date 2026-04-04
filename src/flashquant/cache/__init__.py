"""FlashQuant KV cache integration.

Exports:
    CompressedBuffer: Pre-allocated ring buffer for O(1) append.
    CompressedDynamicCache: HuggingFace DynamicCache integration with
        real VRAM savings via compressed index storage.
"""

from flashquant.cache.buffer_pool import CompressedBuffer
from flashquant.cache.hf_cache import CompressedDynamicCache

__all__ = [
    "CompressedBuffer",
    "CompressedDynamicCache",
]
