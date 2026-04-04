"""FlashQuant kernel dispatch layer.

Provides dispatch functions that try CUDA/Triton kernels via the _C
extension first, falling back to pure PyTorch CPU reference
implementations. All dispatch functions have identical signatures
regardless of the backend.

Exports:
    tq4_compress: TQ4 compression dispatch.
    tq4_decompress: TQ4 decompression dispatch.
    fused_tq_attention: Fused TQ4 Flash Attention dispatch.
    split_k_paged_decode: Split-K paged decode dispatch.

CPU reference implementations:
    compress_reference: Pure PyTorch compress.
    decompress_reference: Pure PyTorch decompress.
    flash_attention_reference: Pure PyTorch flash attention.
    fused_tq_attention_reference: Pure PyTorch fused TQ4 attention.
    paged_decode_reference: Pure PyTorch paged decode.
"""

from flashquant.kernels.attention import fused_tq_attention
from flashquant.kernels.compress import tq4_compress
from flashquant.kernels.cpu_reference import (
    compress_reference,
    decompress_reference,
    flash_attention_reference,
    fused_tq_attention_reference,
    paged_decode_reference,
)
from flashquant.kernels.decompress import tq4_decompress
from flashquant.kernels.paged_decode import split_k_paged_decode

__all__ = [
    "compress_reference",
    "decompress_reference",
    "flash_attention_reference",
    "fused_tq_attention",
    "fused_tq_attention_reference",
    "paged_decode_reference",
    "split_k_paged_decode",
    "tq4_compress",
    "tq4_decompress",
]
