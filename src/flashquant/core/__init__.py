"""FlashQuant core quantization primitives.

Exports:
    CodebookRegistry: Device-aware codebook cache.
    TurboQuantMSE: Stage 1 quantizer (rotation + Lloyd-Max).
    TurboQuantProd: Stage 1 + 2 quantizer (MSE + QJL correction).
    KeyCompressor: Key cache compressor (TurboQuantProd, seed=42).
    ValueCompressor: Value cache compressor (TurboQuantMSE, seed=43).
    nibble_pack: Pack pairs of 4-bit indices into uint8.
    nibble_unpack: Unpack uint8 into pairs of 4-bit indices.
"""

from flashquant.core.codebook import CodebookRegistry
from flashquant.core.compressor import KeyCompressor, ValueCompressor
from flashquant.core.packing import nibble_pack, nibble_unpack
from flashquant.core.quantizer import TurboQuantMSE, TurboQuantProd

__all__ = [
    "CodebookRegistry",
    "KeyCompressor",
    "TurboQuantMSE",
    "TurboQuantProd",
    "ValueCompressor",
    "nibble_pack",
    "nibble_unpack",
]
