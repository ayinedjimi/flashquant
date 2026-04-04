"""FlashQuant: TurboQuant KV cache compression for LLM inference.

Implements the TurboQuant algorithm (arXiv 2504.19874) for compressing
transformer key-value caches to 4 bits per coordinate with near-zero
accuracy loss. Designed for both HuggingFace transformers (via
CompressedDynamicCache) and vLLM (via the FlashQuant attention backend).

Public API:
    FlashQuantConfig: Immutable configuration dataclass.
    TurboQuantMSE: Stage 1 quantizer (rotation + Lloyd-Max).
    TurboQuantProd: Stage 1 + 2 quantizer (MSE + QJL correction).
    KeyCompressor: Key cache compressor (TurboQuantProd, seed=42).
    ValueCompressor: Value cache compressor (TurboQuantMSE, seed=43).
    CompressedDynamicCache: HuggingFace DynamicCache integration.

Examples:
    HuggingFace integration::

        from transformers import DynamicCache
        from flashquant import CompressedDynamicCache, FlashQuantConfig

        config = FlashQuantConfig(bits=4)
        cache = DynamicCache()
        with CompressedDynamicCache(cache, head_dim=128, bits=config.bits):
            # cache.update is patched inside the block
            ...

    vLLM integration::

        from flashquant.vllm import register_flashquant_backend
        register_flashquant_backend()
        # then start vLLM with --attention-backend CUSTOM

    Standalone quantization::

        from flashquant import TurboQuantMSE

        quantizer = TurboQuantMSE(dim=128, bits=4)
        indices, norms = quantizer.quantize(values)
        reconstructed = quantizer.dequantize(indices, norms)
"""

from flashquant._version import __version__
from flashquant.cache.hf_cache import CompressedDynamicCache
from flashquant.config import FlashQuantConfig
from flashquant.core.compressor import KeyCompressor, ValueCompressor
from flashquant.core.quantizer import TurboQuantMSE, TurboQuantProd

__all__ = [
    "CompressedDynamicCache",
    "FlashQuantConfig",
    "KeyCompressor",
    "TurboQuantMSE",
    "TurboQuantProd",
    "ValueCompressor",
    "__version__",
]
