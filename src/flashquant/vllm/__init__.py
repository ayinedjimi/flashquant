"""FlashQuant vLLM integration.

Provides a custom attention backend that stores KV cache pages in
TurboQuant 4-bit format (68 bytes/token/head vs 256 bytes FP16 = 3.76x
compression).

Usage:
    The backend registers automatically via the ``vllm.general_plugins``
    entry point when flashquant is installed with the ``vllm`` extra::

        pip install flashquant[vllm]
        vllm serve <model> --attention-backend CUSTOM

    Or register manually before starting vLLM::

        from flashquant.vllm import register_flashquant_backend
        register_flashquant_backend()

Exports:
    register_flashquant_backend: Register FlashQuant as the CUSTOM backend.
    FlashQuantAttentionBackend: Custom attention backend class.
    FlashQuantAttentionImpl: Attention implementation.
    FlashQuantAttentionSpec: KV cache spec with TQ4 page sizes.
"""

from flashquant.vllm.registration import register_flashquant_backend

__all__ = [
    "register_flashquant_backend",
]
