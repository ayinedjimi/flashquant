"""FlashQuant exception hierarchy.

All FlashQuant-specific exceptions inherit from :class:`FlashQuantError`,
making it easy to catch all package errors in a single handler while
still allowing fine-grained handling of config, compatibility, and cache
errors individually.

Examples:
    Catch any FlashQuant error::

        try:
            config.validate()
        except FlashQuantError as exc:
            logger.error("FlashQuant error: %s", exc)

    Catch only config errors::

        try:
            config.validate()
        except FlashQuantConfigError as exc:
            logger.error("Bad config: %s", exc)
"""


class FlashQuantError(Exception):
    """Base exception for all FlashQuant errors."""


class FlashQuantConfigError(FlashQuantError):
    """Raised when FlashQuantConfig validation fails.

    Examples:
        Invalid bit-width::

            FlashQuantConfigError("bits must be 2, 3, or 4, got 7")
    """


class FlashQuantCompatError(FlashQuantError):
    """Raised when a compatibility check fails.

    Typically caused by an unsupported vLLM version, missing CUDA
    capability, or incompatible transformers version.

    Examples:
        vLLM too old::

            FlashQuantCompatError(
                "vLLM 0.17.0 is not supported. "
                "FlashQuant requires vLLM >= 0.18.0, < 0.22.0. "
                "Run: pip install 'vllm>=0.18.0,<0.22.0'"
            )
    """


class FlashQuantCacheError(FlashQuantError):
    """Raised for KV cache lifecycle errors.

    Examples include double-wrapping a cache, accessing a layer that
    has not been initialized, or buffer overflow.

    Examples:
        Double-wrap detection::

            FlashQuantCacheError(
                "Cache is already wrapped by FlashQuant. "
                "Call restore() on the existing wrapper first."
            )
    """
