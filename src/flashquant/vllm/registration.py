"""FlashQuant backend registration for vLLM.

Performs the following steps:

1. Version check: ensures vLLM is within the supported range.
2. Backend registration: registers FlashQuantAttentionBackend as CUSTOM.
3. Spec manager map: adds FlashQuantAttentionSpec so vLLM's block
   allocator can handle TQ4 page sizes.
4. Monkey-patch: replaces Attention.get_kv_cache_spec to return
   FlashQuantAttentionSpec (with dtype=uint8) instead of FullAttentionSpec.

This is called automatically by the ``vllm.general_plugins`` entry
point, or manually before starting vLLM.
"""

from __future__ import annotations

import logging

from flashquant.errors import FlashQuantCompatError
from flashquant.vllm.compat import check_vllm_version

logger = logging.getLogger(__name__)

_original_get_kv_cache_spec = None
_registered = False


def register_flashquant_backend() -> None:
    """Register FlashQuant as the CUSTOM attention backend in vLLM.

    In addition to registering the backend class, this monkey-patches
    ``Attention.get_kv_cache_spec`` so that decoder attention layers
    return ``FlashQuantAttentionSpec`` (with ``dtype=torch.uint8``
    and TQ4-sized pages) instead of the standard ``FullAttentionSpec``.

    Can be called multiple times safely (idempotent).

    Raises:
        FlashQuantCompatError: If vLLM version is outside the supported
            range, or if required vLLM internals are missing.

    Usage::

        from flashquant.vllm import register_flashquant_backend
        register_flashquant_backend()
        # then start vLLM with --attention-backend CUSTOM
    """
    global _original_get_kv_cache_spec, _registered  # noqa: PLW0603

    if _registered:
        logger.debug("FlashQuant backend already registered, skipping")
        return

    # Step 1: Version check
    version_str = check_vllm_version()

    # Step 2: Import vLLM internals (guarded by version check)
    try:
        import torch
        from vllm.v1.attention.backends.registry import (
            AttentionBackendEnum,
            register_backend,
        )
        from vllm.v1.kv_cache_interface import FullAttentionSpec
    except ImportError as exc:
        raise FlashQuantCompatError(
            f"Failed to import vLLM internals (version {version_str}). "
            f"This may indicate an incompatible vLLM build. "
            f"Error: {exc}"
        ) from exc

    # Step 3: Register the backend class
    from flashquant.vllm.backend import get_flashquant_backend_class

    BackendClass = get_flashquant_backend_class()
    register_backend(
        AttentionBackendEnum.CUSTOM,
        f"{BackendClass.__module__}.{BackendClass.__qualname__}",
    )

    # Step 4: Register FlashQuantAttentionSpec in the KV cache manager map
    from flashquant.vllm.spec import get_flashquant_attention_spec

    FlashQuantAttentionSpec = get_flashquant_attention_spec()

    try:
        from vllm.v1.core.single_type_kv_cache_manager import spec_manager_map

        if FlashQuantAttentionSpec not in spec_manager_map:
            spec_manager_map[FlashQuantAttentionSpec] = spec_manager_map[
                FullAttentionSpec
            ]
    except ImportError:
        logger.warning(
            "Could not register FlashQuantAttentionSpec in spec_manager_map. "
            "vLLM may not allocate TQ4-sized pages correctly."
        )

    # Step 5: Monkey-patch Attention.get_kv_cache_spec
    try:
        from dataclasses import fields as dc_fields

        from vllm.model_executor.layers.attention.attention import Attention

        if _original_get_kv_cache_spec is None:
            _original_get_kv_cache_spec = Attention.get_kv_cache_spec

        def _flashquant_get_kv_cache_spec(self, vllm_config):
            """Patched get_kv_cache_spec that returns FlashQuantAttentionSpec."""
            spec = _original_get_kv_cache_spec(self, vllm_config)
            if isinstance(spec, FullAttentionSpec) and not isinstance(
                spec, FlashQuantAttentionSpec
            ):
                kwargs = {
                    f.name: getattr(spec, f.name) for f in dc_fields(spec)
                }
                kwargs["dtype"] = torch.uint8
                return FlashQuantAttentionSpec(**kwargs)
            return spec

        Attention.get_kv_cache_spec = _flashquant_get_kv_cache_spec  # type: ignore[assignment]
    except ImportError:
        logger.warning(
            "Could not monkey-patch Attention.get_kv_cache_spec. "
            "You may need to configure TQ4 cache spec manually."
        )

    _registered = True
    logger.info(
        "FlashQuant attention backend registered as CUSTOM "
        "(packed TQ4 cache, vLLM %s)",
        version_str,
    )
