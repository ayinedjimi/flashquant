"""vLLM version compatibility checking.

FlashQuant supports a specific range of vLLM versions. This module
provides a clear, actionable error message when the installed vLLM
version is outside the supported range, instead of cryptic import
errors or silent misbehavior.

Supported range: vLLM >= 0.18.0, < 0.22.0
"""

from __future__ import annotations

import logging

from flashquant.errors import FlashQuantCompatError

logger = logging.getLogger(__name__)

VLLM_MIN = "0.18.0"
VLLM_MAX = "0.22.0"


def _parse_version(version_str: str) -> tuple[int, ...]:
    """Parse a version string into a tuple of integers.

    Handles version strings like "0.18.0", "0.20.1.dev0", etc.
    Non-numeric suffixes are stripped.

    Args:
        version_str: Version string (e.g., "0.18.0").

    Returns:
        Tuple of integers (e.g., (0, 18, 0)).
    """
    parts = []
    for part in version_str.split("."):
        # Strip non-numeric suffixes (e.g., "dev0", "rc1")
        numeric = ""
        for ch in part:
            if ch.isdigit():
                numeric += ch
            else:
                break
        if numeric:
            parts.append(int(numeric))
    return tuple(parts)


def check_vllm_version() -> str:
    """Check that the installed vLLM version is within the supported range.

    Returns:
        The detected vLLM version string.

    Raises:
        FlashQuantCompatError: If vLLM is not installed, or if the
            version is outside [VLLM_MIN, VLLM_MAX).
    """
    try:
        import vllm  # type: ignore[import-untyped]

        version_str = getattr(vllm, "__version__", "0.0.0")
    except ImportError:
        raise FlashQuantCompatError(
            "vLLM is not installed. FlashQuant's vLLM backend requires "
            f"vLLM >= {VLLM_MIN}, < {VLLM_MAX}. "
            f"Run: pip install 'vllm>={VLLM_MIN},<{VLLM_MAX}'"
        ) from None

    version = _parse_version(version_str)
    min_version = _parse_version(VLLM_MIN)
    max_version = _parse_version(VLLM_MAX)

    if version < min_version:
        raise FlashQuantCompatError(
            f"vLLM {version_str} is too old. "
            f"FlashQuant requires vLLM >= {VLLM_MIN}, < {VLLM_MAX}. "
            f"Run: pip install 'vllm>={VLLM_MIN},<{VLLM_MAX}'"
        )

    if version >= max_version:
        raise FlashQuantCompatError(
            f"vLLM {version_str} is too new (untested). "
            f"FlashQuant requires vLLM >= {VLLM_MIN}, < {VLLM_MAX}. "
            f"Run: pip install 'vllm>={VLLM_MIN},<{VLLM_MAX}'"
        )

    logger.info("vLLM %s detected (supported range: [%s, %s))",
                version_str, VLLM_MIN, VLLM_MAX)
    return version_str
