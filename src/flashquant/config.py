"""FlashQuant configuration.

Provides a frozen dataclass :class:`FlashQuantConfig` that holds all
tunable parameters for the TurboQuant quantization pipeline. The config
is immutable after construction to prevent accidental mutation during
model serving.

Key design decisions:

- **Separate K/V seeds (42 vs 43)**: Keys and values use independent
  codebooks. Sharing a seed would cause K and V to use identical
  rotation matrices, which is mathematically valid but wastes the
  opportunity for independent error decorrelation.

- **from_env()**: Reads ``FLASHQUANT_*`` environment variables so that
  configuration can be injected without code changes (e.g., in
  Docker/Kubernetes deployments).

Examples:
    Default config::

        cfg = FlashQuantConfig()
        cfg.bits  # 4
        cfg.key_seed  # 42
        cfg.value_seed  # 43

    Override from environment::

        os.environ["FLASHQUANT_BITS"] = "3"
        cfg = FlashQuantConfig.from_env()
        cfg.bits  # 3
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from flashquant.errors import FlashQuantConfigError


@dataclass(frozen=True)
class FlashQuantConfig:
    """Immutable configuration for FlashQuant KV cache compression.

    Attributes:
        bits: Quantization bit-width per coordinate. Must be 2, 3, or 4.
        key_seed: Random seed for key codebook/rotation matrix.
        value_seed: Random seed for value codebook/rotation matrix.
            Must differ from ``key_seed`` for independent codebooks.
        use_fused_paged: Enable fused paged TQ4 decode kernel when
            available. Falls back to decompress-all path when False
            or when the kernel is not compiled.
        use_cuda_graphs: Enable CUDA graph capture for decode steps.
        max_seq_len: Maximum supported sequence length. Controls
            pre-allocated buffer sizes.
    """

    bits: int = 4
    key_seed: int = 42
    value_seed: int = 43
    use_fused_paged: bool = True
    use_cuda_graphs: bool = True
    max_seq_len: int = 32768

    def validate(self) -> None:
        """Validate configuration values.

        Raises:
            FlashQuantConfigError: If any parameter is out of range or
                logically inconsistent.
        """
        if self.bits not in (2, 3, 4):
            raise FlashQuantConfigError(
                f"bits must be 2, 3, or 4, got {self.bits}"
            )

        if self.key_seed == self.value_seed:
            raise FlashQuantConfigError(
                f"key_seed and value_seed must differ for independent "
                f"codebooks, both are {self.key_seed}. "
                f"Use key_seed=42, value_seed=43 (the defaults)."
            )

        if self.max_seq_len <= 0:
            raise FlashQuantConfigError(
                f"max_seq_len must be positive, got {self.max_seq_len}"
            )

        if self.max_seq_len > 1_048_576:
            raise FlashQuantConfigError(
                f"max_seq_len={self.max_seq_len} exceeds the 1M token "
                f"limit. Pre-allocated buffers would consume excessive "
                f"memory."
            )

    @classmethod
    def from_env(cls) -> FlashQuantConfig:
        """Create a config from ``FLASHQUANT_*`` environment variables.

        Reads the following variables (all optional, defaults apply):

        - ``FLASHQUANT_BITS``: int (2, 3, or 4)
        - ``FLASHQUANT_KEY_SEED``: int
        - ``FLASHQUANT_VALUE_SEED``: int
        - ``FLASHQUANT_USE_FUSED_PAGED``: "1"/"true"/"yes" for True
        - ``FLASHQUANT_USE_CUDA_GRAPHS``: "1"/"true"/"yes" for True
        - ``FLASHQUANT_MAX_SEQ_LEN``: int

        Returns:
            A validated FlashQuantConfig instance.

        Raises:
            FlashQuantConfigError: If environment values cannot be
                parsed or fail validation.
        """

        def _parse_bool(key: str, default: bool) -> bool:
            val = os.environ.get(key, "").strip().lower()
            if not val:
                return default
            return val in ("1", "true", "yes")

        def _parse_int(key: str, default: int) -> int:
            val = os.environ.get(key, "").strip()
            if not val:
                return default
            try:
                return int(val)
            except ValueError:
                raise FlashQuantConfigError(
                    f"Cannot parse {key}={val!r} as integer"
                ) from None

        config = cls(
            bits=_parse_int("FLASHQUANT_BITS", cls.bits),
            key_seed=_parse_int("FLASHQUANT_KEY_SEED", cls.key_seed),
            value_seed=_parse_int("FLASHQUANT_VALUE_SEED", cls.value_seed),
            use_fused_paged=_parse_bool(
                "FLASHQUANT_USE_FUSED_PAGED", cls.use_fused_paged
            ),
            use_cuda_graphs=_parse_bool(
                "FLASHQUANT_USE_CUDA_GRAPHS", cls.use_cuda_graphs
            ),
            max_seq_len=_parse_int("FLASHQUANT_MAX_SEQ_LEN", cls.max_seq_len),
        )
        config.validate()
        return config
