"""Tests for FlashQuantConfig."""
from __future__ import annotations

import os
from dataclasses import FrozenInstanceError
from unittest import mock

import pytest

from flashquant.config import FlashQuantConfig
from flashquant.errors import FlashQuantConfigError


class TestDefaultConfig:
    """Default values should be correct."""

    def test_default_config(self) -> None:
        cfg = FlashQuantConfig()
        assert cfg.bits == 4
        assert cfg.key_seed == 42
        assert cfg.value_seed == 43
        assert cfg.use_fused_paged is True
        assert cfg.use_cuda_graphs is True
        assert cfg.max_seq_len == 32768


class TestFromEnv:
    """Should read FLASHQUANT_* environment variables."""

    def test_from_env_bits(self) -> None:
        with mock.patch.dict(os.environ, {"FLASHQUANT_BITS": "3"}):
            cfg = FlashQuantConfig.from_env()
            assert cfg.bits == 3

    def test_from_env_seeds(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"FLASHQUANT_KEY_SEED": "100", "FLASHQUANT_VALUE_SEED": "200"},
        ):
            cfg = FlashQuantConfig.from_env()
            assert cfg.key_seed == 100
            assert cfg.value_seed == 200

    def test_from_env_booleans(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "FLASHQUANT_USE_FUSED_PAGED": "false",
                "FLASHQUANT_USE_CUDA_GRAPHS": "0",
            },
        ):
            cfg = FlashQuantConfig.from_env()
            assert cfg.use_fused_paged is False
            assert cfg.use_cuda_graphs is False

    def test_from_env_max_seq_len(self) -> None:
        with mock.patch.dict(os.environ, {"FLASHQUANT_MAX_SEQ_LEN": "16384"}):
            cfg = FlashQuantConfig.from_env()
            assert cfg.max_seq_len == 16384

    def test_from_env_defaults(self) -> None:
        """Empty environment should use defaults."""
        with mock.patch.dict(os.environ, {}, clear=True):
            cfg = FlashQuantConfig.from_env()
            assert cfg.bits == 4
            assert cfg.key_seed == 42
            assert cfg.value_seed == 43

    def test_from_env_invalid_int(self) -> None:
        with mock.patch.dict(os.environ, {"FLASHQUANT_BITS": "abc"}):
            with pytest.raises(FlashQuantConfigError, match="Cannot parse"):
                FlashQuantConfig.from_env()


class TestValidateInvalidBits:
    """bits=0, bits=1, bits=5, bits=9 should fail validation."""

    @pytest.mark.parametrize("bad_bits", [0, 1, 5, 9])
    def test_validate_invalid_bits(self, bad_bits: int) -> None:
        cfg = FlashQuantConfig(bits=bad_bits)
        with pytest.raises(FlashQuantConfigError, match="bits must be"):
            cfg.validate()


class TestValidateMatchingSeeds:
    """Same key_seed and value_seed should fail validation."""

    def test_validate_matching_seeds(self) -> None:
        cfg = FlashQuantConfig(key_seed=42, value_seed=42)
        with pytest.raises(FlashQuantConfigError, match="key_seed and value_seed"):
            cfg.validate()


class TestValidateMaxSeqLen:
    """Invalid max_seq_len values should fail."""

    def test_validate_zero_seq_len(self) -> None:
        cfg = FlashQuantConfig(max_seq_len=0)
        with pytest.raises(FlashQuantConfigError, match="positive"):
            cfg.validate()

    def test_validate_negative_seq_len(self) -> None:
        cfg = FlashQuantConfig(max_seq_len=-100)
        with pytest.raises(FlashQuantConfigError, match="positive"):
            cfg.validate()

    def test_validate_excessive_seq_len(self) -> None:
        cfg = FlashQuantConfig(max_seq_len=2_000_000)
        with pytest.raises(FlashQuantConfigError, match="1M token"):
            cfg.validate()


class TestFrozen:
    """Config should be immutable after creation."""

    def test_frozen(self) -> None:
        cfg = FlashQuantConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.bits = 3  # type: ignore[misc]

    def test_frozen_seed(self) -> None:
        cfg = FlashQuantConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.key_seed = 99  # type: ignore[misc]
