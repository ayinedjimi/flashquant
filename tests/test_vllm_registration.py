"""Tests for vLLM backend registration (skip if vllm not installed)."""
from __future__ import annotations

import pytest

vllm = pytest.importorskip("vllm", reason="vLLM not installed")


class TestRegistrationIdempotent:
    """Calling register twice should not raise."""

    def test_registration_idempotent(self) -> None:
        from flashquant.errors import FlashQuantCompatError

        try:
            # If the module exists, import it; if not, skip
            from flashquant.vllm import register_flashquant_backend

            register_flashquant_backend()
            register_flashquant_backend()  # second call should be safe
        except ImportError:
            pytest.skip("flashquant.vllm not yet implemented")
        except FlashQuantCompatError:
            pytest.skip("vLLM version incompatible")


class TestVersionCheckBlocksOld:
    """Old vLLM versions should be rejected."""

    def test_version_check_blocks_old(self) -> None:
        from importlib.metadata import version

        vllm_version = version("vllm")
        major_minor = tuple(int(x) for x in vllm_version.split(".")[:2])

        if major_minor < (0, 18):
            from flashquant.errors import FlashQuantCompatError

            try:
                from flashquant.vllm import register_flashquant_backend
            except ImportError:
                pytest.skip("flashquant.vllm not yet implemented")

            with pytest.raises(FlashQuantCompatError):
                register_flashquant_backend()
        else:
            pytest.skip("vLLM version is >= 0.18, skip old-version test")


class TestSpecType:
    """The attention spec should be a proper subclass."""

    def test_spec_type(self) -> None:
        try:
            from flashquant.vllm import FlashQuantAttentionSpec

            from vllm.v1.kv_cache_interface import FullAttentionSpec

            assert issubclass(FlashQuantAttentionSpec, FullAttentionSpec)
        except ImportError:
            pytest.skip("flashquant.vllm not yet implemented")
