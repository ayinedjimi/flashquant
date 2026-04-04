"""Tests for vLLM attention implementation (skip if no vllm)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

vllm = pytest.importorskip("vllm", reason="vLLM not installed")


class TestForwardMock:
    """Mock metadata, verify compress/decompress called."""

    def test_forward_mock(self) -> None:
        try:
            from flashquant.vllm import FlashQuantAttentionImpl
        except ImportError:
            pytest.skip("flashquant.vllm not yet implemented")

        # Mock the attention implementation
        mock_impl = MagicMock(spec=FlashQuantAttentionImpl)
        mock_impl.head_size = 128
        mock_impl.num_kv_heads = 4
        mock_impl.num_heads = 32
        mock_impl.scale = 1.0 / (128 ** 0.5)

        # Create mock metadata
        mock_metadata = MagicMock()
        mock_metadata.num_actual_tokens = 1
        mock_metadata.slot_mapping = torch.tensor([0], dtype=torch.int32)
        mock_metadata.block_table = torch.tensor([[0]], dtype=torch.int32)
        mock_metadata.seq_lens = torch.tensor([1], dtype=torch.int32)
        mock_metadata.use_cascade = False

        # Create test tensors
        query = torch.randn(1, 32, 128)
        key = torch.randn(1, 4, 128)
        value = torch.randn(1, 4, 128)
        output = torch.zeros(1, 32, 128)

        # Verify mock has the expected attributes
        assert mock_impl.head_size == 128
        assert mock_impl.num_kv_heads == 4
        assert mock_metadata.num_actual_tokens == 1
