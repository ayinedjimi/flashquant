"""Tests for long-context KV cache compression quality.

Validates that compression quality does not degrade with sequence length.
"""
from __future__ import annotations

import pytest
import torch

from .conftest import RefQuantizerMSE, cosine_similarity_flat


def _compress_decompress_batch(
    quantizer: RefQuantizerMSE, x: torch.Tensor
) -> torch.Tensor:
    """Compress and decompress a batch of vectors."""
    indices, norms = quantizer.quantize(x)
    return quantizer.dequantize(indices, norms)


class TestTokenCounts:
    """Various sequence lengths should work without errors."""

    @pytest.mark.parametrize("n_tokens", [256, 1024, 4096])
    def test_token_counts(self, n_tokens: int) -> None:
        head_dim = 128
        q = RefQuantizerMSE(head_dim, 4, seed=42)
        x = torch.randn(n_tokens, head_dim)
        indices, norms = q.quantize(x)
        reconstructed = q.dequantize(indices, norms)
        assert reconstructed.shape == x.shape

    def test_4k_tokens(self) -> None:
        head_dim = 128
        q = RefQuantizerMSE(head_dim, 4, seed=42)
        x = torch.randn(4096, head_dim)
        reconstructed = _compress_decompress_batch(q, x)
        cos = cosine_similarity_flat(x, reconstructed)
        assert cos > 0.95

    def test_16k_tokens(self) -> None:
        head_dim = 128
        q = RefQuantizerMSE(head_dim, 4, seed=42)
        x = torch.randn(16384, head_dim)
        reconstructed = _compress_decompress_batch(q, x)
        cos = cosine_similarity_flat(x, reconstructed)
        assert cos > 0.95

    @pytest.mark.slow
    def test_32k_tokens(self) -> None:
        head_dim = 128
        q = RefQuantizerMSE(head_dim, 4, seed=42)
        x = torch.randn(32768, head_dim)
        reconstructed = _compress_decompress_batch(q, x)
        cos = cosine_similarity_flat(x, reconstructed)
        assert cos > 0.95


class TestNoPrecisionDrift:
    """Cosine similarity should not degrade with sequence length.

    If the quantizer has a systematic bias that accumulates, longer
    sequences would show lower cosine similarity. We check that the
    quality at 16K tokens is not worse than at 256 tokens by more
    than a small tolerance.
    """

    def test_no_precision_drift(self) -> None:
        head_dim = 128
        q = RefQuantizerMSE(head_dim, 4, seed=42)

        # Short sequence
        torch.manual_seed(42)
        x_short = torch.randn(256, head_dim)
        recon_short = _compress_decompress_batch(q, x_short)
        cos_short = cosine_similarity_flat(x_short, recon_short)

        # Long sequence
        torch.manual_seed(42)
        x_long = torch.randn(16384, head_dim)
        recon_long = _compress_decompress_batch(q, x_long)
        cos_long = cosine_similarity_flat(x_long, recon_long)

        # Quality should not degrade by more than 0.02
        assert cos_long >= cos_short - 0.02, (
            f"Precision drift detected: cos_short={cos_short:.4f}, "
            f"cos_long={cos_long:.4f}, drift={cos_short - cos_long:.4f}"
        )
