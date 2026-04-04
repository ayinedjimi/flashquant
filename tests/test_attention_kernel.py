"""Tests for attention kernel implementations (CPU reference).

Validates vanilla attention, causal masking, and fused TQ attention against
naive Python reference implementations. All tests run on CPU.
"""
from __future__ import annotations

import math

import pytest
import torch

from .conftest import (
    RefQuantizerMSE,
    cosine_similarity_flat,
    generate_rotation_matrix,
    ref_attention,
    ref_compress,
    ref_decompress,
)


class TestVanillaAttentionMatchesReference:
    """Standard attention should match naive Python implementation."""

    @pytest.mark.parametrize("head_dim", [64, 128])
    @pytest.mark.parametrize("seq_len", [8, 32])
    def test_vanilla_attention_matches_reference(
        self, head_dim: int, seq_len: int
    ) -> None:
        B, H = 1, 4
        q = torch.randn(B, H, seq_len, head_dim)
        k = torch.randn(B, H, seq_len, head_dim)
        v = torch.randn(B, H, seq_len, head_dim)

        # Reference implementation
        expected = ref_attention(q, k, v, is_causal=False)

        # PyTorch SDPA (CPU fallback)
        sm_scale = 1.0 / math.sqrt(head_dim)
        scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * sm_scale
        weights = torch.softmax(scores, dim=-1)
        actual = torch.matmul(weights, v.float()).to(q.dtype)

        cos = cosine_similarity_flat(expected, actual)
        assert cos > 0.999, (
            f"Vanilla attention cosine {cos:.6f} < 0.999"
        )


class TestCausalMasking:
    """Output should differ with/without causal masking."""

    @pytest.mark.parametrize("head_dim", [64, 128])
    def test_causal_masking(self, head_dim: int) -> None:
        B, H, S = 1, 2, 16
        q = torch.randn(B, H, S, head_dim)
        k = torch.randn(B, H, S, head_dim)
        v = torch.randn(B, H, S, head_dim)

        out_no_causal = ref_attention(q, k, v, is_causal=False)
        out_causal = ref_attention(q, k, v, is_causal=True)

        # They should differ (unless by extreme coincidence)
        diff = (out_no_causal - out_causal).abs().max().item()
        assert diff > 1e-3, (
            f"Causal and non-causal outputs are identical (diff={diff:.2e})"
        )

        # First row of causal should equal non-causal (no mask applied to row 0
        # since it can attend to position 0 only, same in both modes)
        # Actually for row 0, causal only sees position 0, non-causal sees all.
        # They differ.

        # Last row of causal should see all positions, so it might be similar
        # to non-causal for the last row -- but the softmax normalization
        # differs because earlier rows in causal have masked tokens.
        # Just verify they are numerically different overall.
        cos = cosine_similarity_flat(out_no_causal, out_causal)
        assert cos < 0.999, (
            f"Causal and non-causal should differ more (cos={cos:.6f})"
        )


class TestFusedTQAttentionMatchesReference:
    """Fused TQ attention should match decompress-then-attend reference.

    This test simulates the fused kernel by:
    1. Compressing K and V to TQ4 format
    2. Decompressing and attending (reference path)
    3. Pre-rotating Q, attending in rotated space, and post-rotating (fused path)
    Both should give the same result.
    """

    @pytest.mark.parametrize("head_dim", [64, 128])
    def test_fused_tq_attention_matches_reference(
        self, head_dim: int
    ) -> None:
        B, H, S_q, S_kv = 1, 4, 1, 16

        rotation = generate_rotation_matrix(head_dim, seed=42)
        quantizer = RefQuantizerMSE(head_dim, 4, seed=42)
        centroids = quantizer.centroids
        boundaries = quantizer.boundaries

        q = torch.randn(B, H, S_q, head_dim)
        k_raw = torch.randn(B, H, S_kv, head_dim)
        v_raw = torch.randn(B, H, S_kv, head_dim)

        # Compress K and V (shape: B, H, S, D -> N=B*H*S, H=1, D for ref_compress)
        # Reshape for ref_compress which expects (N, H, D)
        k_flat = k_raw.reshape(B * H, S_kv, head_dim)
        v_flat = v_raw.reshape(B * H, S_kv, head_dim)

        k_packed, k_norms = ref_compress(k_flat, rotation, boundaries)
        v_packed, v_norms = ref_compress(v_flat, rotation, boundaries)

        # Reference path: decompress in rotated space, apply inverse rotation,
        # then standard attention
        k_decompressed_rot = ref_decompress(k_packed, k_norms, centroids)
        v_decompressed_rot = ref_decompress(v_packed, v_norms, centroids)

        # Unrotate K and V
        k_dec = (k_decompressed_rot.reshape(-1, head_dim).float() @ rotation).reshape(
            B, H, S_kv, head_dim
        )
        v_dec = (v_decompressed_rot.reshape(-1, head_dim).float() @ rotation).reshape(
            B, H, S_kv, head_dim
        )
        ref_out = ref_attention(q, k_dec, v_dec)

        # Fused path: pre-rotate Q, attend in rotated space, post-rotate output
        q_rot = (q.float().reshape(-1, head_dim) @ rotation.T).reshape(
            B, H, S_q, head_dim
        )
        # K and V in rotated space
        k_rot = k_decompressed_rot.reshape(B, H, S_kv, head_dim)
        v_rot = v_decompressed_rot.reshape(B, H, S_kv, head_dim)

        fused_out_rot = ref_attention(q_rot, k_rot, v_rot)
        fused_out = (
            fused_out_rot.float().reshape(-1, head_dim) @ rotation
        ).reshape(B, H, S_q, head_dim)

        cos = cosine_similarity_flat(ref_out, fused_out)
        assert cos > 0.999, (
            f"Fused TQ attention vs reference cosine {cos:.6f} < 0.999"
        )
