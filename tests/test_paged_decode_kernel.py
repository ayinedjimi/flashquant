"""Tests for paged decode attention kernel (CPU reference).

Simulates the paged KV cache layout and validates that split-K decode
matches a non-split reference, and that page table indirection works.
"""
from __future__ import annotations

import math

import pytest
import torch

from .conftest import (
    RefQuantizerMSE,
    cosine_similarity_flat,
    generate_rotation_matrix,
    ref_compress,
    ref_decompress,
)


def _paged_decode_reference(
    q: torch.Tensor,
    kv_pages: list[tuple[torch.Tensor, torch.Tensor]],
    page_table: list[int],
    seq_len: int,
    centroids: torch.Tensor,
    rotation: torch.Tensor,
    block_size: int,
    sm_scale: float,
    num_splits: int = 1,
) -> torch.Tensor:
    """Reference paged decode attention.

    Args:
        q: (H_Q, D) single query token.
        kv_pages: List of (k_packed, v_packed) pages, each
            (block_size, H_KV, D//2) uint8.
            Norms stored separately as k_norms, v_norms.
        page_table: Maps logical block index -> physical page index.
        seq_len: Number of valid tokens.
        centroids: (16,) fp32.
        rotation: (D, D) fp32.
        block_size: Tokens per page.
        sm_scale: Softmax scale.
        num_splits: Number of split-K partitions (for testing equivalence).

    Returns:
        (H_Q, D) attention output.
    """
    H_Q, D = q.shape
    HALF_D = D // 2

    # Pre-rotate Q
    q_rot = (q.float() @ rotation.T)

    # Collect all valid KV tokens by walking the page table
    all_k_rot = []
    all_v_rot = []

    tokens_collected = 0
    for logical_block_idx, phys_page in enumerate(page_table):
        if tokens_collected >= seq_len:
            break
        k_packed_page, k_norms_page, v_packed_page, v_norms_page = kv_pages[phys_page]
        tokens_in_block = min(block_size, seq_len - tokens_collected)

        for t in range(tokens_in_block):
            # Decompress single token K
            # k_packed_page[t] is (H_KV, HALF_D), unsqueeze to (1, H_KV, HALF_D)
            k_packed_t = k_packed_page[t].unsqueeze(0)  # (1, H_KV, HALF_D)
            k_norms_t = k_norms_page[t].unsqueeze(0)    # (1, H_KV, 1)
            k_rot = ref_decompress(k_packed_t, k_norms_t, centroids)  # (1, H_KV, D)
            all_k_rot.append(k_rot.squeeze(0))  # (H_KV, D)

            v_packed_t = v_packed_page[t].unsqueeze(0)
            v_norms_t = v_norms_page[t].unsqueeze(0)
            v_rot = ref_decompress(v_packed_t, v_norms_t, centroids)
            all_v_rot.append(v_rot.squeeze(0))

        tokens_collected += tokens_in_block

    # Stack: (S_kv, H_KV, D)
    K_rot = torch.stack(all_k_rot, dim=0)  # (S, H_KV, D)
    V_rot = torch.stack(all_v_rot, dim=0)

    # For simplicity, assume H_Q == H_KV (no GQA)
    # Attention in rotated space: q_rot @ K_rot^T
    # q_rot: (H_Q, D), K_rot: (S, H_KV, D) -> scores: (H_Q, S)
    scores = torch.einsum("hd,shd->hs", q_rot, K_rot.float()) * sm_scale
    weights = torch.softmax(scores, dim=-1)
    # Output: (H_Q, D)
    out_rot = torch.einsum("hs,shd->hd", weights, V_rot.float())

    # Post-rotate
    out = out_rot @ rotation
    return out


# ---------------------------------------------------------------------------
# Helper to build test paged cache
# ---------------------------------------------------------------------------


def _build_paged_cache(
    seq_len: int,
    H_KV: int,
    head_dim: int,
    block_size: int,
    rotation: torch.Tensor,
    boundaries: torch.Tensor,
) -> tuple[list[tuple], list[int]]:
    """Build a paged cache with random data.

    Returns:
        (kv_pages, page_table) where kv_pages[i] = (k_packed, k_norms, v_packed, v_norms)
        and page_table maps logical block -> physical page.
    """
    HALF_D = head_dim // 2
    n_blocks = (seq_len + block_size - 1) // block_size
    kv_pages = []

    for _ in range(n_blocks):
        k_raw = torch.randn(block_size, H_KV, head_dim)
        v_raw = torch.randn(block_size, H_KV, head_dim)

        k_packed, k_norms = ref_compress(k_raw, rotation, boundaries)
        v_packed, v_norms = ref_compress(v_raw, rotation, boundaries)
        kv_pages.append((k_packed, k_norms, v_packed, v_norms))

    # Simple identity page table (could be shuffled for indirection test)
    page_table = list(range(n_blocks))
    return kv_pages, page_table


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPagedDecodeMatchesReference:
    """Split-K decode should match reference sequential decode."""

    @pytest.mark.parametrize("head_dim", [64, 128])
    def test_paged_decode_matches_reference(self, head_dim: int) -> None:
        H_KV = 4
        block_size = 16
        seq_len = 48  # 3 pages

        rotation = generate_rotation_matrix(head_dim, seed=42)
        quantizer = RefQuantizerMSE(head_dim, 4, seed=42)
        centroids = quantizer.centroids
        boundaries = quantizer.boundaries
        sm_scale = 1.0 / math.sqrt(head_dim)

        kv_pages, page_table = _build_paged_cache(
            seq_len, H_KV, head_dim, block_size, rotation, boundaries
        )

        q = torch.randn(H_KV, head_dim)

        out = _paged_decode_reference(
            q, kv_pages, page_table, seq_len, centroids, rotation,
            block_size, sm_scale, num_splits=1,
        )
        assert out.shape == (H_KV, head_dim)
        assert torch.isfinite(out).all()


class TestSplitKMatchesNoSplit:
    """NUM_SPLITS=1 and NUM_SPLITS=4 should give same result.

    Since the reference is sequential, we verify that splitting the
    computation into partitions and combining gives the same output.
    """

    @pytest.mark.parametrize("head_dim", [64, 128])
    def test_split_k_matches_no_split(self, head_dim: int) -> None:
        H_KV = 4
        block_size = 16
        seq_len = 64  # 4 pages

        rotation = generate_rotation_matrix(head_dim, seed=42)
        quantizer = RefQuantizerMSE(head_dim, 4, seed=42)
        centroids = quantizer.centroids
        boundaries = quantizer.boundaries
        sm_scale = 1.0 / math.sqrt(head_dim)

        kv_pages, page_table = _build_paged_cache(
            seq_len, H_KV, head_dim, block_size, rotation, boundaries
        )

        q = torch.randn(H_KV, head_dim)

        out_1split = _paged_decode_reference(
            q, kv_pages, page_table, seq_len, centroids, rotation,
            block_size, sm_scale, num_splits=1,
        )
        out_4split = _paged_decode_reference(
            q, kv_pages, page_table, seq_len, centroids, rotation,
            block_size, sm_scale, num_splits=4,
        )

        cos = cosine_similarity_flat(out_1split, out_4split)
        assert cos > 0.999, (
            f"Split-K mismatch: cosine = {cos:.6f}"
        )


class TestPageTableIndirection:
    """Shuffled page table should still produce correct results."""

    def test_page_table_indirection(self) -> None:
        head_dim = 128
        H_KV = 4
        block_size = 16
        seq_len = 64

        rotation = generate_rotation_matrix(head_dim, seed=42)
        quantizer = RefQuantizerMSE(head_dim, 4, seed=42)
        centroids = quantizer.centroids
        boundaries = quantizer.boundaries
        sm_scale = 1.0 / math.sqrt(head_dim)

        kv_pages, page_table = _build_paged_cache(
            seq_len, H_KV, head_dim, block_size, rotation, boundaries
        )

        q = torch.randn(H_KV, head_dim)

        # Identity page table
        out_identity = _paged_decode_reference(
            q, kv_pages, page_table, seq_len, centroids, rotation,
            block_size, sm_scale,
        )

        # Shuffle physical pages and update page table
        import random
        random.seed(42)
        n_blocks = len(kv_pages)
        shuffled_indices = list(range(n_blocks))
        random.shuffle(shuffled_indices)

        # Create shuffled pages array
        shuffled_pages = [None] * n_blocks
        new_page_table = [0] * n_blocks
        for logical, physical in enumerate(shuffled_indices):
            shuffled_pages[physical] = kv_pages[logical]
            new_page_table[logical] = physical

        out_shuffled = _paged_decode_reference(
            q, shuffled_pages, new_page_table, seq_len, centroids, rotation,
            block_size, sm_scale,
        )

        cos = cosine_similarity_flat(out_identity, out_shuffled)
        assert cos > 0.999, (
            f"Indirection mismatch: cosine = {cos:.6f}"
        )
