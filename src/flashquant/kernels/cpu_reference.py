"""CPU reference implementations (pure PyTorch) for ALL kernels.

These implementations are numerically correct but not performance-
optimized. They serve as:

1. Correctness baselines for Triton/CUDA kernel testing.
2. Fallback implementations when CUDA is not available.
3. Documentation of the exact algorithm for each kernel.

All functions operate in float32 for numerical precision, casting
outputs to the requested dtype at the end.
"""

from __future__ import annotations

import math
from typing import Optional

import torch


def compress_reference(
    x: torch.Tensor,
    rotation: torch.Tensor,
    boundaries: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure PyTorch TQ4 compression: norm + rotate + bucketize + pack.

    Args:
        x: Input vectors ``(N, H, D)`` in any float dtype.
        rotation: Orthogonal rotation matrix ``(D, D)`` fp32.
        boundaries: Quantization boundaries ``(2^bits - 1,)`` fp32.

    Returns:
        Tuple of (packed, norms):
            - packed: ``(N, H, D//2)`` uint8 nibble-packed indices.
            - norms: ``(N, H, 1)`` fp32 per-vector norms.
    """
    N, H, D = x.shape
    half_D = D // 2
    flat = x.reshape(N * H, D).float()

    # Compute norms and normalize
    norms = torch.norm(flat, dim=-1, keepdim=True)
    normalized = flat / (norms + 1e-10)

    # Rotate: y = x_hat @ Pi^T
    rotated = normalized @ rotation.T.to(flat.device)

    # Bucketize each coordinate
    indices = torch.bucketize(rotated, boundaries.to(flat.device))
    indices = indices.clamp(0, 2**4 - 1)

    # Nibble pack: high=even, low=odd
    idx_u8 = indices.to(torch.uint8)
    packed = (idx_u8[:, 0::2] << 4) | idx_u8[:, 1::2]

    return (
        packed.reshape(N, H, half_D),
        norms.reshape(N, H, 1),
    )


def decompress_reference(
    packed: torch.Tensor,
    norms: torch.Tensor,
    centroids: torch.Tensor,
) -> torch.Tensor:
    """Pure PyTorch TQ4 decompression: unpack + gather + scale.

    Does NOT apply rotation -- output stays in rotated space.

    Args:
        packed: ``(N, H, D//2)`` uint8 nibble-packed indices.
        norms: ``(N, H, 1)`` fp32 per-vector norms.
        centroids: ``(C,)`` fp32 centroid table (C=16 for TQ4).

    Returns:
        ``(N, H, D)`` fp32 tensor in rotated space.
    """
    N, H, half_D = packed.shape
    D = half_D * 2

    # Nibble unpack
    high = (packed >> 4).long()
    low = (packed & 0x0F).long()
    # Interleave: even positions = high, odd = low
    indices = torch.stack([high, low], dim=-1).reshape(N * H, D)

    flat_norms = norms.reshape(N * H, 1)
    reconstructed = centroids.to(indices.device)[indices]
    result = reconstructed * flat_norms

    return result.reshape(N, H, D)


def flash_attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sm_scale: Optional[float] = None,
    is_causal: bool = False,
) -> torch.Tensor:
    """Pure PyTorch flash attention (standard SDPA).

    Implements scaled dot-product attention with optional causal masking
    and GQA support. Uses fp32 accumulation for numerical stability.

    Args:
        q: Query ``(B, H_Q, N_Q, D)`` in any float dtype.
        k: Key ``(B, H_KV, N_KV, D)``.
        v: Value ``(B, H_KV, N_KV, D)``.
        sm_scale: Softmax scale. Defaults to ``1 / sqrt(D)``.
        is_causal: Apply causal masking.

    Returns:
        Attention output ``(B, H_Q, N_Q, D)`` in the input dtype.
    """
    B, H_Q, N_Q, D = q.shape
    _, H_KV, N_KV, _ = k.shape

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    # Single query token never needs causal masking
    if is_causal and N_Q == 1:
        is_causal = False

    # GQA: repeat KV heads to match Q heads
    if H_Q != H_KV:
        repeat = H_Q // H_KV
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)

    # Attention scores: Q @ K^T
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * sm_scale

    # Causal mask
    if is_causal:
        mask = torch.triu(
            torch.ones(N_Q, N_KV, device=q.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

    # Softmax
    attn_weights = torch.softmax(scores, dim=-1)

    # Output: Attn @ V
    out = torch.matmul(attn_weights, v.float())
    return out.to(q.dtype)


def fused_tq_attention_reference(
    q_rot: torch.Tensor,
    k_packed: torch.Tensor,
    k_norms: torch.Tensor,
    v_packed: torch.Tensor,
    v_norms: torch.Tensor,
    centroids_k: torch.Tensor,
    centroids_v: torch.Tensor,
    rotation: torch.Tensor,
    sm_scale: Optional[float] = None,
    is_causal: bool = False,
) -> torch.Tensor:
    """Pure PyTorch fused TQ4 attention with both K and V compressed.

    Pre-rotates Q, decompresses K and V inline, computes attention,
    then post-rotates the output.

    Args:
        q_rot: Pre-rotated query ``(B, H_Q, N_Q, D)`` fp16/bf16.
        k_packed: Nibble-packed key indices ``(B, H_KV, N_KV, D//2)`` uint8.
        k_norms: Key norms ``(B, H_KV, N_KV, 1)`` fp32.
        v_packed: Nibble-packed value indices ``(B, H_KV, N_KV, D//2)`` uint8.
        v_norms: Value norms ``(B, H_KV, N_KV, 1)`` fp32.
        centroids_k: Key codebook ``(C,)`` fp32.
        centroids_v: Value codebook ``(C,)`` fp32.
        rotation: Orthogonal rotation ``(D, D)`` fp32.
        sm_scale: Softmax scale. Defaults to ``1 / sqrt(D)``.
        is_causal: Apply causal masking.

    Returns:
        Attention output ``(B, H_Q, N_Q, D)`` in original space.
    """
    B, H_Q, N_Q, D = q_rot.shape
    _, H_KV, N_KV, half_D = k_packed.shape

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    # Pre-rotate Q (if not already rotated, but the name suggests it is)
    # The function signature says q_rot, so we assume it is pre-rotated.
    q = q_rot.float()

    # Decompress K (in rotated space, no rotation applied)
    k_decompressed = decompress_reference(
        k_packed.reshape(B * H_KV, N_KV, half_D).unsqueeze(0).reshape(
            B * H_KV, 1, N_KV, half_D
        ).squeeze(1),
        k_norms.reshape(B * H_KV, N_KV, 1).unsqueeze(0).reshape(
            B * H_KV, 1, N_KV, 1
        ).squeeze(1),
        centroids_k,
    ).reshape(B, H_KV, N_KV, D)

    # Decompress V (in rotated space)
    v_decompressed = decompress_reference(
        v_packed.reshape(B * H_KV, N_KV, half_D).unsqueeze(0).reshape(
            B * H_KV, 1, N_KV, half_D
        ).squeeze(1),
        v_norms.reshape(B * H_KV, N_KV, 1).unsqueeze(0).reshape(
            B * H_KV, 1, N_KV, 1
        ).squeeze(1),
        centroids_v,
    ).reshape(B, H_KV, N_KV, D)

    # Standard attention in rotated space
    out_rot = flash_attention_reference(
        q, k_decompressed, v_decompressed,
        sm_scale=sm_scale, is_causal=is_causal,
    )

    # Post-rotate: out = out_rot @ Pi
    rot = rotation.to(out_rot.device)
    out = torch.matmul(out_rot.float(), rot).to(q_rot.dtype)
    return out


def paged_decode_reference(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    centroids_k: torch.Tensor,
    centroids_v: torch.Tensor,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    """Pure PyTorch paged decode attention from packed TQ4 cache.

    Reads compressed KV data from a paged block table, decompresses,
    and computes single-token decode attention. Output is in the
    original (unrotated) space -- the caller provides Q in original
    space and the function handles pre/post rotation.

    NOTE: This reference does NOT handle rotation since it does not
    have access to the rotation matrix. It assumes Q is already in
    rotated space and the caller will post-rotate the output.
    For a complete end-to-end reference, use the rotation externally.

    Args:
        q: Query ``(num_seqs, H_Q, D)`` fp16/bf16 (pre-rotated).
        kv_cache: Packed paged cache ``(num_blocks, block_size, total_bytes)``
            uint8.
        block_table: Page table ``(num_seqs, max_blocks_per_seq)`` int32.
        seq_lens: Sequence lengths ``(num_seqs,)`` int32.
        centroids_k: Key codebook ``(C,)`` fp32.
        centroids_v: Value codebook ``(C,)`` fp32.
        num_kv_heads: Number of KV heads.
        head_dim: Head dimension.
        block_size: Tokens per block.
        sm_scale: Softmax scale. Defaults to ``1 / sqrt(D)``.

    Returns:
        Attention output ``(num_seqs, H_Q, D)`` in rotated space.
    """
    num_seqs, H_Q, D = q.shape
    half_D = head_dim // 2

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    # Byte layout within the packed cache row
    k_idx_end = num_kv_heads * half_D
    k_norm_end = k_idx_end + num_kv_heads * 4
    v_idx_end = k_norm_end + num_kv_heads * half_D
    # v_norm_end = v_idx_end + num_kv_heads * 4

    output = torch.zeros_like(q)

    for seq_idx in range(num_seqs):
        seq_len = int(seq_lens[seq_idx].item())
        if seq_len == 0:
            continue

        # Gather all tokens for this sequence from the page table
        k_all = []
        v_all = []

        for t in range(seq_len):
            block_idx = t // block_size
            within_block = t % block_size
            phys_block = int(block_table[seq_idx, block_idx].item())

            row = kv_cache[phys_block, within_block]  # (total_bytes,) uint8

            for h_kv in range(num_kv_heads):
                # Extract K packed indices for this head
                k_start = h_kv * half_D
                k_packed = row[k_start : k_start + half_D]

                # Extract K norm (4 bytes -> fp32)
                norm_start = k_idx_end + h_kv * 4
                k_norm_bytes = row[norm_start : norm_start + 4]
                k_norm = k_norm_bytes.view(torch.float32)

                # Nibble unpack K
                k_hi = (k_packed >> 4).long()
                k_lo = (k_packed & 0x0F).long()
                k_indices = torch.stack([k_hi, k_lo], dim=-1).flatten()
                k_vec = centroids_k.to(row.device)[k_indices] * k_norm

                # Extract V packed indices
                v_start = k_norm_end + h_kv * half_D
                v_packed = row[v_start : v_start + half_D]

                # Extract V norm
                vnorm_start = v_idx_end + h_kv * 4
                v_norm_bytes = row[vnorm_start : vnorm_start + 4]
                v_norm = v_norm_bytes.view(torch.float32)

                # Nibble unpack V
                v_hi = (v_packed >> 4).long()
                v_lo = (v_packed & 0x0F).long()
                v_indices = torch.stack([v_hi, v_lo], dim=-1).flatten()
                v_vec = centroids_v.to(row.device)[v_indices] * v_norm

                if t == 0:
                    k_all.append([])
                    v_all.append([])
                k_all[h_kv].append(k_vec)
                v_all[h_kv].append(v_vec)

        # Stack into tensors: (H_KV, seq_len, D)
        k_cache = torch.stack(
            [torch.stack(k_all[h], dim=0) for h in range(num_kv_heads)],
            dim=0,
        ).float()
        v_cache = torch.stack(
            [torch.stack(v_all[h], dim=0) for h in range(num_kv_heads)],
            dim=0,
        ).float()

        # GQA: expand KV heads to match Q heads
        gqa_ratio = H_Q // num_kv_heads
        if gqa_ratio > 1:
            k_cache = k_cache.repeat_interleave(gqa_ratio, dim=0)
            v_cache = v_cache.repeat_interleave(gqa_ratio, dim=0)

        # Compute attention for this sequence
        # q_seq: (H_Q, D), k_cache: (H_Q, seq_len, D)
        q_seq = q[seq_idx].float()  # (H_Q, D)

        # Scores: (H_Q, 1, D) @ (H_Q, D, seq_len) -> (H_Q, 1, seq_len)
        scores = torch.bmm(
            q_seq.unsqueeze(1), k_cache.transpose(-2, -1)
        ) * sm_scale

        attn_weights = torch.softmax(scores, dim=-1)

        # Output: (H_Q, 1, seq_len) @ (H_Q, seq_len, D) -> (H_Q, 1, D)
        out = torch.bmm(attn_weights, v_cache).squeeze(1)
        output[seq_idx] = out.to(q.dtype)

    return output
