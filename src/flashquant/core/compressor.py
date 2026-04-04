"""Production-grade compressors for transformer KV cache tensors.

Wraps the core TurboQuant quantizers to handle real model tensor shapes
([batch, heads, seq_len, head_dim]), dtype conversion, and device placement.

- KeyCompressor: For key cache -- uses TurboQuantProd (seed=42) with QJL
  correction for unbiased inner product estimation in attention scoring.
- ValueCompressor: For value cache -- uses TurboQuantMSE (seed=43) for
  MSE-optimal reconstruction. Lighter weight since values only need
  reconstruction accuracy (not inner-product preservation).

**Separate codebooks**: K and V use different seeds (42 vs 43) so they
get independent rotation matrices and codebooks. This provides better
error decorrelation than sharing a single codebook.

Reference: Section 5 of arXiv 2504.19874.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

from flashquant.core.quantizer import TurboQuantMSE, TurboQuantProd


# ---------------------------------------------------------------------------
# Compressed representations
# ---------------------------------------------------------------------------


@dataclass
class CompressedKeys:
    """Compressed key cache representation.

    Stores all components needed to compute attention scores from
    compressed keys without full dequantization.

    Attributes:
        indices: Lloyd-Max centroid indices, shape (batch, heads, seq, head_dim).
        norms: Vector norms, shape (batch, heads, seq, 1).
        qjl_signs: QJL sign bits, shape (batch, heads, seq, qjl_dim).
        residual_norms: Residual norms, shape (batch, heads, seq, 1).
        original_dtype: Original tensor dtype for casting results.
    """

    indices: torch.Tensor
    norms: torch.Tensor
    qjl_signs: torch.Tensor
    residual_norms: torch.Tensor
    original_dtype: torch.dtype = torch.float16


@dataclass
class CompressedValues:
    """Compressed value cache representation.

    Stores components needed to reconstruct value vectors.

    Attributes:
        indices: Lloyd-Max centroid indices, shape (batch, heads, seq, head_dim).
        norms: Vector norms, shape (batch, heads, seq, 1).
        original_dtype: Original tensor dtype for casting results.
    """

    indices: torch.Tensor
    norms: torch.Tensor
    original_dtype: torch.dtype = torch.float16


# ---------------------------------------------------------------------------
# KeyCompressor (TurboQuantProd, seed=42)
# ---------------------------------------------------------------------------


class KeyCompressor:
    """Key cache compressor with unbiased attention score estimation.

    Uses the full two-stage TurboQuantProd algorithm (seed=42) to
    compress key vectors while preserving accurate inner product
    estimation for attention computation (Q*K^T).

    Attributes:
        quantizer: Two-stage TurboQuantProd instance.
        bits: Total bit budget per coordinate.
        head_dim: Model head dimension.

    Examples:
        >>> comp = KeyCompressor(head_dim=128, bits=4)
        >>> compressed = comp.compress(key_states)
        >>> scores = comp.asymmetric_attention_scores(query, compressed)
    """

    def __init__(
        self, head_dim: int, bits: int = 4, *, seed: int = 42
    ) -> None:
        """Initialize the key compressor.

        Args:
            head_dim: Dimension of each attention head.
            bits: Total bits per coordinate (default 4).
            seed: Random seed for reproducibility. Default 42 (key seed).
        """
        self.head_dim = head_dim
        self.bits = bits
        self.quantizer = TurboQuantProd(head_dim, bits, seed=seed)

    def compress(self, keys: torch.Tensor) -> CompressedKeys:
        """Compress key tensors.

        Args:
            keys: Key tensor of shape (batch, heads, seq_len, head_dim).

        Returns:
            CompressedKeys containing all components for attention estimation.
        """
        original_dtype = keys.dtype
        indices, norms, qjl_signs, residual_norms = self.quantizer.quantize(
            keys.float()
        )
        return CompressedKeys(
            indices=indices,
            norms=norms,
            qjl_signs=qjl_signs,
            residual_norms=residual_norms,
            original_dtype=original_dtype,
        )

    def decompress(self, compressed: CompressedKeys) -> torch.Tensor:
        """Reconstruct key tensors from compressed representation.

        Note: For attention, prefer ``asymmetric_attention_scores()`` which
        uses the QJL-corrected inner product estimator for better accuracy.

        Args:
            compressed: CompressedKeys from compress().

        Returns:
            Reconstructed key tensor in the original dtype.
        """
        result = self.quantizer.dequantize(
            compressed.indices,
            compressed.norms,
            compressed.qjl_signs,
            compressed.residual_norms,
        )
        return result.to(compressed.original_dtype)

    def asymmetric_attention_scores(
        self, query: torch.Tensor, compressed: CompressedKeys
    ) -> torch.Tensor:
        """Compute attention scores directly from compressed keys.

        Uses the unbiased two-stage inner product estimator rather than
        decompressing keys and computing standard dot products. This is
        both more memory-efficient and more accurate.

        Warning:
            The current implementation expands tensors to
            (batch, heads, q_len, kv_len, dim) for broadcasting.
            Suitable for correctness testing on short sequences only.

        Args:
            query: Query tensor, shape (batch, heads, q_len, head_dim).
            compressed: CompressedKeys from compress().

        Returns:
            Attention logits, shape (batch, heads, q_len, kv_len).
        """
        b, h, q_len, d = query.shape
        _, _, kv_len, _ = compressed.indices.shape

        q_exp = query.float().unsqueeze(3).expand(b, h, q_len, kv_len, d)
        idx_exp = compressed.indices.unsqueeze(2).expand(
            b, h, q_len, kv_len, d
        )
        n_exp = compressed.norms.unsqueeze(2).expand(
            b, h, q_len, kv_len, 1
        )
        qjl_exp = compressed.qjl_signs.unsqueeze(2).expand(
            b, h, q_len, kv_len, self.quantizer.qjl_dim
        )
        rn_exp = compressed.residual_norms.unsqueeze(2).expand(
            b, h, q_len, kv_len, 1
        )

        scores = self.quantizer.estimate_inner_product(
            q_exp, idx_exp, n_exp, qjl_exp, rn_exp
        )
        return scores.squeeze(-1).to(query.dtype)


# ---------------------------------------------------------------------------
# ValueCompressor (TurboQuantMSE, seed=43)
# ---------------------------------------------------------------------------


class ValueCompressor:
    """Value cache compressor with MSE-optimal reconstruction.

    Uses Stage 1 only (TurboQuantMSE, seed=43) for value vectors.
    Values appear in the ``softmax(scores) @ V`` multiplication where
    reconstruction quality matters but inner-product structure does not.

    **Separate seed (43)**: Ensures the value codebook and rotation
    matrix are independent from the key codebook (seed=42).

    Attributes:
        quantizer: TurboQuantMSE instance.
        bits: Bits per coordinate.
        head_dim: Model head dimension.

    Examples:
        >>> comp = ValueCompressor(head_dim=128, bits=4)
        >>> compressed = comp.compress(value_states)
        >>> reconstructed = comp.decompress(compressed)
    """

    def __init__(
        self, head_dim: int, bits: int = 4, *, seed: int = 43
    ) -> None:
        """Initialize the value compressor.

        Args:
            head_dim: Dimension of each attention head.
            bits: Bits per coordinate (default 4).
            seed: Random seed for reproducibility. Default 43 (value seed).
        """
        self.head_dim = head_dim
        self.bits = bits
        self.quantizer = TurboQuantMSE(head_dim, bits, seed=seed)

    def compress(self, values: torch.Tensor) -> CompressedValues:
        """Compress value tensors.

        Args:
            values: Value tensor of shape (batch, heads, seq_len, head_dim).

        Returns:
            CompressedValues containing indices and norms.
        """
        original_dtype = values.dtype
        indices, norms = self.quantizer.quantize(values.float())
        return CompressedValues(
            indices=indices,
            norms=norms,
            original_dtype=original_dtype,
        )

    def decompress(self, compressed: CompressedValues) -> torch.Tensor:
        """Reconstruct value tensors from compressed representation.

        Args:
            compressed: CompressedValues from compress().

        Returns:
            Reconstructed value tensor in the original dtype.
        """
        result = self.quantizer.dequantize(compressed.indices, compressed.norms)
        return result.to(compressed.original_dtype)
