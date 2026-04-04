"""Shared pytest fixtures for FlashQuant tests.

Provides deterministic seeding, parametrized fixtures for head dimensions,
bit widths, dtypes, and devices, plus common helper utilities. All tests
must pass on CPU without a compiled _C extension.
"""
from __future__ import annotations

import math
from typing import Any

import pytest
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------


def pytest_configure(config: Any) -> None:
    """Register custom markers."""
    config.addinivalue_line("markers", "gpu: tests that require CUDA GPU")
    config.addinivalue_line("markers", "slow: tests that take > 10 seconds")


# ---------------------------------------------------------------------------
# Autouse seed fixture
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _seed_torch() -> None:
    """Fix torch random seed before every test for reproducibility."""
    torch.manual_seed(42)


# ---------------------------------------------------------------------------
# Parametrized fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(params=[64, 128, 256], ids=["dim64", "dim128", "dim256"])
def head_dim(request: pytest.FixtureRequest) -> int:
    """Head dimension fixture: 64, 128, 256."""
    return request.param


@pytest.fixture(params=[2, 3, 4], ids=["2bit", "3bit", "4bit"])
def bits(request: pytest.FixtureRequest) -> int:
    """Bit-width fixture: 2, 3, 4."""
    return request.param


@pytest.fixture(
    params=[torch.float16, torch.bfloat16], ids=["fp16", "bf16"]
)
def dtype(request: pytest.FixtureRequest) -> torch.dtype:
    """Dtype fixture: float16, bfloat16."""
    return request.param


@pytest.fixture(
    params=[
        "cpu",
        pytest.param("cuda", marks=pytest.mark.gpu),
    ]
)
def device(request: pytest.FixtureRequest) -> str:
    """Device fixture: cpu always, cuda when available."""
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return request.param


# ---------------------------------------------------------------------------
# Pure-Python reference implementations (no _C required)
# ---------------------------------------------------------------------------
# These replicate the algorithms from turboquant-vllm so that tests can
# validate without compiled extensions or the original package installed.


def _gaussian_centroids_boundaries(
    dim: int, bits: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-Python Lloyd-Max solver for N(0, 1/d) via iterative updates."""
    n_levels = 1 << bits
    sigma = 1.0 / math.sqrt(dim)
    lo, hi = -3.5 * sigma, 3.5 * sigma

    centroids = torch.tensor(
        [lo + (hi - lo) * (i + 0.5) / n_levels for i in range(n_levels)],
        dtype=torch.float64,
    )

    inv_sigma = 1.0 / sigma
    inv_sigma_sq2 = inv_sigma / math.sqrt(2.0)
    norm_const = 1.0 / (sigma * math.sqrt(2.0 * math.pi))

    def _pdf(x: torch.Tensor) -> torch.Tensor:
        return norm_const * torch.exp(-0.5 * (x * inv_sigma) ** 2)

    def _cdf(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * (1.0 + torch.erf(x * inv_sigma_sq2))

    def _cond_exp(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        pdf_a, pdf_b = _pdf(a), _pdf(b)
        cdf_a, cdf_b = _cdf(a), _cdf(b)
        denom = cdf_b - cdf_a
        safe = torch.where(denom > 1e-15, denom, torch.ones_like(denom))
        result = (sigma**2) * (pdf_a - pdf_b) / safe
        return torch.where(denom > 1e-15, result, (a + b) / 2.0)

    for _ in range(200):
        boundaries = (centroids[:-1] + centroids[1:]) / 2.0
        edges = torch.cat(
            [torch.tensor([lo], dtype=torch.float64), boundaries,
             torch.tensor([hi], dtype=torch.float64)]
        )
        new_centroids = _cond_exp(edges[:-1], edges[1:])
        max_shift = (new_centroids - centroids).abs().max().item()
        centroids = new_centroids
        if max_shift < 1e-10:
            break

    boundaries = (centroids[:-1] + centroids[1:]) / 2.0
    return centroids.float(), boundaries.float()


def generate_rotation_matrix(dim: int, seed: int = 42) -> torch.Tensor:
    """Generate a Haar-distributed random orthogonal matrix."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    gaussian = torch.randn(dim, dim, generator=gen, device="cpu", dtype=torch.float32)
    q, r = torch.linalg.qr(gaussian)
    diag_sign = torch.sign(torch.diag(r))
    diag_sign[diag_sign == 0] = 1.0
    return q * diag_sign.unsqueeze(0)


def split_rotation(rotation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split rotation.T into even/odd column halves."""
    rot_t = rotation.T.contiguous()
    return rot_t[:, 0::2].contiguous(), rot_t[:, 1::2].contiguous()


class RefQuantizerMSE:
    """Pure-Python reference TurboQuantMSE for CPU testing."""

    def __init__(self, dim: int, bits: int, seed: int = 42) -> None:
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if bits < 1 or bits > 8:
            raise ValueError(f"bits must be in [1, 8], got {bits}")
        self.dim = dim
        self.bits = bits
        self.centroids, self.boundaries = _gaussian_centroids_boundaries(dim, bits)
        self.rotation = generate_rotation_matrix(dim, seed=seed)

    def quantize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        orig_shape = x.shape
        flat = x.reshape(-1, self.dim).float()
        norms = torch.norm(flat, dim=-1, keepdim=True)
        normalized = flat / (norms + 1e-10)
        rotated = normalized @ self.rotation.to(flat.device).T
        indices = torch.bucketize(rotated, self.boundaries.to(rotated.device))
        return indices.reshape(orig_shape), norms.reshape(*orig_shape[:-1], 1)

    def dequantize(
        self, indices: torch.Tensor, norms: torch.Tensor
    ) -> torch.Tensor:
        orig_shape = indices.shape
        flat_idx = indices.reshape(-1, self.dim)
        flat_norms = norms.reshape(-1, 1)
        reconstructed = self.centroids.to(flat_idx.device)[flat_idx]
        pi = self.rotation.to(reconstructed.device)
        unrotated = reconstructed @ pi
        result = unrotated * flat_norms
        return result.reshape(orig_shape)


class RefQuantizerProd:
    """Pure-Python reference TurboQuantProd for CPU testing."""

    def __init__(self, dim: int, bits: int, seed: int = 42) -> None:
        if bits < 2:
            raise ValueError(f"TurboQuantProd requires bits >= 2, got {bits}")
        self.dim = dim
        self.bits = bits
        self.mse = RefQuantizerMSE(dim, bits - 1, seed=seed)
        self.qjl_dim = dim
        gen = torch.Generator().manual_seed(seed + 1)
        self.qjl_matrix = torch.randn(dim, dim, generator=gen) / math.sqrt(dim)

    def quantize(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        indices, norms = self.mse.quantize(x)
        reconstructed = self.mse.dequantize(indices, norms)
        residual = x.float() - reconstructed
        residual_norms = torch.norm(residual, dim=-1, keepdim=True)
        s = self.qjl_matrix.to(x.device)
        projected = residual.reshape(-1, self.dim) @ s.T
        qjl_signs = torch.sign(projected).reshape(*x.shape[:-1], self.qjl_dim)
        qjl_signs[qjl_signs == 0] = 1.0
        return indices, norms, qjl_signs, residual_norms

    def estimate_inner_product(
        self,
        query: torch.Tensor,
        indices: torch.Tensor,
        norms: torch.Tensor,
        qjl_signs: torch.Tensor,
        residual_norms: torch.Tensor,
    ) -> torch.Tensor:
        k_mse = self.mse.dequantize(indices, norms)
        mse_term = (query.float() * k_mse).sum(dim=-1, keepdim=True)
        s = self.qjl_matrix.to(query.device)
        q_proj = query.float().reshape(-1, self.dim) @ s.T
        q_proj = q_proj.reshape(*query.shape[:-1], self.qjl_dim)
        correction = (q_proj * qjl_signs).sum(dim=-1, keepdim=True)
        scale = residual_norms * math.sqrt(math.pi / 2.0) / self.qjl_dim
        return mse_term + scale * correction


def ref_nibble_pack(indices: torch.Tensor) -> torch.Tensor:
    """Pack pairs of 4-bit indices into uint8."""
    if indices.shape[-1] % 2 != 0:
        raise ValueError(
            f"Last dimension must be even for nibble packing, got {indices.shape[-1]}"
        )
    if (indices > 15).any():
        raise ValueError("Values > 15 cannot be packed into 4-bit nibbles")
    even = indices[..., 0::2].to(torch.uint8)
    odd = indices[..., 1::2].to(torch.uint8)
    return (even << 4) | odd


def ref_nibble_unpack(packed: torch.Tensor) -> torch.Tensor:
    """Unpack uint8 bytes into pairs of 4-bit indices."""
    high = (packed >> 4).long()
    low = (packed & 0x0F).long()
    return torch.stack([high, low], dim=-1).flatten(-2)


def ref_pack_2bit(indices: torch.Tensor) -> torch.Tensor:
    """Pack 4 values of 2-bit into uint8."""
    d = indices.shape[-1]
    if d % 4 != 0:
        raise ValueError(f"Last dim must be divisible by 4, got {d}")
    flat = indices.reshape(-1, d).to(torch.uint8)
    a = flat[:, 0::4]
    b = flat[:, 1::4]
    c = flat[:, 2::4]
    d_ = flat[:, 3::4]
    packed = (a << 6) | (b << 4) | (c << 2) | d_
    return packed.reshape(*indices.shape[:-1], indices.shape[-1] // 4)


def ref_unpack_2bit(packed: torch.Tensor, dim: int) -> torch.Tensor:
    """Unpack uint8 bytes into 2-bit indices."""
    a = ((packed >> 6) & 0x03).long()
    b = ((packed >> 4) & 0x03).long()
    c = ((packed >> 2) & 0x03).long()
    d_ = (packed & 0x03).long()
    return torch.stack([a, b, c, d_], dim=-1).flatten(-2)


def cosine_similarity_flat(a: torch.Tensor, b: torch.Tensor) -> float:
    """Flat cosine similarity between two tensors."""
    return F.cosine_similarity(
        a.flatten().float(), b.flatten().float(), dim=0
    ).item()


def ref_compress(
    x: torch.Tensor, rotation: torch.Tensor, boundaries: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference compress: norm + rotate + quantize + nibble pack."""
    N, H, D = x.shape
    HALF_D = D // 2
    flat = x.reshape(N * H, D).float()
    raw_norms = torch.norm(flat, dim=-1, keepdim=True)
    normalized = flat / (raw_norms + 1e-10)
    rotated = normalized @ rotation.T
    indices = torch.bucketize(rotated, boundaries)
    indices = indices.clamp(0, 2 ** 4 - 1).to(torch.uint8)
    packed = (indices[:, 0::2] << 4) | indices[:, 1::2]
    return packed.reshape(N, H, HALF_D), raw_norms.reshape(N, H, 1)


def ref_decompress(
    packed: torch.Tensor,
    norms: torch.Tensor,
    centroids: torch.Tensor,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Reference decompress: unpack + gather + scale (NO rotation)."""
    N, H, half_D = packed.shape
    D = half_D * 2
    high = (packed >> 4).long()
    low = (packed & 0x0F).long()
    indices = torch.stack([high, low], dim=-1).reshape(N * H, D)
    flat_norms = norms.reshape(N * H, 1)
    reconstructed = centroids[indices]
    result = (reconstructed * flat_norms).reshape(N, H, D).to(dtype)
    return result


def ref_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    is_causal: bool = False, sm_scale: float | None = None,
) -> torch.Tensor:
    """Naive scaled dot-product attention on CPU. (B, H, S_q, D)."""
    B, H, S_q, D = q.shape
    _, _, S_kv, _ = k.shape
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * sm_scale
    if is_causal:
        mask = torch.triu(
            torch.ones(S_q, S_kv, device=q.device, dtype=torch.bool), diagonal=1
        )
        scores.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    out = torch.matmul(weights, v.float())
    return out.to(q.dtype)
