"""Tests for nibble (4-bit) and 2-bit packing/unpacking."""
from __future__ import annotations

import pytest
import torch

from .conftest import (
    ref_nibble_pack,
    ref_nibble_unpack,
    ref_pack_2bit,
    ref_unpack_2bit,
)


# ---------------------------------------------------------------------------
# Parametrized fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(params=[64, 128, 256], ids=["d64", "d128", "d256"])
def pack_dim(request: pytest.FixtureRequest) -> int:
    return request.param


# ---------------------------------------------------------------------------
# Nibble (4-bit) tests
# ---------------------------------------------------------------------------


class TestNibbleRoundtrip:
    """Pack then unpack should return identity."""

    def test_nibble_roundtrip(self, pack_dim: int) -> None:
        indices = torch.randint(0, 16, (32, pack_dim), dtype=torch.uint8)
        packed = ref_nibble_pack(indices.long())
        assert packed.shape == (32, pack_dim // 2)

        unpacked = ref_nibble_unpack(packed)
        assert unpacked.shape == indices.shape
        assert torch.equal(unpacked, indices.long())


class TestNibbleSpecificValues:
    """Known input/output pairs for nibble packing."""

    def test_nibble_specific_values(self) -> None:
        # [0, 15] should pack to 0x0F = 15
        indices = torch.tensor([[0, 15]], dtype=torch.long)
        packed = ref_nibble_pack(indices)
        assert packed[0, 0].item() == 0x0F

        # [15, 0] should pack to 0xF0 = 240
        indices = torch.tensor([[15, 0]], dtype=torch.long)
        packed = ref_nibble_pack(indices)
        assert packed[0, 0].item() == 0xF0

        # [5, 10] -> 0x5A = 90
        indices = torch.tensor([[5, 10]], dtype=torch.long)
        packed = ref_nibble_pack(indices)
        assert packed[0, 0].item() == 0x5A

        # Roundtrip
        unpacked = ref_nibble_unpack(packed)
        assert torch.equal(unpacked, indices)


class TestNibbleRoundtripBatch:
    """Nibble roundtrip with batch dimensions."""

    def test_nibble_roundtrip_batch(self, pack_dim: int) -> None:
        indices = torch.randint(0, 16, (4, 8, pack_dim), dtype=torch.long)
        packed = ref_nibble_pack(indices)
        assert packed.shape == (4, 8, pack_dim // 2)

        unpacked = ref_nibble_unpack(packed)
        assert torch.equal(unpacked, indices)


# ---------------------------------------------------------------------------
# 2-bit tests
# ---------------------------------------------------------------------------


class Test2BitRoundtrip:
    """Pack then unpack should return identity for 2-bit."""

    def test_2bit_roundtrip(self, pack_dim: int) -> None:
        indices = torch.randint(0, 4, (32, pack_dim), dtype=torch.long)
        packed = ref_pack_2bit(indices)
        assert packed.shape == (32, pack_dim // 4)

        unpacked = ref_unpack_2bit(packed, pack_dim)
        assert unpacked.shape == indices.shape
        assert torch.equal(unpacked, indices)


# ---------------------------------------------------------------------------
# Overflow / error handling
# ---------------------------------------------------------------------------


class TestOverflowDetection:
    """Values > 15 should raise ValueError for 4-bit packing."""

    def test_overflow_detection(self) -> None:
        indices = torch.tensor([[16, 0]], dtype=torch.long)
        with pytest.raises(ValueError, match="Values > 15"):
            ref_nibble_pack(indices)


class TestOddDimRaises:
    """Nibble pack with odd last dim should raise."""

    def test_odd_dim_raises(self) -> None:
        indices = torch.randint(0, 16, (10, 3), dtype=torch.long)
        with pytest.raises(ValueError, match="even"):
            ref_nibble_pack(indices)
