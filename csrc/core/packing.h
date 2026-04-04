// Copyright 2026 FlashQuant Authors. Apache-2.0 License.
#pragma once

#include <torch/torch.h>

namespace flashquant {

// Pack pairs of 4-bit indices into uint8 bytes.
// Layout: high_nibble = even index, low_nibble = odd index
// Input: (..., D) with values in [0, 15], D must be even
// Output: (..., D/2) uint8
torch::Tensor nibble_pack(const torch::Tensor& indices);

// Unpack uint8 bytes into pairs of 4-bit indices.
// Input: (..., D/2) uint8
// Output: (..., D) int32
torch::Tensor nibble_unpack(const torch::Tensor& packed);

// Pack 2-bit indices: 4 values per byte.
// Input: (..., D) with values in [0, 3], D must be divisible by 4
// Output: (..., D/4) uint8
torch::Tensor pack_2bit(const torch::Tensor& indices);

// Unpack 2-bit packed indices.
// Input: (..., D/4) uint8
// Output: (..., D) int32
torch::Tensor unpack_2bit(const torch::Tensor& packed, int64_t dim);

// Validate that all indices are in [0, 2^bits - 1].
// Throws std::invalid_argument if any value is out of range.
void validate_indices(const torch::Tensor& indices, int bits);

}  // namespace flashquant
