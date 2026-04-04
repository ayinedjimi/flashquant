// Copyright 2026 FlashQuant Authors. Apache-2.0 License.
#include "packing.h"

#include <stdexcept>

namespace flashquant {

void validate_indices(const torch::Tensor& indices, int bits) {
    int max_val = (1 << bits) - 1;
    auto min_v = indices.min().item<int64_t>();
    auto max_v = indices.max().item<int64_t>();
    if (min_v < 0 || max_v > max_val) {
        throw std::invalid_argument(
            "Indices out of range for " + std::to_string(bits) +
            "-bit packing. Expected [0, " + std::to_string(max_val) +
            "], got [" + std::to_string(min_v) + ", " +
            std::to_string(max_v) + "]");
    }
}

torch::Tensor nibble_pack(const torch::Tensor& indices) {
    TORCH_CHECK(indices.size(-1) % 2 == 0,
                "Last dimension must be even for nibble packing, got ",
                indices.size(-1));

    validate_indices(indices, 4);

    auto idx = indices.to(torch::kUInt8);
    auto even = idx.index({torch::indexing::Ellipsis,
                           torch::indexing::Slice(0, torch::indexing::None, 2)});
    auto odd = idx.index({torch::indexing::Ellipsis,
                          torch::indexing::Slice(1, torch::indexing::None, 2)});

    return (even << 4) | odd;
}

torch::Tensor nibble_unpack(const torch::Tensor& packed) {
    auto high = (packed >> 4).to(torch::kInt32) & 0x0F;
    auto low = packed.to(torch::kInt32) & 0x0F;

    // Interleave: [h0, l0, h1, l1, ...]
    auto stacked = torch::stack({high, low}, -1);
    auto shape = packed.sizes().vec();
    shape.back() *= 2;
    return stacked.reshape(shape);
}

torch::Tensor pack_2bit(const torch::Tensor& indices) {
    TORCH_CHECK(indices.size(-1) % 4 == 0,
                "Last dimension must be divisible by 4 for 2-bit packing, got ",
                indices.size(-1));

    validate_indices(indices, 2);

    auto idx = indices.to(torch::kUInt8);
    auto v0 = idx.index({torch::indexing::Ellipsis,
                         torch::indexing::Slice(0, torch::indexing::None, 4)});
    auto v1 = idx.index({torch::indexing::Ellipsis,
                         torch::indexing::Slice(1, torch::indexing::None, 4)});
    auto v2 = idx.index({torch::indexing::Ellipsis,
                         torch::indexing::Slice(2, torch::indexing::None, 4)});
    auto v3 = idx.index({torch::indexing::Ellipsis,
                         torch::indexing::Slice(3, torch::indexing::None, 4)});

    return (v0 << 6) | (v1 << 4) | (v2 << 2) | v3;
}

torch::Tensor unpack_2bit(const torch::Tensor& packed, int64_t dim) {
    auto p = packed.to(torch::kInt32);
    auto v0 = (p >> 6) & 0x03;
    auto v1 = (p >> 4) & 0x03;
    auto v2 = (p >> 2) & 0x03;
    auto v3 = p & 0x03;

    auto stacked = torch::stack({v0, v1, v2, v3}, -1);
    auto shape = packed.sizes().vec();
    shape.back() = dim;
    return stacked.reshape(shape);
}

}  // namespace flashquant
