// Copyright 2026 FlashQuant Authors. Apache-2.0 License.
#pragma once

#include <torch/torch.h>
#include <cstdint>

namespace flashquant {

struct LloydMaxCodebook {
    torch::Tensor centroids;   // (n_levels,) float32
    torch::Tensor boundaries;  // (n_levels - 1,) float32
    int dim;
    int bits;
};

struct QuantizedMSE {
    torch::Tensor indices;  // (..., dim) int32
    torch::Tensor norms;    // (..., 1) float32
};

struct QuantizedProd {
    torch::Tensor indices;         // (..., dim) int32
    torch::Tensor norms;           // (..., 1) float32
    torch::Tensor qjl_signs;      // (..., dim) int8 — NOT float32
    torch::Tensor residual_norms;  // (..., 1) float32
};

struct CompressedKeys {
    torch::Tensor packed_indices;  // (..., dim/2) uint8 nibble-packed
    torch::Tensor norms;           // (...) float32
    torch::Tensor qjl_signs;      // (..., dim) int8
    torch::Tensor residual_norms;  // (...) float32
};

struct CompressedValues {
    torch::Tensor packed_indices;  // (..., dim/2) uint8 nibble-packed
    torch::Tensor norms;           // (...) float32
};

}  // namespace flashquant
