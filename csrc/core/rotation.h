// Copyright 2026 FlashQuant Authors. Apache-2.0 License.
#pragma once

#include <torch/torch.h>
#include <tuple>

namespace flashquant {

// Generate Haar-distributed random orthogonal matrix via QR decomposition.
// Always uses CPU generator in float32 regardless of default device/dtype.
// Returns: (dim, dim) float32 tensor on CPU.
torch::Tensor haar_orthogonal(int dim, int64_t seed = 42);

// Pre-split rotation.T into even/odd column halves for the compress kernel.
// Input: rotation (dim, dim)
// Returns: {rotation_T_even (dim, dim/2), rotation_T_odd (dim, dim/2)}
// Both contiguous, ready for fused compress kernel.
std::tuple<torch::Tensor, torch::Tensor> split_rotation(
    const torch::Tensor& rotation);

}  // namespace flashquant
