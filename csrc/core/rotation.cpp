// Copyright 2026 FlashQuant Authors. Apache-2.0 License.
#include "rotation.h"

#include <stdexcept>

namespace flashquant {

torch::Tensor haar_orthogonal(int dim, int64_t seed) {
    if (dim < 1) {
        throw std::invalid_argument("dim must be >= 1, got " +
                                    std::to_string(dim));
    }

    // Always generate on CPU with explicit seed for reproducibility.
    // This ensures the same rotation matrix regardless of:
    // - torch.set_default_device()
    // - CUDA availability
    // - Global RNG state
    auto gen = torch::make_generator<torch::CPUGeneratorImpl>();
    gen.set_current_seed(seed);

    auto gaussian = torch::randn({dim, dim},
                                 gen,
                                 torch::TensorOptions()
                                     .dtype(torch::kFloat32)
                                     .device(torch::kCPU));

    // QR decomposition produces Haar-uniform distribution over O(d)
    auto [q, r] = torch::linalg_qr(gaussian);

    // Sign correction: ensure unique decomposition by making
    // diagonal of R positive. This produces a uniformly distributed
    // orthogonal matrix (not just up to sign).
    auto diag_sign = torch::sign(torch::diag(r));
    diag_sign = torch::where(diag_sign == 0,
                             torch::ones_like(diag_sign), diag_sign);

    return q * diag_sign.unsqueeze(0);
}

std::tuple<torch::Tensor, torch::Tensor> split_rotation(
    const torch::Tensor& rotation) {
    TORCH_CHECK(rotation.dim() == 2 && rotation.size(0) == rotation.size(1),
                "rotation must be square 2D tensor, got shape ",
                rotation.sizes());
    TORCH_CHECK(rotation.size(1) % 2 == 0,
                "rotation dim must be even, got ", rotation.size(1));

    // rotation.T[:, ::2] and rotation.T[:, 1::2]
    auto rot_t = rotation.t().contiguous();
    auto rot_t_even =
        rot_t.index({torch::indexing::Slice(),
                     torch::indexing::Slice(0, torch::indexing::None, 2)})
            .contiguous();
    auto rot_t_odd =
        rot_t.index({torch::indexing::Slice(),
                     torch::indexing::Slice(1, torch::indexing::None, 2)})
            .contiguous();

    return {rot_t_even, rot_t_odd};
}

}  // namespace flashquant
