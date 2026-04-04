// Copyright 2026 FlashQuant Authors. Apache-2.0 License.
#include "quantizer.h"
#include "packing.h"

#include <cmath>

namespace flashquant {

// ---- TurboQuantMSE ----

TurboQuantMSE::TurboQuantMSE(int dim, int bits, int64_t seed)
    : dim_(dim), bits_(bits) {
    if (dim < 1) {
        throw std::invalid_argument("dim must be >= 1");
    }
    if (bits < 1 || bits > 8) {
        throw std::invalid_argument("bits must be in [1, 8]");
    }

    rotation_ = haar_orthogonal(dim, seed);
    codebook_ = solve_lloyd_max(dim, bits);
}

QuantizedMSE TurboQuantMSE::quantize(const torch::Tensor& x) const {
    TORCH_CHECK(x.size(-1) == dim_,
                "Input last dim must be ", dim_, ", got ", x.size(-1));

    auto orig_shape = x.sizes().vec();
    auto flat = x.reshape({-1, dim_}).to(torch::kFloat32);

    // Compute L2 norms
    auto norms = torch::linalg_vector_norm(flat, 2, /*dim=*/-1, /*keepdim=*/true);

    // Normalize (avoid division by zero with eps)
    auto normalized = flat / (norms + 1e-10f);

    // Rotate: y = x_normalized @ Pi^T
    auto pi = rotation_.to(flat.device());
    auto rotated = torch::matmul(normalized, pi.t());

    // Scalar quantize each coordinate
    auto boundaries = codebook_.boundaries.to(flat.device());
    auto indices = codebook_quantize(rotated, boundaries);

    // Reshape outputs
    auto norms_shape = orig_shape;
    norms_shape.back() = 1;
    return QuantizedMSE{
        indices.reshape(orig_shape),
        norms.reshape(norms_shape),
    };
}

torch::Tensor TurboQuantMSE::dequantize(const torch::Tensor& indices,
                                         const torch::Tensor& norms) const {
    auto orig_shape = indices.sizes().vec();
    auto flat_idx = indices.reshape({-1, dim_});
    auto flat_norms = norms.reshape({-1, 1});

    // Lookup centroids
    auto centroids = codebook_.centroids.to(indices.device());
    auto reconstructed = codebook_dequantize(flat_idx, centroids);

    // Inverse rotation: x = y @ Pi
    auto pi = rotation_.to(indices.device());
    auto unrotated = torch::matmul(reconstructed, pi);

    // Rescale by norms
    auto result = unrotated * flat_norms;

    return result.reshape(orig_shape).to(torch::kFloat32);
}

// ---- TurboQuantProd ----

TurboQuantProd::TurboQuantProd(int dim, int bits, int64_t seed)
    : dim_(dim),
      bits_(bits),
      mse_quantizer_(dim, bits - 1, seed) {
    if (bits < 2) {
        throw std::invalid_argument(
            "TurboQuantProd requires bits >= 2 (1 bit for QJL), got " +
            std::to_string(bits));
    }

    // QJL random projection matrix: S ~ N(0, 1)^{d x d} / sqrt(d)
    // FIX: explicit CPU device (bug in turboquant-vllm)
    auto gen = torch::make_generator<torch::CPUGeneratorImpl>();
    gen.set_current_seed(seed + 1);  // Different seed from rotation

    qjl_matrix_ = torch::randn({dim, dim},
                                gen,
                                torch::TensorOptions()
                                    .dtype(torch::kFloat32)
                                    .device(torch::kCPU)) /
                   std::sqrt(static_cast<double>(dim));
}

QuantizedProd TurboQuantProd::quantize(const torch::Tensor& x) const {
    TORCH_CHECK(x.size(-1) == dim_,
                "Input last dim must be ", dim_, ", got ", x.size(-1));

    // Stage 1: MSE quantization with (bits-1) bits
    auto [indices, norms] = mse_quantizer_.quantize(x);

    // Compute residual: r = x - dequant(quant(x))
    auto reconstructed = mse_quantizer_.dequantize(indices, norms);
    auto residual = x.to(torch::kFloat32) - reconstructed;

    // Residual norms
    auto residual_flat = residual.reshape({-1, dim_});
    auto residual_norms =
        torch::linalg_vector_norm(residual_flat, 2, /*dim=*/-1, /*keepdim=*/true);

    // Stage 2: QJL projection on residuals
    // signs = sign(S @ r)
    auto s = qjl_matrix_.to(x.device());
    auto projected = torch::matmul(residual_flat, s.t());
    auto signs = torch::sign(projected);
    signs = torch::where(signs == 0, torch::ones_like(signs), signs);

    // Store as int8 ({-1, +1}) — NOT float32 (32x memory saving)
    auto signs_int8 = signs.to(torch::kInt8);

    auto norms_shape = x.sizes().vec();
    norms_shape.back() = 1;

    return QuantizedProd{
        indices,
        norms,
        signs_int8.reshape(x.sizes()),
        residual_norms.reshape(norms_shape),
    };
}

torch::Tensor TurboQuantProd::dequantize(const torch::Tensor& indices,
                                          const torch::Tensor& norms) const {
    // MSE-only reconstruction. QJL correction is applied only in
    // estimate_inner_product() for attention score computation.
    return mse_quantizer_.dequantize(indices, norms);
}

torch::Tensor TurboQuantProd::estimate_inner_product(
    const torch::Tensor& query, const QuantizedProd& compressed) const {
    // Unbiased estimator from Theorem 2 of TurboQuant paper:
    // <q, x> ≈ <q, x̃_mse> + γ * sqrt(π/2) / d * <S@q, signs(S@r)>
    //
    // Term 1: MSE reconstruction inner product
    // Term 2: QJL bias correction

    auto q_flat = query.reshape({-1, dim_}).to(torch::kFloat32);

    // Term 1: <q, x̃_mse>
    auto x_mse = mse_quantizer_.dequantize(compressed.indices, compressed.norms);
    auto x_mse_flat = x_mse.reshape({-1, dim_});

    // Batched dot product: (N, D) x (M, D) -> (N, M) via matmul
    // But we want element-wise dot, so handle shapes carefully.
    auto mse_term = torch::sum(q_flat * x_mse_flat, /*dim=*/-1, /*keepdim=*/true);

    // Term 2: QJL correction
    auto s = qjl_matrix_.to(query.device());

    // Project query: S @ q
    auto q_proj = torch::matmul(q_flat, s.t());  // (N, d)

    // Signs are int8, convert to float for dot product
    auto signs_float = compressed.qjl_signs.reshape({-1, dim_}).to(torch::kFloat32);

    // <S@q, signs(S@r)>
    auto qjl_dot = torch::sum(q_proj * signs_float, /*dim=*/-1, /*keepdim=*/true);

    // Scale: γ * sqrt(π/2) / d
    auto residual_norms_flat = compressed.residual_norms.reshape({-1, 1});
    constexpr double sqrt_pi_over_2 = 1.2533141373155003;  // sqrt(π/2)
    auto scale = residual_norms_flat * (sqrt_pi_over_2 / dim_);

    auto result = mse_term + scale * qjl_dot;

    // Reshape to match query batch dims
    auto out_shape = query.sizes().vec();
    out_shape.back() = 1;
    return result.reshape(out_shape);
}

}  // namespace flashquant
