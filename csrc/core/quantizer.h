// Copyright 2026 FlashQuant Authors. Apache-2.0 License.
#pragma once

#include "codebook.h"
#include "rotation.h"
#include "types.h"

namespace flashquant {

// TurboQuant Stage 1: MSE-optimal quantization.
// Pipeline: norm -> normalize -> rotate -> scalar quantize per coordinate.
//
// Used for value cache compression where reconstruction quality matters.
class TurboQuantMSE {
public:
    TurboQuantMSE(int dim, int bits, int64_t seed = 42);

    // Quantize input vectors.
    // Input: x (..., dim) any float type
    // Returns: {indices (..., dim) int32, norms (..., 1) float32}
    QuantizedMSE quantize(const torch::Tensor& x) const;

    // Dequantize (reconstruct).
    // Returns: (..., dim) float32
    torch::Tensor dequantize(const torch::Tensor& indices,
                             const torch::Tensor& norms) const;

    // Accessors for kernel integration
    const torch::Tensor& rotation() const { return rotation_; }
    const LloydMaxCodebook& codebook() const { return codebook_; }
    int dim() const { return dim_; }
    int bits() const { return bits_; }

private:
    int dim_;
    int bits_;
    torch::Tensor rotation_;   // (dim, dim) float32 on CPU
    LloydMaxCodebook codebook_;
};

// TurboQuant Stage 1 + QJL: Unbiased inner product estimation.
// Stage 1: MSE quantization with (bits-1) bits
// Stage 2: QJL sign-projection on residual with 1 bit per coordinate
//
// Used for key cache compression where attention score accuracy matters.
//
// IMPORTANT: dequantize() returns MSE-only reconstruction (no QJL).
// For attention scores, use estimate_inner_product() which applies QJL correction.
class TurboQuantProd {
public:
    TurboQuantProd(int dim, int bits, int64_t seed = 42);

    // Quantize with QJL correction.
    // Returns: {indices, norms, qjl_signs (int8!), residual_norms}
    QuantizedProd quantize(const torch::Tensor& x) const;

    // MSE-only reconstruction (ignores QJL). Use for visualization only.
    torch::Tensor dequantize(const torch::Tensor& indices,
                             const torch::Tensor& norms) const;

    // Unbiased inner product estimation with QJL correction.
    // <q, x> ≈ <q, x̃_mse> + γ·√(π/2)/d · <S·q, sign(S·r)>
    //
    // Input: query (..., dim), compressed QuantizedProd
    // Output: (...) attention scores
    //
    // FIX: No 5D expansion — uses chunked computation.
    torch::Tensor estimate_inner_product(const torch::Tensor& query,
                                         const QuantizedProd& compressed) const;

    // Accessors
    const TurboQuantMSE& mse_quantizer() const { return mse_quantizer_; }
    const torch::Tensor& qjl_matrix() const { return qjl_matrix_; }

private:
    int dim_;
    int bits_;
    TurboQuantMSE mse_quantizer_;
    torch::Tensor qjl_matrix_;  // (dim, dim) float32 on CPU
};

}  // namespace flashquant
