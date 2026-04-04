// Copyright 2026 FlashQuant Authors. Apache-2.0 License.
#pragma once

#include "types.h"
#include <mutex>
#include <unordered_map>

namespace flashquant {

// Closed-form Lloyd-Max for Gaussian N(0, 1/d).
// No scipy dependency — uses erfinv + closed-form conditional expectations.
//
// Boundaries: b_i = sigma * sqrt(2) * erfinv(2*i/n_levels - 1)
// Centroids:  c_i = sigma * [phi(a_i) - phi(b_i)] / [Phi(b_i) - Phi(a_i)]
//
// where phi = Gaussian PDF, Phi = Gaussian CDF, sigma = 1/sqrt(d)
LloydMaxCodebook gaussian_lloyd_max(int dim, int bits);

// Iterative Lloyd-Max for exact Beta((d-1)/2, (d-1)/2) distribution.
// Only used when dim < 64 where Gaussian approximation is poor.
// Falls back to gaussian_lloyd_max if dim >= 64.
LloydMaxCodebook beta_lloyd_max(int dim, int bits, int max_iter = 200,
                                 double tol = 1e-10);

// Auto-dispatch: gaussian for dim >= 64, beta for dim < 64.
LloydMaxCodebook solve_lloyd_max(int dim, int bits);

// Quantize using pre-computed boundaries: x -> bucket index in [0, 2^bits - 1]
torch::Tensor codebook_quantize(const torch::Tensor& x,
                                const torch::Tensor& boundaries);

// Dequantize: indices -> centroid values
torch::Tensor codebook_dequantize(const torch::Tensor& indices,
                                  const torch::Tensor& centroids);

// Thread-safe, device-aware codebook cache.
// Codebooks are computed on CPU, then moved to target device once.
class CodebookRegistry {
public:
    static CodebookRegistry& instance();

    // Get or compute codebook for (dim, bits) on given device.
    // Thread-safe. Returns device-resident tensors.
    const LloydMaxCodebook& get(int dim, int bits, torch::Device device);

    // Pre-compute codebooks for a model's configuration.
    void warmup(const std::vector<int>& dims,
                const std::vector<int>& bits_options, torch::Device device);

    void clear();

private:
    CodebookRegistry() = default;

    struct CacheKey {
        int dim;
        int bits;
        torch::Device device;
        bool operator==(const CacheKey& other) const;
    };

    struct CacheKeyHash {
        size_t operator()(const CacheKey& k) const;
    };

    std::mutex mutex_;
    std::unordered_map<CacheKey, LloydMaxCodebook, CacheKeyHash> cache_;
};

}  // namespace flashquant
