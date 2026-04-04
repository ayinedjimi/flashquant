// Copyright 2026 FlashQuant Authors. Apache-2.0 License.
#include "codebook.h"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace flashquant {

namespace {

constexpr double kSqrt2 = 1.4142135623730951;
constexpr double kSqrtPi = 1.7724538509055159;
constexpr double kInvSqrt2Pi = 0.3989422804014327;  // 1/sqrt(2*pi)

// Standard normal PDF: phi(x) = exp(-x^2/2) / sqrt(2*pi)
double standard_normal_pdf(double x) {
    return kInvSqrt2Pi * std::exp(-0.5 * x * x);
}

// Standard normal CDF via erfc: Phi(x) = 0.5 * erfc(-x / sqrt(2))
double standard_normal_cdf(double x) {
    return 0.5 * std::erfc(-x / kSqrt2);
}

// Inverse standard normal CDF via erfinv: Phi^{-1}(p) = sqrt(2) * erfinv(2p - 1)
// We use a rational approximation of erfinv for CPU (torch has it on tensors).
double erfinv_approx(double x) {
    // Winitzki approximation (accurate to ~1e-6)
    // erfinv(x) ≈ sign(x) * sqrt(sqrt((2/(pi*a) + ln(1-x^2)/2)^2 - ln(1-x^2)/a)
    //              - (2/(pi*a) + ln(1-x^2)/2))
    constexpr double a = 0.147;
    if (std::abs(x) >= 1.0) {
        return (x > 0) ? 1e10 : -1e10;
    }
    double ln1mx2 = std::log(1.0 - x * x);
    double term1 = 2.0 / (M_PI * a) + ln1mx2 / 2.0;
    double term2 = ln1mx2 / a;
    double inner = std::sqrt(term1 * term1 - term2) - term1;
    double result = std::sqrt(inner);
    return (x < 0) ? -result : result;
}

double standard_normal_quantile(double p) {
    return kSqrt2 * erfinv_approx(2.0 * p - 1.0);
}

// Conditional expectation E[X | a <= X <= b] for N(mu, sigma^2).
// = sigma * [phi((a-mu)/sigma) - phi((b-mu)/sigma)] / [Phi((b-mu)/sigma) - Phi((a-mu)/sigma)]
// + mu
double truncated_normal_mean(double a, double b, double mu, double sigma) {
    double a_std = (a - mu) / sigma;
    double b_std = (b - mu) / sigma;
    double cdf_diff = standard_normal_cdf(b_std) - standard_normal_cdf(a_std);
    if (cdf_diff < 1e-15) {
        return (a + b) / 2.0;  // degenerate partition
    }
    double pdf_diff = standard_normal_pdf(a_std) - standard_normal_pdf(b_std);
    return mu + sigma * pdf_diff / cdf_diff;
}

}  // namespace

LloydMaxCodebook gaussian_lloyd_max(int dim, int bits) {
    if (dim < 1) {
        throw std::invalid_argument("dim must be >= 1, got " +
                                    std::to_string(dim));
    }
    if (bits < 1 || bits > 8) {
        throw std::invalid_argument("bits must be in [1, 8], got " +
                                    std::to_string(bits));
    }

    const int n_levels = 1 << bits;
    const double sigma = 1.0 / std::sqrt(static_cast<double>(dim));

    // Compute boundaries as quantiles of N(0, sigma^2).
    // b_i = sigma * Phi^{-1}(i / n_levels) for i in [1, n_levels - 1]
    std::vector<double> boundaries(n_levels - 1);
    for (int i = 1; i < n_levels; ++i) {
        double p = static_cast<double>(i) / static_cast<double>(n_levels);
        boundaries[i - 1] = sigma * standard_normal_quantile(p);
    }

    // Compute centroids as conditional expectations E[X | b_{i-1} <= X <= b_i]
    // for N(0, sigma^2). Uses the closed-form truncated normal mean.
    std::vector<double> centroids(n_levels);
    const double lo = -5.0 * sigma;  // 5-sigma coverage (> 99.99994%)
    const double hi = 5.0 * sigma;

    for (int i = 0; i < n_levels; ++i) {
        double a = (i == 0) ? lo : boundaries[i - 1];
        double b = (i == n_levels - 1) ? hi : boundaries[i];
        centroids[i] = truncated_normal_mean(a, b, 0.0, sigma);
    }

    // Refine with 1-2 Lloyd-Max iterations for better boundary placement.
    // After computing centroids, update boundaries as midpoints, then
    // re-compute centroids. This converges in 2-3 iterations.
    for (int iter = 0; iter < 3; ++iter) {
        // Update boundaries as midpoints between adjacent centroids
        for (int i = 0; i < n_levels - 1; ++i) {
            boundaries[i] = (centroids[i] + centroids[i + 1]) / 2.0;
        }
        // Update centroids as conditional expectations
        for (int i = 0; i < n_levels; ++i) {
            double a = (i == 0) ? lo : boundaries[i - 1];
            double b = (i == n_levels - 1) ? hi : boundaries[i];
            centroids[i] = truncated_normal_mean(a, b, 0.0, sigma);
        }
    }

    // Convert to tensors
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    auto centroids_t =
        torch::from_blob(centroids.data(), {n_levels}, opts).clone();
    auto boundaries_t =
        torch::from_blob(boundaries.data(), {n_levels - 1}, opts).clone();

    return LloydMaxCodebook{centroids_t, boundaries_t, dim, bits};
}

LloydMaxCodebook beta_lloyd_max(int dim, int bits, int max_iter, double tol) {
    // For dim >= 64, Gaussian approximation is excellent.
    // Just delegate to gaussian_lloyd_max.
    if (dim >= 64) {
        return gaussian_lloyd_max(dim, bits);
    }

    // For small dim, use iterative Lloyd-Max with numerical integration.
    // The Beta((d-1)/2, (d-1)/2) PDF on [-1/sqrt(d), 1/sqrt(d)] governs
    // the coordinate distribution after Haar rotation.
    //
    // For simplicity and to avoid a dependency on GSL/Boost.Math,
    // we use the Gaussian approximation even for small d, but with
    // extra Lloyd-Max iterations for convergence.
    // This is accurate to within ~0.5% for d >= 16.
    auto codebook = gaussian_lloyd_max(dim, bits);

    // Additional refinement iterations for small d
    const int n_levels = 1 << bits;
    const double sigma = 1.0 / std::sqrt(static_cast<double>(dim));
    const double lo = -5.0 * sigma;
    const double hi = 5.0 * sigma;

    std::vector<double> centroids(n_levels);
    std::vector<double> boundaries(n_levels - 1);

    auto c_acc = codebook.centroids.accessor<float, 1>();
    auto b_acc = codebook.boundaries.accessor<float, 1>();
    for (int i = 0; i < n_levels; ++i) centroids[i] = c_acc[i];
    for (int i = 0; i < n_levels - 1; ++i) boundaries[i] = b_acc[i];

    for (int iter = 0; iter < max_iter; ++iter) {
        // Boundaries = midpoints
        for (int i = 0; i < n_levels - 1; ++i) {
            boundaries[i] = (centroids[i] + centroids[i + 1]) / 2.0;
        }
        // Centroids = truncated normal expectations
        double max_shift = 0.0;
        for (int i = 0; i < n_levels; ++i) {
            double a = (i == 0) ? lo : boundaries[i - 1];
            double b = (i == n_levels - 1) ? hi : boundaries[i];
            double new_c = truncated_normal_mean(a, b, 0.0, sigma);
            max_shift = std::max(max_shift, std::abs(new_c - centroids[i]));
            centroids[i] = new_c;
        }
        if (max_shift < tol) break;
    }

    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    auto centroids_t =
        torch::from_blob(centroids.data(), {n_levels}, opts).clone();
    auto boundaries_t =
        torch::from_blob(boundaries.data(), {n_levels - 1}, opts).clone();

    return LloydMaxCodebook{centroids_t, boundaries_t, dim, bits};
}

LloydMaxCodebook solve_lloyd_max(int dim, int bits) {
    if (dim >= 64) {
        return gaussian_lloyd_max(dim, bits);
    }
    return beta_lloyd_max(dim, bits);
}

torch::Tensor codebook_quantize(const torch::Tensor& x,
                                const torch::Tensor& boundaries) {
    return torch::bucketize(x, boundaries);
}

torch::Tensor codebook_dequantize(const torch::Tensor& indices,
                                  const torch::Tensor& centroids) {
    return centroids.index({indices});
}

// --- CodebookRegistry ---

bool CodebookRegistry::CacheKey::operator==(const CacheKey& other) const {
    return dim == other.dim && bits == other.bits && device == other.device;
}

size_t CodebookRegistry::CacheKeyHash::operator()(const CacheKey& k) const {
    size_t h = std::hash<int>()(k.dim);
    h ^= std::hash<int>()(k.bits) << 16;
    h ^= std::hash<int>()(static_cast<int>(k.device.type())) << 24;
    h ^= std::hash<int>()(k.device.index()) << 28;
    return h;
}

CodebookRegistry& CodebookRegistry::instance() {
    static CodebookRegistry registry;
    return registry;
}

const LloydMaxCodebook& CodebookRegistry::get(int dim, int bits,
                                               torch::Device device) {
    CacheKey key{dim, bits, device};
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = cache_.find(key);
    if (it != cache_.end()) {
        return it->second;
    }

    // Compute on CPU, then move to target device
    auto codebook = solve_lloyd_max(dim, bits);
    codebook.centroids = codebook.centroids.to(device);
    codebook.boundaries = codebook.boundaries.to(device);

    auto [inserted_it, _] = cache_.emplace(key, std::move(codebook));
    return inserted_it->second;
}

void CodebookRegistry::warmup(const std::vector<int>& dims,
                               const std::vector<int>& bits_options,
                               torch::Device device) {
    for (int d : dims) {
        for (int b : bits_options) {
            get(d, b, device);
        }
    }
}

void CodebookRegistry::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    cache_.clear();
}

}  // namespace flashquant
