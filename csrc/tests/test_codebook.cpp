// Copyright 2026 FlashQuant Authors. Apache-2.0 License.
#include <gtest/gtest.h>
#include "codebook.h"

using namespace flashquant;

TEST(GaussianLloydMax, CentroidsMonotone) {
    for (int bits = 1; bits <= 4; ++bits) {
        auto cb = gaussian_lloyd_max(128, bits);
        auto c = cb.centroids.accessor<float, 1>();
        for (int i = 1; i < (1 << bits); ++i) {
            EXPECT_GT(c[i], c[i - 1])
                << "Centroids not monotone at bits=" << bits << " i=" << i;
        }
    }
}

TEST(GaussianLloydMax, BoundariesBetweenCentroids) {
    auto cb = gaussian_lloyd_max(128, 4);
    auto c = cb.centroids.accessor<float, 1>();
    auto b = cb.boundaries.accessor<float, 1>();
    for (int i = 0; i < 15; ++i) {
        EXPECT_GT(b[i], c[i]);
        EXPECT_LT(b[i], c[i + 1]);
    }
}

TEST(GaussianLloydMax, Symmetry) {
    auto cb = gaussian_lloyd_max(128, 4);
    auto c = cb.centroids.accessor<float, 1>();
    int n = 16;
    for (int i = 0; i < n / 2; ++i) {
        EXPECT_NEAR(c[i], -c[n - 1 - i], 1e-6)
            << "Centroids not symmetric at i=" << i;
    }
}

TEST(GaussianLloydMax, DifferentDimensions) {
    for (int dim : {16, 64, 128, 256}) {
        auto cb = gaussian_lloyd_max(dim, 4);
        EXPECT_EQ(cb.centroids.size(0), 16);
        EXPECT_EQ(cb.boundaries.size(0), 15);
        EXPECT_EQ(cb.dim, dim);
        EXPECT_EQ(cb.bits, 4);
    }
}

TEST(GaussianLloydMax, InvalidDim) {
    EXPECT_THROW(gaussian_lloyd_max(0, 4), std::invalid_argument);
}

TEST(GaussianLloydMax, InvalidBits) {
    EXPECT_THROW(gaussian_lloyd_max(128, 0), std::invalid_argument);
    EXPECT_THROW(gaussian_lloyd_max(128, 9), std::invalid_argument);
}

TEST(GaussianLloydMax, CentroidsMagnitudeScalesWithDim) {
    auto cb64 = gaussian_lloyd_max(64, 4);
    auto cb256 = gaussian_lloyd_max(256, 4);
    // sigma = 1/sqrt(d), so centroids for d=256 should be ~half of d=64
    float max_c64 = cb64.centroids.max().item<float>();
    float max_c256 = cb256.centroids.max().item<float>();
    EXPECT_GT(max_c64, max_c256);
}

TEST(SolveLloydMax, AutoDispatch) {
    // dim < 64 uses beta path, dim >= 64 uses gaussian
    auto cb_small = solve_lloyd_max(32, 4);
    auto cb_large = solve_lloyd_max(128, 4);
    EXPECT_EQ(cb_small.centroids.size(0), 16);
    EXPECT_EQ(cb_large.centroids.size(0), 16);
}

TEST(CodebookQuantize, IndicesInRange) {
    auto cb = gaussian_lloyd_max(128, 4);
    auto x = torch::randn({100, 128}) * (1.0f / std::sqrt(128.0f));
    auto indices = codebook_quantize(x, cb.boundaries);
    EXPECT_GE(indices.min().item<int64_t>(), 0);
    EXPECT_LE(indices.max().item<int64_t>(), 15);
}

TEST(CodebookDequantize, ReturnsCorrectValues) {
    auto cb = gaussian_lloyd_max(128, 4);
    auto indices = torch::randint(0, 16, {10, 128});
    auto values = codebook_dequantize(indices, cb.centroids);
    EXPECT_EQ(values.sizes(), indices.sizes());
    // Each value should be one of the 16 centroids
    auto c = cb.centroids.accessor<float, 1>();
    auto v = values.accessor<float, 2>();
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 128; ++j) {
            int idx = indices[i][j].item<int64_t>();
            EXPECT_FLOAT_EQ(v[i][j], c[idx]);
        }
    }
}

TEST(CodebookRegistry, CachesResults) {
    auto& reg = CodebookRegistry::instance();
    reg.clear();
    const auto& cb1 = reg.get(128, 4, torch::kCPU);
    const auto& cb2 = reg.get(128, 4, torch::kCPU);
    // Same pointer (cached)
    EXPECT_EQ(cb1.centroids.data_ptr(), cb2.centroids.data_ptr());
}

TEST(CodebookRegistry, DifferentKeysDifferentCodebooks) {
    auto& reg = CodebookRegistry::instance();
    reg.clear();
    const auto& cb3 = reg.get(128, 3, torch::kCPU);
    const auto& cb4 = reg.get(128, 4, torch::kCPU);
    EXPECT_NE(cb3.centroids.data_ptr(), cb4.centroids.data_ptr());
}
