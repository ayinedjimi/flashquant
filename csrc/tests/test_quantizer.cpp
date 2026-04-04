// Copyright 2026 FlashQuant Authors. Apache-2.0 License.
#include <gtest/gtest.h>
#include "quantizer.h"

using namespace flashquant;

class TurboQuantMSETest : public ::testing::TestWithParam<std::tuple<int, int>> {
protected:
    void SetUp() override {
        auto [dim, bits] = GetParam();
        quantizer_ = std::make_unique<TurboQuantMSE>(dim, bits, 42);
    }
    std::unique_ptr<TurboQuantMSE> quantizer_;
};

TEST_P(TurboQuantMSETest, RoundtripCosine) {
    auto [dim, bits] = GetParam();
    torch::manual_seed(42);
    auto x = torch::randn({32, 8, dim});

    auto [indices, norms] = quantizer_->quantize(x);
    auto x_hat = quantizer_->dequantize(indices, norms);

    // Cosine similarity
    auto flat_x = x.reshape({-1, dim}).to(torch::kFloat32);
    auto flat_xh = x_hat.reshape({-1, dim}).to(torch::kFloat32);
    auto cos = torch::nn::functional::cosine_similarity(
        flat_x, flat_xh,
        torch::nn::functional::CosineSimilarityFuncOptions().dim(-1));
    float mean_cos = cos.mean().item<float>();

    // Thresholds: 4-bit >= 0.95, 3-bit >= 0.92, 2-bit >= 0.80
    float threshold = (bits >= 4) ? 0.95f : (bits >= 3) ? 0.92f : 0.80f;
    EXPECT_GE(mean_cos, threshold)
        << "Cosine too low for dim=" << dim << " bits=" << bits
        << " got " << mean_cos;
}

TEST_P(TurboQuantMSETest, IndicesInRange) {
    auto [dim, bits] = GetParam();
    auto x = torch::randn({16, dim});
    auto [indices, norms] = quantizer_->quantize(x);
    int max_val = (1 << bits) - 1;
    EXPECT_GE(indices.min().item<int64_t>(), 0);
    EXPECT_LE(indices.max().item<int64_t>(), max_val);
}

TEST_P(TurboQuantMSETest, NormsPositive) {
    auto [dim, bits] = GetParam();
    auto x = torch::randn({16, dim});
    auto [indices, norms] = quantizer_->quantize(x);
    EXPECT_TRUE((norms >= 0).all().item<bool>());
}

TEST_P(TurboQuantMSETest, OutputShape) {
    auto [dim, bits] = GetParam();
    auto x = torch::randn({4, 8, 16, dim});
    auto [indices, norms] = quantizer_->quantize(x);
    EXPECT_EQ(indices.sizes(), x.sizes());
    auto expected_norms_shape = x.sizes().vec();
    expected_norms_shape.back() = 1;
    EXPECT_EQ(norms.sizes(), torch::IntArrayRef(expected_norms_shape));
}

INSTANTIATE_TEST_SUITE_P(
    Parametrized, TurboQuantMSETest,
    ::testing::Combine(
        ::testing::Values(64, 128, 256),
        ::testing::Values(2, 3, 4)
    ),
    [](const auto& info) {
        auto [dim, bits] = info.param;
        return "dim" + std::to_string(dim) + "_bits" + std::to_string(bits);
    }
);

TEST(TurboQuantProd, QJLSignsAreInt8) {
    TurboQuantProd prod(128, 4, 42);
    auto x = torch::randn({16, 128});
    auto result = prod.quantize(x);
    EXPECT_EQ(result.qjl_signs.dtype(), torch::kInt8);
}

TEST(TurboQuantProd, ResidualNormsPositive) {
    TurboQuantProd prod(128, 4, 42);
    auto x = torch::randn({16, 128});
    auto result = prod.quantize(x);
    EXPECT_TRUE((result.residual_norms >= 0).all().item<bool>());
}

TEST(TurboQuantProd, BitsAtLeast2) {
    EXPECT_THROW(TurboQuantProd(128, 1, 42), std::invalid_argument);
}

TEST(TurboQuantProd, InnerProductApproxUnbiased) {
    torch::manual_seed(42);
    TurboQuantProd prod(128, 4, 42);

    int N = 500;
    auto keys = torch::randn({N, 128});
    auto queries = torch::randn({N, 128});

    auto compressed = prod.quantize(keys);

    // True inner products
    auto true_ip = torch::sum(queries * keys, -1);

    // Estimated inner products
    auto est_ip = prod.estimate_inner_product(queries, compressed).squeeze(-1);

    // Relative bias should be small
    float mean_true = true_ip.mean().item<float>();
    float mean_est = est_ip.mean().item<float>();
    float relative_bias = std::abs(mean_est - mean_true) /
                          (std::abs(mean_true) + 1e-8f);
    EXPECT_LT(relative_bias, 0.10f)
        << "Inner product estimator biased: true=" << mean_true
        << " est=" << mean_est;
}
