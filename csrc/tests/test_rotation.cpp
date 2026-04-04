// Copyright 2026 FlashQuant Authors. Apache-2.0 License.
#include <gtest/gtest.h>
#include "rotation.h"

using namespace flashquant;

TEST(HaarOrthogonal, Orthogonality) {
    for (int dim : {64, 128, 256}) {
        auto R = haar_orthogonal(dim, 42);
        auto identity = torch::matmul(R.t(), R);
        auto eye = torch::eye(dim);
        EXPECT_TRUE(torch::allclose(identity, eye, /*rtol=*/1e-5, /*atol=*/1e-5))
            << "R^T R != I for dim=" << dim;
    }
}

TEST(HaarOrthogonal, Determinism) {
    auto R1 = haar_orthogonal(128, 42);
    auto R2 = haar_orthogonal(128, 42);
    EXPECT_TRUE(torch::equal(R1, R2));
}

TEST(HaarOrthogonal, DifferentSeeds) {
    auto R1 = haar_orthogonal(128, 42);
    auto R2 = haar_orthogonal(128, 99);
    EXPECT_FALSE(torch::equal(R1, R2));
}

TEST(HaarOrthogonal, Shape) {
    auto R = haar_orthogonal(128, 42);
    EXPECT_EQ(R.dim(), 2);
    EXPECT_EQ(R.size(0), 128);
    EXPECT_EQ(R.size(1), 128);
    EXPECT_EQ(R.dtype(), torch::kFloat32);
}

TEST(HaarOrthogonal, InvalidDim) {
    EXPECT_THROW(haar_orthogonal(0, 42), std::invalid_argument);
}

TEST(SplitRotation, Shapes) {
    auto R = haar_orthogonal(128, 42);
    auto [even, odd] = split_rotation(R);
    EXPECT_EQ(even.size(0), 128);
    EXPECT_EQ(even.size(1), 64);
    EXPECT_EQ(odd.size(0), 128);
    EXPECT_EQ(odd.size(1), 64);
}

TEST(SplitRotation, Contiguous) {
    auto R = haar_orthogonal(128, 42);
    auto [even, odd] = split_rotation(R);
    EXPECT_TRUE(even.is_contiguous());
    EXPECT_TRUE(odd.is_contiguous());
}

TEST(SplitRotation, ReconstructsRotationT) {
    auto R = haar_orthogonal(128, 42);
    auto [even, odd] = split_rotation(R);
    auto Rt = R.t().contiguous();

    // even = Rt[:, ::2], odd = Rt[:, 1::2]
    auto Rt_even = Rt.index({torch::indexing::Slice(),
                             torch::indexing::Slice(0, torch::indexing::None, 2)});
    auto Rt_odd = Rt.index({torch::indexing::Slice(),
                            torch::indexing::Slice(1, torch::indexing::None, 2)});

    EXPECT_TRUE(torch::allclose(even, Rt_even, 1e-6, 1e-6));
    EXPECT_TRUE(torch::allclose(odd, Rt_odd, 1e-6, 1e-6));
}
