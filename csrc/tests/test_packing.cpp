// Copyright 2026 FlashQuant Authors. Apache-2.0 License.
#include <gtest/gtest.h>
#include "packing.h"

using namespace flashquant;

TEST(NibblePack, Roundtrip) {
    auto indices = torch::randint(0, 16, {32, 128});
    auto packed = nibble_pack(indices);
    auto unpacked = nibble_unpack(packed);
    EXPECT_TRUE(torch::equal(indices, unpacked));
}

TEST(NibblePack, Shape) {
    auto indices = torch::randint(0, 16, {8, 4, 128});
    auto packed = nibble_pack(indices);
    EXPECT_EQ(packed.size(-1), 64);
    EXPECT_EQ(packed.size(0), 8);
    EXPECT_EQ(packed.size(1), 4);
}

TEST(NibblePack, SpecificValues) {
    auto indices = torch::tensor({5, 3, 15, 0, 7, 9}, torch::kInt32);
    auto packed = nibble_pack(indices);
    // 5<<4|3 = 83, 15<<4|0 = 240, 7<<4|9 = 121
    auto expected = torch::tensor({83, 240, 121}, torch::kUInt8);
    EXPECT_TRUE(torch::equal(packed, expected));
}

TEST(NibblePack, OddDimRaises) {
    auto indices = torch::randint(0, 16, {5});
    EXPECT_THROW(nibble_pack(indices), c10::Error);
}

TEST(NibblePack, OverflowRaises) {
    auto indices = torch::tensor({16, 0}, torch::kInt32);
    EXPECT_THROW(nibble_pack(indices), std::invalid_argument);
}

TEST(NibblePack, NegativeRaises) {
    auto indices = torch::tensor({-1, 5}, torch::kInt32);
    EXPECT_THROW(nibble_pack(indices), std::invalid_argument);
}

TEST(Pack2Bit, Roundtrip) {
    auto indices = torch::randint(0, 4, {32, 128});
    auto packed = pack_2bit(indices);
    auto unpacked = unpack_2bit(packed, 128);
    EXPECT_TRUE(torch::equal(indices, unpacked));
}

TEST(Pack2Bit, Shape) {
    auto indices = torch::randint(0, 4, {8, 256});
    auto packed = pack_2bit(indices);
    EXPECT_EQ(packed.size(-1), 64);
}

TEST(Pack2Bit, OverflowRaises) {
    auto indices = torch::tensor({4, 0, 0, 0}, torch::kInt32);
    EXPECT_THROW(pack_2bit(indices), std::invalid_argument);
}

TEST(ValidateIndices, AcceptsValid) {
    auto indices = torch::randint(0, 16, {100});
    EXPECT_NO_THROW(validate_indices(indices, 4));
}

TEST(ValidateIndices, RejectsOutOfRange) {
    auto indices = torch::tensor({0, 8, 16}, torch::kInt32);
    EXPECT_THROW(validate_indices(indices, 4), std::invalid_argument);
}
