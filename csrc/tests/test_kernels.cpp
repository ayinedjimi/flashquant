// Copyright 2026 FlashQuant Authors. Apache-2.0 License.
// Placeholder for CUDA kernel tests — requires GPU build
#include <gtest/gtest.h>

#ifdef FLASHQUANT_CUDA

// TODO: Add CUDA kernel tests when building with -DFLASHQUANT_CUDA=ON
TEST(CUDAKernels, Placeholder) {
    GTEST_SKIP() << "CUDA kernel tests require GPU build";
}

#else

TEST(CUDAKernels, SkippedNoCUDA) {
    GTEST_SKIP() << "CUDA not enabled in this build";
}

#endif
