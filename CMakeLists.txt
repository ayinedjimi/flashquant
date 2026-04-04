cmake_minimum_required(VERSION 3.20)
project(flashquant LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find PyTorch
find_package(Torch REQUIRED)
find_package(pybind11 REQUIRED)

# ---- Core library (C++ only, no CUDA) ----
add_library(flashquant_core STATIC
    csrc/core/codebook.cpp
    csrc/core/rotation.cpp
    csrc/core/quantizer.cpp
    csrc/core/packing.cpp
)
target_include_directories(flashquant_core PUBLIC csrc/core)
target_link_libraries(flashquant_core PUBLIC ${TORCH_LIBRARIES})

# ---- CUDA kernels (optional) ----
option(FLASHQUANT_CUDA "Build CUDA kernels" OFF)

if(FLASHQUANT_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    find_package(CUDAToolkit REQUIRED)

    add_library(flashquant_cuda STATIC
        csrc/cuda/compress.cu
        csrc/cuda/decompress.cu
        csrc/cuda/flash_attention.cu
        csrc/cuda/fused_tq_attention.cu
        csrc/cuda/paged_decode.cu
        csrc/cuda/split_k_reduce.cu
    )
    target_include_directories(flashquant_cuda PUBLIC csrc/cuda)
    target_link_libraries(flashquant_cuda PUBLIC flashquant_core CUDA::cudart)
    set_target_properties(flashquant_cuda PROPERTIES
        CUDA_ARCHITECTURES "80;86;89;90"
        CUDA_SEPARABLE_COMPILATION ON
    )
    target_compile_definitions(flashquant_cuda PUBLIC FLASHQUANT_CUDA)
endif()

# ---- Python bindings ----
pybind11_add_module(_C
    csrc/bindings/module.cpp
    csrc/bindings/codebook_bind.cpp
    csrc/bindings/quantizer_bind.cpp
    csrc/bindings/packing_bind.cpp
)
target_link_libraries(_C PRIVATE flashquant_core)

if(FLASHQUANT_CUDA)
    target_link_libraries(_C PRIVATE flashquant_cuda)
    target_compile_definitions(_C PRIVATE FLASHQUANT_CUDA)
endif()

# Install the Python extension module
install(TARGETS _C LIBRARY DESTINATION flashquant)

# ---- C++ Tests (optional) ----
option(FLASHQUANT_TESTS "Build C++ tests" OFF)

if(FLASHQUANT_TESTS)
    enable_testing()
    find_package(GTest QUIET)

    if(GTest_FOUND)
        add_executable(flashquant_tests
            csrc/tests/test_codebook.cpp
            csrc/tests/test_rotation.cpp
            csrc/tests/test_quantizer.cpp
            csrc/tests/test_packing.cpp
        )
        target_link_libraries(flashquant_tests
            flashquant_core
            GTest::gtest_main
        )
        if(FLASHQUANT_CUDA)
            target_sources(flashquant_tests PRIVATE csrc/tests/test_kernels.cpp)
            target_link_libraries(flashquant_tests flashquant_cuda)
        endif()

        include(GoogleTest)
        gtest_discover_tests(flashquant_tests)
    else()
        message(WARNING "GTest not found, skipping C++ tests")
    endif()
endif()
