// Copyright 2026 FlashQuant Authors. Apache-2.0 License.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

namespace py = pybind11;

// Forward declarations from other binding files
void bind_codebook(py::module_& m);
void bind_quantizer(py::module_& m);
void bind_packing(py::module_& m);

PYBIND11_MODULE(_C, m) {
    m.doc() = "FlashQuant C++/CUDA backend — TurboQuant KV cache compression";

    bind_codebook(m);
    bind_quantizer(m);
    bind_packing(m);

    // CUDA kernel bindings added when CUDA is available
#ifdef FLASHQUANT_CUDA
    void bind_kernels(py::module_& m);
    bind_kernels(m);
#endif
}
