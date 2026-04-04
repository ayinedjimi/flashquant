// Copyright 2026 FlashQuant Authors. Apache-2.0 License.
#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "../core/packing.h"

namespace py = pybind11;
using namespace flashquant;

void bind_packing(py::module_& m) {
    auto p = m.def_submodule("packing", "Bit packing/unpacking utilities");

    p.def("nibble_pack", &nibble_pack,
          py::arg("indices"),
          "Pack pairs of 4-bit indices into uint8. Shape (..., D) -> (..., D/2).");

    p.def("nibble_unpack", &nibble_unpack,
          py::arg("packed"),
          "Unpack uint8 to pairs of 4-bit indices. Shape (..., D/2) -> (..., D).");

    p.def("pack_2bit", &pack_2bit,
          py::arg("indices"),
          "Pack 2-bit indices: 4 values per byte.");

    p.def("unpack_2bit", &unpack_2bit,
          py::arg("packed"), py::arg("dim"),
          "Unpack 2-bit packed indices.");

    p.def("validate_indices", &validate_indices,
          py::arg("indices"), py::arg("bits"),
          "Validate that all indices are in [0, 2^bits - 1].");
}
