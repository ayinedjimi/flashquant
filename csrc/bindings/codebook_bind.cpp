// Copyright 2026 FlashQuant Authors. Apache-2.0 License.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include "../core/codebook.h"

namespace py = pybind11;
using namespace flashquant;

void bind_codebook(py::module_& m) {
    auto cb = m.def_submodule("codebook", "Lloyd-Max codebook operations");

    py::class_<LloydMaxCodebook>(cb, "LloydMaxCodebook")
        .def_readwrite("centroids", &LloydMaxCodebook::centroids)
        .def_readwrite("boundaries", &LloydMaxCodebook::boundaries)
        .def_readonly("dim", &LloydMaxCodebook::dim)
        .def_readonly("bits", &LloydMaxCodebook::bits);

    cb.def("gaussian_lloyd_max", &gaussian_lloyd_max,
           py::arg("dim"), py::arg("bits"),
           "Closed-form Lloyd-Max for Gaussian N(0, 1/d). No scipy needed.");

    cb.def("beta_lloyd_max", &beta_lloyd_max,
           py::arg("dim"), py::arg("bits"),
           py::arg("max_iter") = 200, py::arg("tol") = 1e-10,
           "Iterative Lloyd-Max for Beta distribution (dim < 64).");

    cb.def("solve_lloyd_max", &solve_lloyd_max,
           py::arg("dim"), py::arg("bits"),
           "Auto-dispatch: Gaussian for dim >= 64, Beta for dim < 64.");

    cb.def("quantize", &codebook_quantize,
           py::arg("x"), py::arg("boundaries"),
           "Quantize values using pre-computed boundaries.");

    cb.def("dequantize", &codebook_dequantize,
           py::arg("indices"), py::arg("centroids"),
           "Dequantize indices to centroid values.");

    py::class_<CodebookRegistry>(cb, "CodebookRegistry")
        .def_static("instance", &CodebookRegistry::instance,
                     py::return_value_policy::reference)
        .def("get", &CodebookRegistry::get,
             py::arg("dim"), py::arg("bits"), py::arg("device"))
        .def("warmup", &CodebookRegistry::warmup,
             py::arg("dims"), py::arg("bits_options"), py::arg("device"))
        .def("clear", &CodebookRegistry::clear);
}
