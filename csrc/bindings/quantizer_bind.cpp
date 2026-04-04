// Copyright 2026 FlashQuant Authors. Apache-2.0 License.
#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "../core/quantizer.h"

namespace py = pybind11;
using namespace flashquant;

void bind_quantizer(py::module_& m) {
    auto q = m.def_submodule("quantizer", "TurboQuant quantizers");

    // QuantizedMSE result type
    py::class_<QuantizedMSE>(q, "QuantizedMSE")
        .def_readwrite("indices", &QuantizedMSE::indices)
        .def_readwrite("norms", &QuantizedMSE::norms);

    // QuantizedProd result type
    py::class_<QuantizedProd>(q, "QuantizedProd")
        .def_readwrite("indices", &QuantizedProd::indices)
        .def_readwrite("norms", &QuantizedProd::norms)
        .def_readwrite("qjl_signs", &QuantizedProd::qjl_signs)
        .def_readwrite("residual_norms", &QuantizedProd::residual_norms);

    // TurboQuantMSE
    py::class_<TurboQuantMSE>(q, "TurboQuantMSE")
        .def(py::init<int, int, int64_t>(),
             py::arg("dim"), py::arg("bits"), py::arg("seed") = 42)
        .def("quantize", &TurboQuantMSE::quantize,
             py::arg("x"),
             "Quantize input vectors. Returns QuantizedMSE(indices, norms).")
        .def("dequantize", &TurboQuantMSE::dequantize,
             py::arg("indices"), py::arg("norms"),
             "Reconstruct vectors from indices and norms.")
        .def_property_readonly("dim", &TurboQuantMSE::dim)
        .def_property_readonly("bits", &TurboQuantMSE::bits)
        .def("rotation", [](const TurboQuantMSE& self) {
            return self.rotation();
        });

    // TurboQuantProd
    py::class_<TurboQuantProd>(q, "TurboQuantProd")
        .def(py::init<int, int, int64_t>(),
             py::arg("dim"), py::arg("bits"), py::arg("seed") = 42)
        .def("quantize", &TurboQuantProd::quantize,
             py::arg("x"),
             "Quantize with QJL correction. Returns QuantizedProd.")
        .def("dequantize", &TurboQuantProd::dequantize,
             py::arg("indices"), py::arg("norms"),
             "MSE-only reconstruction (no QJL). Use estimate_inner_product() "
             "for attention scores.")
        .def("estimate_inner_product",
             &TurboQuantProd::estimate_inner_product,
             py::arg("query"), py::arg("compressed"),
             "Unbiased inner product estimation with QJL correction.")
        .def_property_readonly("dim",
                               [](const TurboQuantProd& self) {
                                   return self.mse_quantizer().dim();
                               })
        .def_property_readonly("bits",
                               [](const TurboQuantProd& self) {
                                   return self.mse_quantizer().bits() + 1;
                               });
}
