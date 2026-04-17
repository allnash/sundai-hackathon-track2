#include <torch/extension.h>

std::vector<torch::Tensor> quantize_int4_custom(torch::Tensor input, int group_size);
torch::Tensor gemm_int4_custom(torch::Tensor A_packed, torch::Tensor B_packed,
                                torch::Tensor scales_A, torch::Tensor scales_B, int group_size);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quantize_int4", &quantize_int4_custom,
          "INT4 quantization (your solution)",
          py::arg("input"), py::arg("group_size") = 64);
    m.def("gemm_int4", &gemm_int4_custom,
          "INT4 GEMM (your solution)",
          py::arg("A_packed"), py::arg("B_packed"),
          py::arg("scales_A"), py::arg("scales_B"),
          py::arg("group_size") = 64);
}
