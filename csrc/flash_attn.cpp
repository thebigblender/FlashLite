#include "flash_attn.h"
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Custom FlashAttention-2 CUDA WMMA Extension for Mobile Ampere (sm_86)";

    m.def(
        "forward_naive",
        &flash_attn::flash_attn_forward_naive,
        "Naive Attention Forward (CUDA Baseline)",
        py::arg("q"),
        py::arg("k"),
        py::arg("v"),
        py::arg("sm_scale") = 0.0,
        py::arg("is_causal") = false
    );

    m.def(
        "forward_flash2",
        &flash_attn::flash_attn_forward_flash2,
        "FlashAttention-2 Forward with WMMA (CUDA sm_86)",
        py::arg("q"),
        py::arg("k"),
        py::arg("v"),
        py::arg("sm_scale") = 0.0,
        py::arg("is_causal") = false
    );
}
