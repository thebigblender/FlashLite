import os
from pathlib import Path
from setuptools import setup
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

# Ensure CUDA is available
if CUDA_HOME is None:
    raise EnvironmentError(
        "CUDA_HOME is not set. Please ensure the CUDA Toolkit is installed and accessible."
    )

root_dir = Path(__file__).resolve().parent
csrc_dir = root_dir / "csrc"

sources = [
    str(csrc_dir / "flash_attn.cpp"),
    str(csrc_dir / "kernel_naive.cu"),
    str(csrc_dir / "kernel_flash2.cu"),
]

include_dirs = [
    str(csrc_dir / "includes"),
]

# NVCC compiler flags targeting NVIDIA Ampere sm_86
nvcc_flags = [
    "-O3",
    "-arch=sm_86",
    "--use_fast_math",
    "-std=c++17",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
]

# Host C++ compiler flags
cxx_flags = [
    "-O3",
    "-std=c++17",
]

setup(
    name="flash_attn_wmma",
    version="0.1.0",
    author="AI Accelerator Architecture Lab",
    description="Custom lightweight FlashAttention-2 implementation using CUDA WMMA on Ampere (sm_86)",
    ext_modules=[
        CUDAExtension(
            name="flash_attn_wmma",
            sources=sources,
            include_dirs=include_dirs,
            extra_compile_args={
                "cxx": cxx_flags,
                "nvcc": nvcc_flags,
            },
        )
    ],
    cmdclass={
        "build_ext": BuildExtension.with_options(no_python_abi_suffix=False)
    },
    python_requires=">=3.8",
)
