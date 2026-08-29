cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(flash_attn_wmma LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Target NVIDIA Ampere architecture (sm_86)
set(CMAKE_CUDA_ARCHITECTURES 86)

# Locate Python and PyTorch
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Attempt to locate Torch CMake configuration
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE TORCH_PREFIX_RESULT
)

if (TORCH_PREFIX_RESULT EQUAL 0 AND EXISTS "${TORCH_CMAKE_PREFIX_PATH}")
    list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX_PATH}")
    find_package(Torch REQUIRED)
else()
    message(WARNING "Torch cmake prefix path could not be detected automatically. Please set CMAKE_PREFIX_PATH manually.")
endif()

# Compiler optimization flags
set(CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS}
    -O3
    --use_fast_math
    -U__CUDA_NO_HALF_OPERATORS__
    -U__CUDA_NO_HALF_CONVERSIONS__
    -U__CUDA_NO_HALF2_OPERATORS__
)

# Source files
set(SOURCES
    csrc/flash_attn.cpp
    csrc/kernel_naive.cu
    csrc/kernel_flash2.cu
)

# Include directories
set(INCLUDES
    csrc/includes
    ${TORCH_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
)

# Build Python C++ Extension Module
add_library(flash_attn_wmma MODULE ${SOURCES})

target_include_directories(flash_attn_wmma PRIVATE ${INCLUDES})

target_compile_options(flash_attn_wmma PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>: -O3 -Wall>
    $<$<COMPILE_LANGUAGE:CUDA>: -O3 --use_fast_math -arch=sm_86>
)

# Link Torch and CUDA libraries
target_link_libraries(flash_attn_wmma PRIVATE ${TORCH_LIBRARIES} Python3::Python)

# Pybind11 / Python module properties
set_target_properties(flash_attn_wmma PROPERTIES
    PREFIX ""
    SUFFIX ".so"
    CUDA_SEPARABLE_COMPILATION OFF
)
