# FlashLite

A lightweight, high-occupancy CUDA fused attention kernel optimized for low-VRAM, bandwidth-constrained NVIDIA Ampere GPUs (`sm_86`).

## Features

- **Tensor Core Acceleration**: Warp-level matrix operations using CUDA WMMA (`16x16x16`, FP16 input / FP32 accumulator).
- **Hardware-Tuned SRAM Tiling**: `64x64` block tiling designed to fit shared memory and maximize occupancy.
- **Online Softmax**: Register-level running statistics (FlashAttention-2 algorithm) avoiding full `N x N` materialization in global memory.
- **PyTorch Bindings**: Seamless integration via PyTorch C++/CUDA extension (`pybind11`).

---

## Prerequisites

- **GPU**: NVIDIA Ampere with compute capability `sm_86` (e.g., RTX 30-series, A2000, A4000)
- **CUDA Toolkit**: 11.8+ (`nvcc` on `$PATH`)
- **Host Compiler**: GCC >= 9.3 or Clang (C++17 support)
- **Python**: >= 3.8 with PyTorch (CUDA-enabled) and `pytest`

### Quick Environment Setup

```bash
# Create and activate environment
conda create -n flashlite python=3.10 -y
conda activate flashlite

# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install pytest
```

---

## Build & Installation

Install the extension in editable mode:

```bash
pip install -e .
```

Or compile inplace using `setup.py`:

```bash
python setup.py build_ext --inplace
```

---

## Quick Usage

```python
import math
import torch
import flash_attn_wmma

# Dimensions: [Batch, Heads, SeqLen, HeadDim] (FP16 on CUDA)
B, H, N, D = 2, 8, 1024, 64
q = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda")
k = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda")
v = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda")
sm_scale = 1.0 / math.sqrt(D)

# Run custom FlashAttention-2
out = flash_attn_wmma.forward_flash2(q, k, v, sm_scale, is_causal=False)
```

---

## Testing & Benchmarks

### Correctness Testing

Verify numerical accuracy against `torch.nn.functional.scaled_dot_product_attention`:

```bash
pytest -v tests/test_correctness.py
```

### Throughput Benchmarking

Benchmark latency and TFLOPS against PyTorch SDPA and FlashAttention:

```bash
python benchmarks/benchmark_throughput.py
```

---

## Architecture Overview
 
- **WMMA Tile**: `16x16x16` (`half` inputs, `float` accumulator).
- **Block Tile (`Br x Bc`)**: `64x64` tokens per threadblock (128 threads = 4 warps).
- **Online Softmax**: Iteratively tracks running row-max and normalizer statistics per tile to accumulate attention output directly into registers/SRAM, avoiding materializing intermediate `N x N` attention matrices in global memory (HBM).
