# FlashLite

**Ultra-Fast, Lightweight FlashAttention-2 Engine for Mobile & Consumer NVIDIA GPUs (`sm_86`)**

FlashLite is a ground-up re-architecture of the FlashAttention-2 algorithm designed specifically for the physical constraints of consumer and mobile NVIDIA Ampere silicon (e.g., RTX 3050 Ti / 3060 / 4050 / 4060 Laptop GPUs, Jetson Orin).

By eliminating datacenter CUTLASS bloat and replacing it with hand-tuned CUDA WMMA Tensor Core instructions, in-register online softmax, bank-conflict-free shared memory padding, and dual-granularity dispatching, FlashLite achieves **17.32 TFLOPS sustained throughput — beating PyTorch's native FlashAttention-2 implementation** on mobile hardware.

---

## Benchmark Highlights (RTX 3050 Ti Laptop GPU, 20 SMs, 42.5W TGP)

### Non-Causal Attention ($B=2, H=8, D=64, \text{FP16}$)
| Sequence Length | PyTorch SDPA Latency | PyTorch SDPA TFLOPS | FlashLite WMMA Latency | FlashLite WMMA TFLOPS | Speedup vs. PyTorch |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **512** | 0.108 ms | 9.99 TFLOPS | **0.080 ms** | **13.44 TFLOPS** | **1.35x (+35% faster)** |
| **1024** | 0.276 ms | 15.53 TFLOPS | **0.274 ms** | **15.65 TFLOPS** | **1.01x (Faster)** |
| **2048** | 1.051 ms | 16.35 TFLOPS | **1.072 ms** | **16.03 TFLOPS** | **0.98x (Parity)** |
| **4096** | 4.104 ms | 16.74 TFLOPS | **3.967 ms** | **17.32 TFLOPS** | **1.03x (Beats native FA2)** |

### Causal Attention ($B=2, H=8, D=64, \text{FP16}$)
| Sequence Length | PyTorch SDPA Latency | PyTorch SDPA TFLOPS | FlashLite WMMA Latency | FlashLite WMMA TFLOPS | Speedup vs. PyTorch |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **512** | 0.072 ms | 7.46 TFLOPS | **0.056 ms** | **9.53 TFLOPS** | **1.28x (+28% faster)** |
| **1024** | 0.186 ms | 11.52 TFLOPS | **0.173 ms** | **12.39 TFLOPS** | **1.08x (+8% faster)** |
| **2048** | 0.657 ms | 13.07 TFLOPS | **0.601 ms** | **14.30 TFLOPS** | **1.09x (+9% faster)** |
| **4096** | 2.354 ms | 14.60 TFLOPS | **2.153 ms** | **15.96 TFLOPS** | **1.09x (Beats native FA2)** |

### Peak Hardware Roofline Efficiency
Under sustained profiling on an OEM 42.5W / 1485 MHz power envelope, NVIDIA's proprietary cuBLAS pure matrix multiplication ($4096^3$) reaches **18.01 TFLOPS**.  
**FlashLite reaches 17.32 TFLOPS — operating at 96.2% of the physical hardware GEMM ceiling.**

---

## Peak Memory Reduction ($O(N)$ vs. $O(N^2)$)

| Sequence Length | Un-fused PyTorch (MB) | FlashLite WMMA (MB) | Memory Reduction Factor |
|:---:|:---:|:---:|:---:|
| **512** | 36.12 MB | 21.12 MB | **1.7x** |
| **1024** | 89.12 MB | 26.12 MB | **3.4x** |
| **2048** | 290.12 MB | 36.12 MB | **8.0x** |
| **4096** | 1076.12 MB | 56.12 MB | **19.2x** |

FlashLite eliminates intermediate $N \times N$ attention matrix materialization, slashing activation memory from **>1 GB down to 56 MB** at $N=4096$.

---

## Core Technical Innovations

1. **In-Register Online Softmax (Zero SRAM Roundtrips)**:
   - Transformed score accumulators `acc_s` directly into probability fragments `frag_p` in registers using exact Ampere lane coordinates ($\text{Row } r_0 = \text{lane\_id}/4, r_1 = r_0 + 8$).
   - Replaced warp-wide reductions with fast 2-step `__shfl_xor_sync` (masks 1, 2) across the 4 lanes holding each row.
2. **Shared Memory Bank-Conflict Elimination (`PADDED_HEAD_DIM = 72`)**:
   - Padded rows by 8 halfs (16 bytes = 4 banks), eliminating 16-way bank conflicts during WMMA matrix loads.
3. **Single-Cycle Hardware `EX2` SFU Intrinsics**:
   - Pre-scaled scores by $\log_2(e)$ to emit native inline PTX `"ex2.approx.f32 %0, %1;"`, replacing multi-cycle polynomial `__expf`.
4. **Direct In-Register Global Writeout**:
   - Output normalization and `half2` vectorization occur directly in registers, writing straight to global memory and eliminating 2 `__syncthreads()` barrier delays.
5. **DRAM Traffic Halving via Query Chunking ($B_r = 128$)**:
   - For long sequences ($N \ge 2048$), doubles Query chunk size to $B_r = 128$ (256 threads, 8 warps), cutting passes over $K$ and $V$ matrices in half and reducing GDDR6 traffic by 50%.
6. **Causal `full_tile` Unmasked Fast Path**:
   - 97% of causal tiles are strictly lower-triangular; partitioned execution into branchless unconditional scaling & exp, removing 32 predicated instructions per thread per tile.
7. **Adaptive Dispatcher**:
   - Automatically switches between fine-grained $B_r = 64$ ($N \le 1024$, maximizing 20-SM wave occupancy) and $B_r = 128$ ($N \ge 2048$, minimizing memory bus traffic).

---

## Requirements

- **GPU**: NVIDIA Ampere Architecture (`sm_86`: RTX 3050/3060/3070/3080, A2000, A4000, Jetson Orin)
- **CUDA Toolkit**: 11.8+ (`nvcc` on `$PATH`)
- **Host Compiler**: GCC >= 9.3 or Clang (C++17 support)
- **Python**: >= 3.8 with PyTorch (CUDA-enabled)

---

## Build & Installation

FlashLite compiles in **under 5 seconds** with zero external CUTLASS dependencies:

```bash
# Build inplace
python setup.py build_ext --inplace

# Or install as an editable package
pip install -e .
```

Verify compilation:
```bash
# Check PTXAS report (0 bytes stack frame, 0 register spills!)
pytest -v tests/test_correctness.py
```

---

## Quick Usage

### Standard Call
```python
import math
import torch
import flash_attn_wmma

# Dimensions: [Batch, Heads, SeqLen, HeadDim] (FP16 on CUDA)
B, H, N, D = 2, 8, 4096, 64
q = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda")
k = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda")
v = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda")
sm_scale = 1.0 / math.sqrt(D)

# Non-Causal Forward Pass
out = flash_attn_wmma.forward_flash2(q, k, v, sm_scale=sm_scale, is_causal=False)

# Causal Forward Pass (Autoregressive)
out_causal = flash_attn_wmma.forward_flash2(q, k, v, sm_scale=sm_scale, is_causal=True)
```

### Zero-Allocation Buffer Interface (Optimized for Serving / Decode Loops)
```python
# Pre-allocate output buffer once
out_buffer = torch.empty_like(q)

# Execute attention with zero memory allocation overhead
flash_attn_wmma.forward_flash2(q, k, v, sm_scale=sm_scale, is_causal=False, out=out_buffer)
```

---

## Testing & Benchmarking

### 1. Correctness Testing
Runs 48 automated test configurations comparing FlashLite against PyTorch's native SDPA reference:
```bash
pytest -v tests/test_correctness.py
```

### 2. Throughput & Memory Benchmarks
Measures median latency, TFLOPS, and peak activation memory across sequence lengths 512, 1024, 2048, and 4096 for both causal and non-causal attention:
```bash
python benchmarks/benchmark_throughput.py
```

---

## License

MIT License.
