import math
import time
from typing import Callable, Optional
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

# Try importing custom extension
try:
    import flash_attn_wmma
    HAS_CUSTOM_EXT = True
except ImportError:
    HAS_CUSTOM_EXT = False

# Try importing official FlashAttention-2
try:
    from flash_attn import flash_attn_func as official_flash_attn_func
    HAS_OFFICIAL_FLASH = True
except ImportError:
    HAS_OFFICIAL_FLASH = False


def calculate_flops(batch_size: int, num_heads: int, seq_len: int, head_dim: int, is_causal: bool = False) -> float:
    """
    Theoretical FLOPs for Multi-Head Attention forward pass:
    - QK^T GEMM: 2 * B * H * N * N * D
    - Softmax * V GEMM: 2 * B * H * N * N * D
    Total = 4 * B * H * N^2 * D (or halved for causal mask)
    """
    multiplier = 2.0 if is_causal else 4.0
    return multiplier * batch_size * num_heads * (seq_len ** 2) * head_dim


def measure_latency_ms(fn: Callable, warmup: int = 25, runs: int = 100) -> float:
    """Measure median CUDA execution latency in milliseconds using torch.cuda.Event."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # Timing
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(runs)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(runs)]

    for i in range(runs):
        start_events[i].record()
        fn()
        end_events[i].record()

    torch.cuda.synchronize()

    latencies = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    latencies.sort()
    return latencies[len(latencies) // 2]  # Median latency


def run_benchmarks(
    batch_size: int = 2,
    num_heads: int = 8,
    head_dim: int = 64,
    seq_lengths = (512, 1024, 2048, 4096),
    is_causal: bool = False
):
    if not torch.cuda.is_available():
        print("CUDA is not available. Benchmarking requires a GPU.")
        return

    device = torch.device("cuda:0")
    device_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)

    print("=" * 110)
    print(f" FlashAttention-2 WMMA Throughput Benchmark Suite")
    print(f" Target Device: {device_name} (sm_{capability[0]}{capability[1]})")
    print(f" Config: Batch={batch_size}, Heads={num_heads}, HeadDim={head_dim}, Causal={is_causal}")
    print("=" * 110)

    header = (
        f"{'SeqLen':>8} | "
        f"{'PyTorch FA2 (ms)':>17} | {'PT-FA2 TFLOPS':>13} | "
        f"{'Official FA2 (ms)':>17} | {'FA2 TFLOPS':>11} | "
        f"{'Naive CUDA (ms)':>15} | {'Naive TFLOPS':>12} | "
        f"{'Custom WMMA (ms)':>16} | {'WMMA TFLOPS':>12}"
    )
    print(header)
    print("-" * len(header))

    sm_scale = 1.0 / math.sqrt(head_dim)

    for seq_len in seq_lengths:
        flops = calculate_flops(batch_size, num_heads, seq_len, head_dim, is_causal)

        # Allocate FP16 tensors
        q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device=device)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device=device)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device=device)

        # 1. PyTorch Native SDPA (Strict FlashAttention-2 Backend)
        def sdpa_fn():
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
                return F.scaled_dot_product_attention(
                    q, k, v, attn_mask=None, dropout_p=0.0, is_causal=is_causal, scale=sm_scale
                )
        sdpa_ms = measure_latency_ms(sdpa_fn)
        sdpa_tflops = (flops / (sdpa_ms * 1e-3)) / 1e12

        # 2. Official FlashAttention-2 (if installed)
        if HAS_OFFICIAL_FLASH:
            # Official FlashAttention expects [Batch, SeqLen, Heads, HeadDim]
            q_fa = q.transpose(1, 2).contiguous()
            k_fa = k.transpose(1, 2).contiguous()
            v_fa = v.transpose(1, 2).contiguous()
            fa_fn = lambda: official_flash_attn_func(q_fa, k_fa, v_fa, softmax_scale=sm_scale, causal=is_causal)
            try:
                fa_ms = measure_latency_ms(fa_fn)
                fa_tflops = (flops / (fa_ms * 1e-3)) / 1e12
                fa_ms_str = f"{fa_ms:17.3f}"
                fa_tflops_str = f"{fa_tflops:11.2f}"
            except Exception as e:
                fa_ms_str = f"{'ERROR':>17}"
                fa_tflops_str = f"{'N/A':>11}"
        else:
            fa_ms_str = f"{'N/A (not installed)':>17}"
            fa_tflops_str = f"{'N/A':>11}"

        # 3. Custom Naive CUDA Attention
        if HAS_CUSTOM_EXT:
            naive_fn = lambda: flash_attn_wmma.forward_naive(q, k, v, sm_scale, is_causal)
            naive_ms = measure_latency_ms(naive_fn)
            naive_tflops = (flops / (naive_ms * 1e-3)) / 1e12
            naive_ms_str = f"{naive_ms:15.3f}"
            naive_tflops_str = f"{naive_tflops:12.2f}"
        else:
            naive_ms_str = f"{'N/A':>15}"
            naive_tflops_str = f"{'N/A':>12}"

        # 4. Custom WMMA FlashAttention-2
        if HAS_CUSTOM_EXT:
            wmma_fn = lambda: flash_attn_wmma.forward_flash2(q, k, v, sm_scale, is_causal)
            wmma_ms = measure_latency_ms(wmma_fn)
            wmma_tflops = (flops / (wmma_ms * 1e-3)) / 1e12
            wmma_ms_str = f"{wmma_ms:16.3f}"
            wmma_tflops_str = f"{wmma_tflops:12.2f}"
        else:
            wmma_ms_str = f"{'N/A':>16}"
            wmma_tflops_str = f"{'N/A':>12}"

        print(
            f"{seq_len:>8} | "
            f"{sdpa_ms:17.3f} | {sdpa_tflops:12.2f} | "
            f"{fa_ms_str} | {fa_tflops_str} | "
            f"{naive_ms_str} | {naive_tflops_str} | "
            f"{wmma_ms_str} | {wmma_tflops_str}"
        )

    print("=" * 110)
    print("Notes:")
    print(" - TFLOPS = (FLOPs / (latency_ms * 1e-3)) / 1e12, where FLOPs = 4 * B * H * N^2 * D")
    print(" - Custom WMMA uses fused online softmax with Ampere sm_86 Tensor Core instructions.")

    # ------------------------------------------------------------------------
    # Memory Consumption Comparison (O(N^2) vs O(N))
    # ------------------------------------------------------------------------
    print("\n" + "=" * 85)
    print(f" Peak Memory Allocation Benchmark (Intermediate Activation Overhead)")
    print("=" * 85)
    print(f"{'SeqLen':>8} | {'Un-fused PyTorch (MB)':>22} | {'FlashLite WMMA (MB)':>20} | {'Memory Reduction':>18}")
    print("-" * 85)

    def unfused_attn(q_in, k_in, v_in):
        scores = torch.matmul(q_in, k_in.transpose(-2, -1)) * sm_scale
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, v_in)

    for seq_len in seq_lengths:
        q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device=device)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device=device)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device=device)

        # Baseline: Un-fused attention materializing [B, H, N, N]
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        _ = unfused_attn(q, k, v)
        torch.cuda.synchronize()
        mem_unfused = torch.cuda.max_memory_allocated() / (1024 ** 2)

        # FlashLite WMMA (fused, O(N))
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        if HAS_CUSTOM_EXT:
            _ = flash_attn_wmma.forward_flash2(q, k, v, sm_scale, is_causal)
        torch.cuda.synchronize()
        mem_flash = torch.cuda.max_memory_allocated() / (1024 ** 2)

        ratio_str = f"{mem_unfused / mem_flash:.1f}x" if mem_flash > 0 else "N/A"
        print(f"{seq_len:>8} | {mem_unfused:>20.2f} MB | {mem_flash:>18.2f} MB | {ratio_str:>18}")

    print("=" * 85)


if __name__ == "__main__":
    # 1. Non-Causal Attention Benchmark
    run_benchmarks(
        batch_size=2,
        num_heads=8,
        head_dim=64,
        seq_lengths=[512, 1024, 2048, 4096],
        is_causal=False
    )

    # 2. Causal Attention Benchmark
    run_benchmarks(
        batch_size=2,
        num_heads=8,
        head_dim=64,
        seq_lengths=[512, 1024, 2048, 4096],
        is_causal=True
    )

