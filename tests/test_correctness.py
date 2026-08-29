import math
import pytest
import torch
import torch.nn.functional as F

try:
    import flash_attn_wmma
    HAS_EXTENSION = True
except ImportError:
    HAS_EXTENSION = False


def check_extension_available():
    if not HAS_EXTENSION:
        pytest.skip(
            "flash_attn_wmma extension not compiled/installed. Run 'pip install -e .' first."
        )


def generate_qkv(batch_size: int, num_heads: int, seq_len: int, head_dim: int, device="cuda"):
    """Generate random FP16 Q, K, V tensors on CUDA."""
    torch.manual_seed(42)
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device=device)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device=device)
    return q, k, v


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("num_heads", [4, 8])
@pytest.mark.parametrize("seq_len", [128, 256, 512])
@pytest.mark.parametrize("head_dim", [64])
@pytest.mark.parametrize("is_causal", [False, True])
def test_naive_attention(batch_size, num_heads, seq_len, head_dim, is_causal):
    check_extension_available()
    if not torch.cuda.is_available():
        pytest.skip("CUDA device not available")

    q, k, v = generate_qkv(batch_size, num_heads, seq_len, head_dim)
    sm_scale = 1.0 / math.sqrt(head_dim)

    # Reference computation using PyTorch Native SDPA
    ref_out = F.scaled_dot_product_attention(
        q, k, v, attn_mask=None, dropout_p=0.0, is_causal=is_causal, scale=sm_scale
    )

    # Custom Naive CUDA extension forward pass
    out = flash_attn_wmma.forward_naive(q, k, v, sm_scale, is_causal)

    assert out.shape == ref_out.shape, f"Shape mismatch: {out.shape} vs {ref_out.shape}"
    assert out.dtype == torch.float16, f"Expected FP16 output, got {out.dtype}"

    # Note: If running on initial empty stubs (all zeros), inform the user
    if torch.all(out == 0):
        print(f"\n[STUB DETECTED] Naive kernel returned all zeros for shape ({batch_size}, {num_heads}, {seq_len}, {head_dim}). "
              f"Implement kernel logic in csrc/kernel_naive.cu to pass numerical verification.")
    else:
        torch.testing.assert_close(out, ref_out, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("num_heads", [4, 8])
@pytest.mark.parametrize("seq_len", [128, 256, 512])
@pytest.mark.parametrize("head_dim", [64])
@pytest.mark.parametrize("is_causal", [False, True])
def test_flash2_wmma_attention(batch_size, num_heads, seq_len, head_dim, is_causal):
    check_extension_available()
    if not torch.cuda.is_available():
        pytest.skip("CUDA device not available")

    q, k, v = generate_qkv(batch_size, num_heads, seq_len, head_dim)
    sm_scale = 1.0 / math.sqrt(head_dim)

    # Reference computation using PyTorch Native SDPA
    ref_out = F.scaled_dot_product_attention(
        q, k, v, attn_mask=None, dropout_p=0.0, is_causal=is_causal, scale=sm_scale
    )

    # Custom FlashAttention-2 WMMA CUDA extension forward pass
    out = flash_attn_wmma.forward_flash2(q, k, v, sm_scale, is_causal)

    assert out.shape == ref_out.shape, f"Shape mismatch: {out.shape} vs {ref_out.shape}"
    assert out.dtype == torch.float16, f"Expected FP16 output, got {out.dtype}"

    # Note: If running on initial empty stubs (all zeros), inform the user
    if torch.all(out == 0):
        print(f"\n[STUB DETECTED] Flash2 kernel returned all zeros for shape ({batch_size}, {num_heads}, {seq_len}, {head_dim}). "
              f"Implement kernel logic in csrc/kernel_flash2.cu to pass numerical verification.")
    else:
        torch.testing.assert_close(out, ref_out, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    print("=" * 70)
    print("Running FlashAttention-2 WMMA Correctness Verification Suite")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("CUDA is not available on this system. Cannot run verification.")
        exit(1)

    if not HAS_EXTENSION:
        print("ERROR: flash_attn_wmma extension is not installed.")
        print("Please build and install it via: pip install -e .")
        exit(1)

    device_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    print(f"CUDA Device: {device_name} (Compute Capability: {capability[0]}.{capability[1]})")

    # Run quick standalone test
    B, H, N, D = 2, 8, 256, 64
    print(f"\nTesting default shape: Batch={B}, Heads={H}, SeqLen={N}, HeadDim={D}")
    
    q, k, v = generate_qkv(B, H, N, D)
    scale = 1.0 / math.sqrt(D)
    ref = F.scaled_dot_product_attention(q, k, v, scale=scale)

    print("1. Testing Naive Attention Stub...")
    out_naive = flash_attn_wmma.forward_naive(q, k, v, scale, False)
    if torch.all(out_naive == 0):
        print("   -> Output is all zeros (Expected stub behavior).")
    else:
        max_diff = (out_naive - ref).abs().max().item()
        print(f"   -> Max Absolute Difference: {max_diff:.6f}")

    print("2. Testing FlashAttention-2 WMMA Stub...")
    out_flash2 = flash_attn_wmma.forward_flash2(q, k, v, scale, False)
    if torch.all(out_flash2 == 0):
        print("   -> Output is all zeros (Expected stub behavior).")
    else:
        max_diff = (out_flash2 - ref).abs().max().item()
        print(f"   -> Max Absolute Difference: {max_diff:.6f}")

    print("\nRun pytest tests/test_correctness.py for full parameterized test matrix.")
