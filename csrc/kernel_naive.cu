#include "flash_attn.h"
#include <torch/extension.h>
#include <cuda_fp16.h>

namespace flash_attn {

// ============================================================================
// Naive Attention Dummy CUDA Kernel (Baseline)
// ============================================================================
/**
 * @brief Baseline naive attention kernel stub.
 *        In full implementation, this performs un-fused attention:
 *        S = Q * K^T * sm_scale -> P = softmax(S) -> O = P * V
 */
__global__ void naive_attention_kernel_stub(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float sm_scale,
    bool is_causal
) {
    // Grid configuration:
    // blockIdx.x = query token tile index
    // blockIdx.y = head index
    // blockIdx.z = batch index
    int tid = threadIdx.x;
    int head_idx = blockIdx.y;
    int batch_idx = blockIdx.z;

    if (batch_idx >= batch_size || head_idx >= num_heads) return;

    // Dummy operation: initialize output tile to zero if needed
    // The user will replace this with baseline standard global-memory GEMM/Softmax logic.
    int batch_head_offset = (batch_idx * num_heads + head_idx) * seq_len * head_dim;
    for (int i = tid; i < seq_len * head_dim; i += blockDim.x) {
        O[batch_head_offset + i] = __float2half(0.0f);
    }
}

// ============================================================================
// Host Entrypoint & Validation Stub
// ============================================================================
torch::Tensor flash_attn_forward_naive(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    double sm_scale,
    bool is_causal
) {
    // ------------------------------------------------------------------------
    // 1. Tensor Input Validations
    // ------------------------------------------------------------------------
    TORCH_CHECK(q.is_cuda(), "Query tensor must be on CUDA");
    TORCH_CHECK(k.is_cuda(), "Key tensor must be on CUDA");
    TORCH_CHECK(v.is_cuda(), "Value tensor must be on CUDA");

    TORCH_CHECK(q.is_contiguous(), "Query tensor must be contiguous");
    TORCH_CHECK(k.is_contiguous(), "Key tensor must be contiguous");
    TORCH_CHECK(v.is_contiguous(), "Value tensor must be contiguous");

    TORCH_CHECK(q.dtype() == torch::kHalf, "Query tensor must be FP16 (Half)");
    TORCH_CHECK(k.dtype() == torch::kHalf, "Key tensor must be FP16 (Half)");
    TORCH_CHECK(v.dtype() == torch::kHalf, "Value tensor must be FP16 (Half)");

    TORCH_CHECK(q.dim() == 4, "Query tensor must have 4 dimensions [B, H, N, D]");
    TORCH_CHECK(k.dim() == 4, "Key tensor must have 4 dimensions [B, H, N, D]");
    TORCH_CHECK(v.dim() == 4, "Value tensor must have 4 dimensions [B, H, N, D]");

    const int batch_size = q.size(0);
    const int num_heads  = q.size(1);
    const int seq_len    = q.size(2);
    const int head_dim   = q.size(3);

    TORCH_CHECK(k.size(0) == batch_size && k.size(1) == num_heads && k.size(3) == head_dim,
                "Key dimensions must match Query dimensions [B, H, N_k, D]");
    TORCH_CHECK(v.size(0) == batch_size && v.size(1) == num_heads && v.size(2) == k.size(2) && v.size(3) == head_dim,
                "Value dimensions must match Key dimensions [B, H, N_k, D]");

    if (sm_scale == 0.0) {
        sm_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    }

    // ------------------------------------------------------------------------
    // 2. Allocate Output Tensor
    // ------------------------------------------------------------------------
    auto out = torch::empty_like(q);

    // ------------------------------------------------------------------------
    // 3. Grid & Block Dimension Setup
    // ------------------------------------------------------------------------
    dim3 grid(1, num_heads, batch_size);
    dim3 block(BLOCK_THREADS); // 128 threads per block

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    // ------------------------------------------------------------------------
    // 4. Kernel Launch
    // ------------------------------------------------------------------------
    naive_attention_kernel_stub<<<grid, block, 0, stream>>>(
        reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(k.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(v.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        static_cast<float>(sm_scale),
        is_causal
    );

    CUDA_CHECK(cudaGetLastError());

    return out;
}

} // namespace flash_attn
