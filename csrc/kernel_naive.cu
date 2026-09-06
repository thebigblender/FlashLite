#include "flash_attn.h"
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cmath>

namespace flash_attn {

// ============================================================================
// Warp and Block Level Reduction Primitives
// ============================================================================

/**
 * @brief Intra-warp maximum reduction using warp shuffle down intrinsics.
 *        Computes the maximum float value across all 32 active threads in the warp.
 */
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, mask));
    }
    return val;
}

/**
 * @brief Intra-warp sum reduction using warp shuffle down intrinsics.
 *        Computes the total sum across all 32 active threads in the warp.
 */
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, mask);
    }
    return val;
}

/**
 * @brief Block-wide maximum reduction across all warps in the CTA.
 *        1. Each warp performs an intra-warp reduction via shuffle instructions.
 *        2. Lane 0 of each warp writes its partial max to shared memory scratchpad.
 *        3. Warp 0 performs the final reduction and broadcasts the global maximum.
 */
__device__ __forceinline__ float block_reduce_max(float val, float* s_scratch) {
    const int lane = threadIdx.x % WARP_SIZE;
    const int wid  = threadIdx.x / WARP_SIZE;
    const int num_warps = blockDim.x / WARP_SIZE;

    // Step 1: Intra-warp reduction
    val = warp_reduce_max(val);

    // Step 2: Write warp leader values to shared memory
    if (lane == 0) {
        s_scratch[wid] = val;
    }
    __syncthreads();

    // Step 3: Warp 0 aggregates partial results from all warps
    if (wid == 0) {
        float warp_val = (lane < num_warps) ? s_scratch[lane] : -1e30f;
        warp_val = warp_reduce_max(warp_val);
        if (lane == 0) {
            s_scratch[0] = warp_val; // Global block maximum
        }
    }
    __syncthreads();

    return s_scratch[0];
}

/**
 * @brief Block-wide sum reduction across all warps in the CTA.
 *        Aggregates thread-local sums to compute the total normalizer L_i.
 */
__device__ __forceinline__ float block_reduce_sum(float val, float* s_scratch) {
    const int lane = threadIdx.x % WARP_SIZE;
    const int wid  = threadIdx.x / WARP_SIZE;
    const int num_warps = blockDim.x / WARP_SIZE;

    // Step 1: Intra-warp reduction
    val = warp_reduce_sum(val);

    // Step 2: Write warp leader values to shared memory
    if (lane == 0) {
        s_scratch[wid] = val;
    }
    __syncthreads();

    // Step 3: Warp 0 aggregates partial results from all warps
    if (wid == 0) {
        float warp_val = (lane < num_warps) ? s_scratch[lane] : 0.0f;
        warp_val = warp_reduce_sum(warp_val);
        if (lane == 0) {
            s_scratch[0] = warp_val; // Global block sum
        }
    }
    __syncthreads();

    return s_scratch[0];
}

// ============================================================================
// Naive Scaled Dot-Product Attention CUDA Kernel
// ============================================================================
/**
 * @brief Standard un-fused Scaled Dot-Product Attention baseline kernel.
 * 
 * Mathematical Formulation (Vaswani et al., 2017 "Attention Is All You Need", §3.2.1):
 *   Attention(Q, K, V) = softmax( (Q * K^T) / sqrt(d_k) ) * V
 * 
 * Standard Multi-Pass Execution Model (Dao et al., 2022 "FlashAttention", §2):
 *   1. S_ij = (Q_i . K_j) * sm_scale                  [GEMM 1: Query-Key Scores]
 *   2. m_i  = max_{j} S_ij                            [Row Maximum for Numerical Stability]
 *   3. P_ij = exp(S_ij - m_i)                         [Exponentiation]
 *   4. L_i  = sum_{j} P_ij                            [Row Normalizer]
 *   5. O_id = sum_{j} (P_ij / L_i) * V_jd             [GEMM 2: Value Projection]
 * 
 * Execution Mapping:
 *   - Grid:  dim3(seq_len, num_heads, batch_size)
 *            Each thread block (CTA) is dedicated to computing exactly 1 output query token row (Q_i).
 *   - Block: 128 threads (4 warps) collaborating on dot-products, reductions, and output projection.
 *   - Shared Memory Layout:
 *            s_scratch [32 floats]   : Reduction scratchpad for warp leaders
 *            s_scores  [seq_len floats] : Score buffer P_ij for row i
 *            s_Q       [head_dim halfs] : Cached Query vector Q_i
 */
__global__ void naive_attention_kernel(
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
    // ------------------------------------------------------------------------
    // 1. Grid & Index Decomposition
    // ------------------------------------------------------------------------
    const int q_idx     = blockIdx.x; // Query token row index i in [0, seq_len - 1]
    const int head_idx  = blockIdx.y; // Head index h in [0, num_heads - 1]
    const int batch_idx = blockIdx.z; // Batch index b in [0, batch_size - 1]
    const int tid       = threadIdx.x;

    if (batch_idx >= batch_size || head_idx >= num_heads || q_idx >= seq_len) return;

    // Strided memory offsets for 4D tensor layout: [Batch, Heads, SeqLen, HeadDim]
    const int head_stride = seq_len * head_dim;
    const int batch_head_offset = (batch_idx * num_heads + head_idx) * head_stride;

    const half* q_row_ptr = Q + batch_head_offset + q_idx * head_dim;
    const half* k_base_ptr = K + batch_head_offset;
    const half* v_base_ptr = V + batch_head_offset;
    half* out_row_ptr = O + batch_head_offset + q_idx * head_dim;

    // ------------------------------------------------------------------------
    // 2. Dynamic Shared Memory Partitioning
    // ------------------------------------------------------------------------
    extern __shared__ char smem_raw[];
    float* s_scratch = reinterpret_cast<float*>(smem_raw);                      // [32 * sizeof(float)]
    float* s_scores  = reinterpret_cast<float*>(s_scratch + 32);                 // [seq_len * sizeof(float)]
    half*  s_Q       = reinterpret_cast<half*>(reinterpret_cast<char*>(s_scores + seq_len)); // [head_dim * sizeof(half)]

    // ------------------------------------------------------------------------
    // 3. Step 1: Cache Query Vector Q_i into Shared Memory
    // ------------------------------------------------------------------------
    for (int d = tid; d < head_dim; d += blockDim.x) {
        s_Q[d] = q_row_ptr[d];
    }
    __syncthreads();

    // ------------------------------------------------------------------------
    // 4. Step 2: Compute Scaled Dot-Products S_ij = (Q_i . K_j) * sm_scale
    //            and track thread-local maximum for numerical stability
    // ------------------------------------------------------------------------
    const int max_k_len = is_causal ? (q_idx + 1) : seq_len;
    float thread_max = -1e30f;

    for (int j = tid; j < seq_len; j += blockDim.x) {
        if (is_causal && j > q_idx) {
            // Mask out future tokens in causal auto-regressive attention
            s_scores[j] = -1e30f;
        } else {
            const half* k_j_ptr = k_base_ptr + j * head_dim;
            float dot_product = 0.0f;

            #pragma unroll 4
            for (int d = 0; d < head_dim; ++d) {
                dot_product += __half2float(s_Q[d]) * __half2float(k_j_ptr[d]);
            }

            float score = dot_product * sm_scale;
            s_scores[j] = score;
            thread_max = fmaxf(thread_max, score);
        }
    }

    // ------------------------------------------------------------------------
    // 5. Step 3: Numerically Stable Softmax (Max Subtraction & Normalization)
    // ------------------------------------------------------------------------
    // Compute row maximum m_i across the entire block
    const float row_max = block_reduce_max(thread_max, s_scratch);

    // Compute exponentiated scores P_ij = exp(S_ij - m_i) and accumulate sum
    float thread_sum = 0.0f;
    for (int j = tid; j < seq_len; j += blockDim.x) {
        if (j < max_k_len) {
            float p_ij = __expf(s_scores[j] - row_max);
            s_scores[j] = p_ij;
            thread_sum += p_ij;
        } else {
            s_scores[j] = 0.0f;
        }
    }

    // Compute row normalizer sum L_i = sum_j P_ij across the block
    const float row_sum = block_reduce_sum(thread_sum, s_scratch);
    const float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;

    __syncthreads();

    // ------------------------------------------------------------------------
    // 6. Step 4: Value Projection & Accumulation O_id = sum_j (P_ij / L_i) * V_jd
    // ------------------------------------------------------------------------
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float out_acc = 0.0f;

        #pragma unroll 4
        for (int j = 0; j < max_k_len; ++j) {
            float prob = s_scores[j] * inv_sum;
            const half* v_j_ptr = v_base_ptr + j * head_dim;
            out_acc += prob * __half2float(v_j_ptr[d]);
        }

        out_row_ptr[d] = __float2half(out_acc);
    }
}

// ============================================================================
// Host Entrypoint & Validation
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
    // 3. Grid, Block, and Shared Memory Sizing
    // ------------------------------------------------------------------------
    // Grid: (seq_len, num_heads, batch_size) -> 1 CTA per query token row
    dim3 grid(seq_len, num_heads, batch_size);
    dim3 block(BLOCK_THREADS); // 128 threads (4 warps)

    // Dynamic Shared Memory:
    //  - 32 floats for reduction scratchpad
    //  - seq_len floats for intermediate score vector s_scores
    //  - head_dim halfs for cached query vector s_Q
    const size_t smem_bytes = 32 * sizeof(float) +
                              seq_len * sizeof(float) +
                              head_dim * sizeof(half);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    // ------------------------------------------------------------------------
    // 4. Kernel Launch
    // ------------------------------------------------------------------------
    naive_attention_kernel<<<grid, block, smem_bytes, stream>>>(
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

