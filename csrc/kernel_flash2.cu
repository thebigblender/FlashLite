#include "flash_attn.h"
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <c10/cuda/CUDAStream.h>

namespace flash_attn {

using namespace nvcuda::wmma;

// ============================================================================
// Helper: In-Register Accumulator Rescaling (Ampere sm_86 WMMA Layout)
// ============================================================================
/**
 * @brief Rescales a 16x16 FP32 accumulator fragment across rows.
 * 
 * On NVIDIA Ampere (sm_86), thread lane_id in [0..31] holds elements for:
 *   - row r0 = lane_id / 4 (elements 0, 1, 4, 5)
 *   - row r1 = (lane_id / 4) + 8 (elements 2, 3, 6, 7)
 */
__device__ __forceinline__ void rescale_acc(
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float>& acc,
    float rescale_r0,
    float rescale_r1
) {
    acc.x[0] *= rescale_r0;
    acc.x[1] *= rescale_r0;
    acc.x[2] *= rescale_r1;
    acc.x[3] *= rescale_r1;
    acc.x[4] *= rescale_r0;
    acc.x[5] *= rescale_r0;
    acc.x[6] *= rescale_r1;
    acc.x[7] *= rescale_r1;
}

// ============================================================================
// Custom FlashAttention-2 WMMA CUDA Kernel (sm_86 Ampere)
// ============================================================================
/**
 * @brief FlashAttention-2 CUDA Kernel using Tensor Core WMMA instructions.
 * 
 * Target Architecture: Compute Capability sm_86 (NVIDIA Ampere)
 * Execution Model:
 *  - Grid:  (dim3((seq_len + BLOCK_M - 1) / BLOCK_M, num_heads, batch_size))
 *  - Block: 128 threads (4 warps: warp_id in [0..3], lane_id in [0..31])
 * 
 * Tiling:
 *  - Query Tile (Br): 64 tokens (16 rows per warp)
 *  - Key/Value Tile (Bc): 64 tokens
 *  - Head Dimension (D): 64 channels
 */
template <int Br = BLOCK_M, int Bc = BLOCK_N, int HeadDim = DEFAULT_HEAD_DIM>
__global__ void flash_attn2_wmma_kernel(
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
    // Thread / Warp Indexing
    // ------------------------------------------------------------------------
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE; // 0..3 (4 warps per CTA)
    const int lane_id = tid % WARP_SIZE; // 0..31

    const int block_m_idx = blockIdx.x;  // Query tile index (Br chunk)
    const int head_idx    = blockIdx.y;  // Attention head index
    const int batch_idx   = blockIdx.z;  // Batch index

    if (batch_idx >= batch_size || head_idx >= num_heads) return;

    // ------------------------------------------------------------------------
    // Shared Memory Allocation (48 KB Total, statically allocated)
    // ------------------------------------------------------------------------
    __shared__ half  s_Q[Br * HeadDim];  // 8 KB
    __shared__ half  s_K[Bc * HeadDim];  // 8 KB
    __shared__ half  s_V[Bc * HeadDim];  // 8 KB
    __shared__ float s_S[Br * Bc];       // 16 KB (intermediate scores & output buffer)
    __shared__ half  s_P[Br * Bc];       // 8 KB (attention probabilities for GEMM 2)

    // Strided memory offsets for 4D tensor layout: [Batch, Heads, SeqLen, HeadDim]
    const int head_stride = seq_len * head_dim;
    const int batch_head_offset = (batch_idx * num_heads + head_idx) * head_stride;

    const half* q_base_ptr = Q + batch_head_offset;
    const half* k_base_ptr = K + batch_head_offset;
    const half* v_base_ptr = V + batch_head_offset;
    half*       o_base_ptr = O + batch_head_offset;

    // ------------------------------------------------------------------------
    // Step 1: Cooperative Load Q Block (Br x HeadDim) using 128-bit Vector Loads
    // ------------------------------------------------------------------------
    const int q_start = block_m_idx * Br;
    #pragma unroll
    for (int step = 0; step < 4; ++step) {
        int elem_idx = step * (BLOCK_THREADS * 8) + tid * 8; // 8 halfs = 16 bytes = uint4
        int r = elem_idx / HeadDim;
        int d = elem_idx % HeadDim;
        int global_row = q_start + r;

        uint4 loaded = make_uint4(0, 0, 0, 0);
        if (global_row < seq_len) {
            loaded = *reinterpret_cast<const uint4*>(q_base_ptr + global_row * head_dim + d);
        }
        *reinterpret_cast<uint4*>(s_Q + elem_idx) = loaded;
    }
    __syncthreads();

    // ------------------------------------------------------------------------
    // Step 2: Initialize Output Accumulators and Softmax Statistics
    // Each warp handles 16 rows: [warp_id * 16, warp_id * 16 + 15]
    // ------------------------------------------------------------------------
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_o[4];
    #pragma unroll
    for (int d = 0; d < 4; ++d) {
        fill_fragment(acc_o[d], 0.0f);
    }

    // Row indices owned by this thread in the 16x16 WMMA accumulator fragment
    const int r0 = lane_id / 4;     // 0..7
    const int r1 = r0 + 8;          // 8..15

    float m_prev[16];
    float l_prev[16];
    #pragma unroll
    for (int r = 0; r < 16; ++r) {
        m_prev[r] = -1e30f;
        l_prev[r] = 0.0f;
    }

    // Total Key/Value tiles
    const int num_tiles_c = (seq_len + Bc - 1) / Bc;
    // In causal attention, tiles where all keys are strictly ahead of queries are skipped
    const int max_tiles_c = is_causal
        ? min(num_tiles_c, (q_start + Br - 1) / Bc + 1)
        : num_tiles_c;

    // ------------------------------------------------------------------------
    // Step 3: Outer Loop over Key / Value Tiles
    // ------------------------------------------------------------------------
    for (int j = 0; j < max_tiles_c; ++j) {
        const int k_start = j * Bc;

        // 3.1: Cooperative Vectorized Load K_j, V_j into Shared Memory
        #pragma unroll
        for (int step = 0; step < 4; ++step) {
            int elem_idx = step * (BLOCK_THREADS * 8) + tid * 8;
            int r = elem_idx / HeadDim;
            int d = elem_idx % HeadDim;
            int global_row = k_start + r;

            uint4 k_val = make_uint4(0, 0, 0, 0);
            uint4 v_val = make_uint4(0, 0, 0, 0);
            if (global_row < seq_len) {
                k_val = *reinterpret_cast<const uint4*>(k_base_ptr + global_row * head_dim + d);
                v_val = *reinterpret_cast<const uint4*>(v_base_ptr + global_row * head_dim + d);
            }
            *reinterpret_cast<uint4*>(s_K + elem_idx) = k_val;
            *reinterpret_cast<uint4*>(s_V + elem_idx) = v_val;
        }
        __syncthreads();

        // 3.2: Compute Attention Scores S_ij = Q_i * K_j^T using WMMA
        fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_s[4];
        #pragma unroll
        for (int n = 0; n < 4; ++n) {
            fill_fragment(acc_s[n], 0.0f);
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> frag_q;
                fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, col_major> frag_k;
                load_matrix_sync(frag_q, &s_Q[(warp_id * 16) * HeadDim + k * 16], HeadDim);
                load_matrix_sync(frag_k, &s_K[(n * 16) * HeadDim + k * 16], HeadDim);
                mma_sync(acc_s[n], frag_q, frag_k, acc_s[n]);
            }
            // Store S tile to shared memory for softmax processing
            store_matrix_sync(&s_S[(warp_id * 16) * Bc + n * 16], acc_s[n], Bc, mem_row_major);
        }

        // 3.3: Softmax Scaling, Causal Masking, Online Statistics & Exponentiation
        float rescale_r0 = 1.0f;
        float rescale_r1 = 1.0f;

        for (int r = 0; r < 16; ++r) {
            const int global_row = q_start + warp_id * 16 + r;
            const int col0 = lane_id;
            const int col1 = lane_id + 32;

            const int global_col0 = k_start + col0;
            const int global_col1 = k_start + col1;

            bool valid0 = (global_col0 < seq_len) && (!is_causal || global_col0 <= global_row) && (global_row < seq_len);
            bool valid1 = (global_col1 < seq_len) && (!is_causal || global_col1 <= global_row) && (global_row < seq_len);

            float val0 = valid0 ? (s_S[(warp_id * 16 + r) * Bc + col0] * sm_scale) : -1e30f;
            float val1 = valid1 ? (s_S[(warp_id * 16 + r) * Bc + col1] * sm_scale) : -1e30f;

            // Intra-warp reduction for row max
            float max_val = fmaxf(val0, val1);
            #pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1) {
                max_val = fmaxf(max_val, __shfl_xor_sync(0xffffffff, max_val, mask));
            }
            float row_max = max_val;

            // Online softmax update
            float m_new = fmaxf(m_prev[r], row_max);
            float rescale = (m_prev[r] > -1e20f) ? __expf(m_prev[r] - m_new) : 0.0f;

            if (r == r0) rescale_r0 = rescale;
            if (r == r1) rescale_r1 = rescale;

            float p0 = valid0 ? __expf(val0 - m_new) : 0.0f;
            float p1 = valid1 ? __expf(val1 - m_new) : 0.0f;

            // Intra-warp reduction for row sum
            float sum_val = p0 + p1;
            #pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1) {
                sum_val += __shfl_xor_sync(0xffffffff, sum_val, mask);
            }
            float row_sum = sum_val;

            float l_new = l_prev[r] * rescale + row_sum;
            m_prev[r] = m_new;
            l_prev[r] = l_new;

            // Store P tile as FP16 into s_P for GEMM 2
            s_P[(warp_id * 16 + r) * Bc + col0] = __float2half(p0);
            s_P[(warp_id * 16 + r) * Bc + col1] = __float2half(p1);
        }

        // 3.4: Rescale Previous Output Accumulators in Registers
        #pragma unroll
        for (int d = 0; d < 4; ++d) {
            rescale_acc(acc_o[d], rescale_r0, rescale_r1);
        }

        // 3.5: Accumulate O_i += P_ij * V_j using WMMA
        #pragma unroll
        for (int d = 0; d < 4; ++d) {
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> frag_p;
                fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, row_major> frag_v;
                load_matrix_sync(frag_p, &s_P[(warp_id * 16) * Bc + k * 16], Bc);
                load_matrix_sync(frag_v, &s_V[(k * 16) * HeadDim + d * 16], HeadDim);
                mma_sync(acc_o[d], frag_p, frag_v, acc_o[d]);
            }
        }

        __syncthreads();
    } // End loop over Key/Value tiles

    // ------------------------------------------------------------------------
    // Step 4: Final Output Normalization (O_i = O_i / l_final)
    // ------------------------------------------------------------------------
    float inv_l0 = (l_prev[r0] > 0.0f) ? (1.0f / l_prev[r0]) : 0.0f;
    float inv_l1 = (l_prev[r1] > 0.0f) ? (1.0f / l_prev[r1]) : 0.0f;

    #pragma unroll
    for (int d = 0; d < 4; ++d) {
        rescale_acc(acc_o[d], inv_l0, inv_l1);
        // Store normalized output to shared memory s_S (reused buffer)
        store_matrix_sync(&s_S[(warp_id * 16) * HeadDim + d * 16], acc_o[d], HeadDim, mem_row_major);
    }
    __syncthreads();

    // ------------------------------------------------------------------------
    // Step 5: Write Normalized Output from Shared Memory to Global Memory (Vectorized)
    // ------------------------------------------------------------------------
    #pragma unroll
    for (int step = 0; step < 4; ++step) {
        int elem_idx = step * (BLOCK_THREADS * 8) + tid * 8;
        int r = elem_idx / HeadDim;
        int d = elem_idx % HeadDim;
        int global_row = q_start + r;

        if (global_row < seq_len) {
            half h_out[8];
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                h_out[k] = __float2half(s_S[elem_idx + k]);
            }
            *reinterpret_cast<uint4*>(o_base_ptr + global_row * head_dim + d) =
                *reinterpret_cast<uint4*>(h_out);
        }
    }
}

// ============================================================================
// Host Entrypoint & Validation
// ============================================================================
torch::Tensor flash_attn_forward_flash2(
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
    TORCH_CHECK(head_dim == DEFAULT_HEAD_DIM,
                "FlashLite WMMA kernel currently requires head_dim == 64");

    if (sm_scale == 0.0) {
        sm_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    }

    // ------------------------------------------------------------------------
    // 2. Allocate Output Tensor
    // ------------------------------------------------------------------------
    auto out = torch::empty_like(q);

    // ------------------------------------------------------------------------
    // 3. Grid & Block Dimension Setup (128 threads / 4 warps per CTA)
    // ------------------------------------------------------------------------
    const int grid_m = (seq_len + BLOCK_M - 1) / BLOCK_M;
    dim3 grid(grid_m, num_heads, batch_size);
    dim3 block(BLOCK_THREADS); // 128 threads (4 warps)

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    // ------------------------------------------------------------------------
    // 4. Kernel Launch Dispatch
    // ------------------------------------------------------------------------
    flash_attn2_wmma_kernel<BLOCK_M, BLOCK_N, DEFAULT_HEAD_DIM><<<grid, block, 0, stream>>>(
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
