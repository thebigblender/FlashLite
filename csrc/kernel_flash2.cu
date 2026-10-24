#include "flash_attn.h"
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <c10/cuda/CUDAStream.h>

namespace flash_attn {

using namespace nvcuda::wmma;

// ============================================================================
// Architecture Constants & Tuning Parameters (sm_86 Ampere)
// ============================================================================
constexpr int PAD_DIM = 8;
constexpr int PADDED_HEAD_DIM = DEFAULT_HEAD_DIM + PAD_DIM; // 72 elements = 144 bytes (+4 banks/row)
constexpr float LOG2_E = 1.44269504088896340736f;           // log2(e) for single-cycle EX2 hardware SFU

/**
 * @brief Fast single-cycle base-2 exponential using native NVIDIA hardware EX2 SFU.
 * Emits ex2.approx.f32 instruction directly, replacing multi-cycle __expf.
 */
__device__ __forceinline__ float fast_exp2(float x) {
    float res;
    asm("ex2.approx.f32 %0, %1;" : "=f"(res) : "f"(x));
    return res;
}

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
// Kernel 1: FlashAttention-2 WMMA Br=64 Bc=64 (Optimized for N <= 1024)
// 128 threads (4 warps), 27.6 KB SRAM, maximizes CTA occupancy on 20 SMs
// ============================================================================
template <bool IsCausal = false>
__global__ void __launch_bounds__(BLOCK_THREADS, 3) flash_attn2_wmma_br64_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float sm_scale_log2
) {
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE; // 0..3 (4 warps)
    const int lane_id = tid % WARP_SIZE; // 0..31

    const int block_m_idx = blockIdx.x;  // Query tile index (Br=64 chunk)
    const int head_idx    = blockIdx.y;  // Attention head index
    const int batch_idx   = blockIdx.z;  // Batch index

    if (batch_idx >= batch_size || head_idx >= num_heads) return;

    // Shared Memory: 64x72 padded tiles (27.6 KB total)
    struct SmemStorage {
        half s_Q[BLOCK_M * PADDED_HEAD_DIM]; // 9.2 KB
        half s_K[BLOCK_N * PADDED_HEAD_DIM]; // 9.2 KB
        half s_V[BLOCK_N * PADDED_HEAD_DIM]; // 9.2 KB
    };
    __shared__ SmemStorage smem;

    const int head_stride = seq_len * head_dim;
    const int batch_head_offset = (batch_idx * num_heads + head_idx) * head_stride;

    const half* q_base_ptr = Q + batch_head_offset;
    const half* k_base_ptr = K + batch_head_offset;
    const half* v_base_ptr = V + batch_head_offset;
    half*       o_base_ptr = O + batch_head_offset;

    // Step 1: Cooperative Load Q Block (Br x HeadDim)
    const int q_start = block_m_idx * BLOCK_M;
    #pragma unroll
    for (int step = 0; step < 4; ++step) {
        int vec_idx = step * BLOCK_THREADS + tid; // 0..511 uint4 vectors
        int r = vec_idx >> 3;                     // Row: 0..63
        int d = (vec_idx & 7) << 3;               // Col: 0..56
        int global_row = q_start + r;

        uint4 loaded = make_uint4(0, 0, 0, 0);
        if (global_row < seq_len) {
            loaded = *reinterpret_cast<const uint4*>(q_base_ptr + global_row * head_dim + d);
        }
        *reinterpret_cast<uint4*>(smem.s_Q + r * PADDED_HEAD_DIM + d) = loaded;
    }
    __syncthreads();

    // Hoist Query fragments into registers
    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> frag_q[4];
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        load_matrix_sync(frag_q[k], &smem.s_Q[(warp_id * 16) * PADDED_HEAD_DIM + k * 16], PADDED_HEAD_DIM);
    }

    // Step 2: Initialize Output Accumulators and Softmax Statistics
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_o[4];
    #pragma unroll
    for (int d = 0; d < 4; ++d) {
        fill_fragment(acc_o[d], 0.0f);
    }

    const int r0 = lane_id / 4;          // Local row 0..7
    const int r1 = r0 + 8;               // Local row 8..15
    const int c_base = (lane_id % 4) * 2; // Base col: 0, 2, 4, 6

    const int global_q_r0 = q_start + warp_id * 16 + r0;
    const int global_q_r1 = q_start + warp_id * 16 + r1;

    float m_prev0 = -1e30f;
    float m_prev1 = -1e30f;
    float l_prev0 = 0.0f;
    float l_prev1 = 0.0f;

    const int num_tiles_c = (seq_len + BLOCK_N - 1) / BLOCK_N;
    const int max_tiles_c = IsCausal
        ? min(num_tiles_c, (q_start + BLOCK_M - 1) / BLOCK_N + 1)
        : num_tiles_c;

    // Step 3: Outer Loop over Key / Value Tiles
    for (int j = 0; j < max_tiles_c; ++j) {
        const int k_start = j * BLOCK_N;

        // 3.1: Cooperative Vectorized Load K_j, V_j
        #pragma unroll
        for (int step = 0; step < 4; ++step) {
            int vec_idx = step * BLOCK_THREADS + tid;
            int r = vec_idx >> 3;
            int d = (vec_idx & 7) << 3;
            int global_row = k_start + r;

            uint4 k_val = make_uint4(0, 0, 0, 0);
            uint4 v_val = make_uint4(0, 0, 0, 0);
            if (global_row < seq_len) {
                k_val = *reinterpret_cast<const uint4*>(k_base_ptr + global_row * head_dim + d);
                v_val = *reinterpret_cast<const uint4*>(v_base_ptr + global_row * head_dim + d);
            }
            *reinterpret_cast<uint4*>(smem.s_K + r * PADDED_HEAD_DIM + d) = k_val;
            *reinterpret_cast<uint4*>(smem.s_V + r * PADDED_HEAD_DIM + d) = v_val;
        }
        __syncthreads();

        // 3.2: GEMM 1: S = Q * K^T (using hoisted frag_q)
        fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_s[4];
        #pragma unroll
        for (int n = 0; n < 4; ++n) {
            fill_fragment(acc_s[n], 0.0f);
        }

        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            #pragma unroll
            for (int n = 0; n < 4; ++n) {
                fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, col_major> frag_k;
                load_matrix_sync(frag_k, &smem.s_K[(n * 16) * PADDED_HEAD_DIM + k * 16], PADDED_HEAD_DIM);
                mma_sync(acc_s[n], frag_q[k], frag_k, acc_s[n]);
            }
        }

        // 3.3: Softmax Scaling & Masking (Partitioned: Full Tile vs Diagonal/Boundary Tile)
        const bool is_diagonal = IsCausal && (j == block_m_idx);
        const bool is_boundary = (k_start + BLOCK_N > seq_len);
        const bool full_tile   = (!IsCausal || (j < block_m_idx)) && !is_boundary && (q_start + BLOCK_M <= seq_len);

        if (full_tile) {
            #pragma unroll
            for (int n = 0; n < 4; ++n) {
                acc_s[n].x[0] *= sm_scale_log2;
                acc_s[n].x[1] *= sm_scale_log2;
                acc_s[n].x[2] *= sm_scale_log2;
                acc_s[n].x[3] *= sm_scale_log2;
                acc_s[n].x[4] *= sm_scale_log2;
                acc_s[n].x[5] *= sm_scale_log2;
                acc_s[n].x[6] *= sm_scale_log2;
                acc_s[n].x[7] *= sm_scale_log2;
            }
        } else {
            #pragma unroll
            for (int n = 0; n < 4; ++n) {
                const int col0 = n * 16 + c_base + 0;
                const int col1 = n * 16 + c_base + 1;
                const int col2 = n * 16 + c_base + 8;
                const int col3 = n * 16 + c_base + 9;

                const int g_c0 = k_start + col0;
                const int g_c1 = k_start + col1;
                const int g_c2 = k_start + col2;
                const int g_c3 = k_start + col3;

                bool v0_r0 = (g_c0 < seq_len) && (!is_diagonal || g_c0 <= global_q_r0) && (global_q_r0 < seq_len);
                bool v1_r0 = (g_c1 < seq_len) && (!is_diagonal || g_c1 <= global_q_r0) && (global_q_r0 < seq_len);
                bool v2_r0 = (g_c2 < seq_len) && (!is_diagonal || g_c2 <= global_q_r0) && (global_q_r0 < seq_len);
                bool v3_r0 = (g_c3 < seq_len) && (!is_diagonal || g_c3 <= global_q_r0) && (global_q_r0 < seq_len);

                acc_s[n].x[0] = v0_r0 ? (acc_s[n].x[0] * sm_scale_log2) : -1e30f;
                acc_s[n].x[1] = v1_r0 ? (acc_s[n].x[1] * sm_scale_log2) : -1e30f;
                acc_s[n].x[4] = v2_r0 ? (acc_s[n].x[4] * sm_scale_log2) : -1e30f;
                acc_s[n].x[5] = v3_r0 ? (acc_s[n].x[5] * sm_scale_log2) : -1e30f;

                bool v0_r1 = (g_c0 < seq_len) && (!is_diagonal || g_c0 <= global_q_r1) && (global_q_r1 < seq_len);
                bool v1_r1 = (g_c1 < seq_len) && (!is_diagonal || g_c1 <= global_q_r1) && (global_q_r1 < seq_len);
                bool v2_r1 = (g_c2 < seq_len) && (!is_diagonal || g_c2 <= global_q_r1) && (global_q_r1 < seq_len);
                bool v3_r1 = (g_c3 < seq_len) && (!is_diagonal || g_c3 <= global_q_r1) && (global_q_r1 < seq_len);

                acc_s[n].x[2] = v0_r1 ? (acc_s[n].x[2] * sm_scale_log2) : -1e30f;
                acc_s[n].x[3] = v1_r1 ? (acc_s[n].x[3] * sm_scale_log2) : -1e30f;
                acc_s[n].x[6] = v2_r1 ? (acc_s[n].x[6] * sm_scale_log2) : -1e30f;
                acc_s[n].x[7] = v3_r1 ? (acc_s[n].x[7] * sm_scale_log2) : -1e30f;
            }
        }

        // 3.4: In-Register Online Softmax Reduction
        float row_max0 = -1e30f;
        float row_max1 = -1e30f;
        #pragma unroll
        for (int n = 0; n < 4; ++n) {
            row_max0 = fmaxf(row_max0, fmaxf(acc_s[n].x[0], acc_s[n].x[1]));
            row_max0 = fmaxf(row_max0, fmaxf(acc_s[n].x[4], acc_s[n].x[5]));

            row_max1 = fmaxf(row_max1, fmaxf(acc_s[n].x[2], acc_s[n].x[3]));
            row_max1 = fmaxf(row_max1, fmaxf(acc_s[n].x[6], acc_s[n].x[7]));
        }

        row_max0 = fmaxf(row_max0, __shfl_xor_sync(0xffffffff, row_max0, 1));
        row_max0 = fmaxf(row_max0, __shfl_xor_sync(0xffffffff, row_max0, 2));

        row_max1 = fmaxf(row_max1, __shfl_xor_sync(0xffffffff, row_max1, 1));
        row_max1 = fmaxf(row_max1, __shfl_xor_sync(0xffffffff, row_max1, 2));

        float m_new0 = fmaxf(m_prev0, row_max0);
        float rescale0 = (m_prev0 > -1e20f) ? fast_exp2(m_prev0 - m_new0) : 0.0f;

        float m_new1 = fmaxf(m_prev1, row_max1);
        float rescale1 = (m_prev1 > -1e20f) ? fast_exp2(m_prev1 - m_new1) : 0.0f;

        // 3.5: Fast EX2 Hardware Exponentiation & Sum Accumulation
        float row_sum0 = 0.0f;
        float row_sum1 = 0.0f;
        fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> frag_p[4];

        if (full_tile) {
            #pragma unroll
            for (int n = 0; n < 4; ++n) {
                float p0 = fast_exp2(acc_s[n].x[0] - m_new0);
                float p1 = fast_exp2(acc_s[n].x[1] - m_new0);
                float p4 = fast_exp2(acc_s[n].x[4] - m_new0);
                float p5 = fast_exp2(acc_s[n].x[5] - m_new0);

                row_sum0 += (p0 + p1 + p4 + p5);

                float p2 = fast_exp2(acc_s[n].x[2] - m_new1);
                float p3 = fast_exp2(acc_s[n].x[3] - m_new1);
                float p6 = fast_exp2(acc_s[n].x[6] - m_new1);
                float p7 = fast_exp2(acc_s[n].x[7] - m_new1);

                row_sum1 += (p2 + p3 + p6 + p7);

                frag_p[n].x[0] = __float2half(p0);
                frag_p[n].x[1] = __float2half(p1);
                frag_p[n].x[2] = __float2half(p2);
                frag_p[n].x[3] = __float2half(p3);
                frag_p[n].x[4] = __float2half(p4);
                frag_p[n].x[5] = __float2half(p5);
                frag_p[n].x[6] = __float2half(p6);
                frag_p[n].x[7] = __float2half(p7);

                frag_p[n].x[8]  = frag_p[n].x[0];
                frag_p[n].x[9]  = frag_p[n].x[1];
                frag_p[n].x[10] = frag_p[n].x[2];
                frag_p[n].x[11] = frag_p[n].x[3];
                frag_p[n].x[12] = frag_p[n].x[4];
                frag_p[n].x[13] = frag_p[n].x[5];
                frag_p[n].x[14] = frag_p[n].x[6];
                frag_p[n].x[15] = frag_p[n].x[7];
            }
        } else {
            #pragma unroll
            for (int n = 0; n < 4; ++n) {
                float p0 = (acc_s[n].x[0] > -1e20f) ? fast_exp2(acc_s[n].x[0] - m_new0) : 0.0f;
                float p1 = (acc_s[n].x[1] > -1e20f) ? fast_exp2(acc_s[n].x[1] - m_new0) : 0.0f;
                float p4 = (acc_s[n].x[4] > -1e20f) ? fast_exp2(acc_s[n].x[4] - m_new0) : 0.0f;
                float p5 = (acc_s[n].x[5] > -1e20f) ? fast_exp2(acc_s[n].x[5] - m_new0) : 0.0f;

                row_sum0 += (p0 + p1 + p4 + p5);

                float p2 = (acc_s[n].x[2] > -1e20f) ? fast_exp2(acc_s[n].x[2] - m_new1) : 0.0f;
                float p3 = (acc_s[n].x[3] > -1e20f) ? fast_exp2(acc_s[n].x[3] - m_new1) : 0.0f;
                float p6 = (acc_s[n].x[6] > -1e20f) ? fast_exp2(acc_s[n].x[6] - m_new1) : 0.0f;
                float p7 = (acc_s[n].x[7] > -1e20f) ? fast_exp2(acc_s[n].x[7] - m_new1) : 0.0f;

                row_sum1 += (p2 + p3 + p6 + p7);

                frag_p[n].x[0] = __float2half(p0);
                frag_p[n].x[1] = __float2half(p1);
                frag_p[n].x[2] = __float2half(p2);
                frag_p[n].x[3] = __float2half(p3);
                frag_p[n].x[4] = __float2half(p4);
                frag_p[n].x[5] = __float2half(p5);
                frag_p[n].x[6] = __float2half(p6);
                frag_p[n].x[7] = __float2half(p7);

                frag_p[n].x[8]  = frag_p[n].x[0];
                frag_p[n].x[9]  = frag_p[n].x[1];
                frag_p[n].x[10] = frag_p[n].x[2];
                frag_p[n].x[11] = frag_p[n].x[3];
                frag_p[n].x[12] = frag_p[n].x[4];
                frag_p[n].x[13] = frag_p[n].x[5];
                frag_p[n].x[14] = frag_p[n].x[6];
                frag_p[n].x[15] = frag_p[n].x[7];
            }
        }

        row_sum0 += __shfl_xor_sync(0xffffffff, row_sum0, 1);
        row_sum0 += __shfl_xor_sync(0xffffffff, row_sum0, 2);

        row_sum1 += __shfl_xor_sync(0xffffffff, row_sum1, 1);
        row_sum1 += __shfl_xor_sync(0xffffffff, row_sum1, 2);

        l_prev0 = l_prev0 * rescale0 + row_sum0;
        l_prev1 = l_prev1 * rescale1 + row_sum1;
        m_prev0 = m_new0;
        m_prev1 = m_new1;

        #pragma unroll
        for (int d = 0; d < 4; ++d) {
            rescale_acc(acc_o[d], rescale0, rescale1);
        }

        // 3.7: GEMM 2: O = P * V
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            #pragma unroll
            for (int d = 0; d < 4; ++d) {
                fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, row_major> frag_v;
                load_matrix_sync(frag_v, &smem.s_V[(k * 16) * PADDED_HEAD_DIM + d * 16], PADDED_HEAD_DIM);
                mma_sync(acc_o[d], frag_p[k], frag_v, acc_o[d]);
            }
        }

        __syncthreads();
    } // End loop over Key/Value tiles

    // Step 4: Final Output Normalization & Direct In-Register Global Writeout
    float inv_l0 = (l_prev0 > 0.0f) ? (1.0f / l_prev0) : 0.0f;
    float inv_l1 = (l_prev1 > 0.0f) ? (1.0f / l_prev1) : 0.0f;

    #pragma unroll
    for (int d = 0; d < 4; ++d) {
        rescale_acc(acc_o[d], inv_l0, inv_l1);

        int col_base = d * 16 + c_base;

        if (global_q_r0 < seq_len) {
            half2 h0_01 = make_half2(__float2half(acc_o[d].x[0]), __float2half(acc_o[d].x[1]));
            half2 h0_89 = make_half2(__float2half(acc_o[d].x[4]), __float2half(acc_o[d].x[5]));
            *reinterpret_cast<half2*>(o_base_ptr + global_q_r0 * head_dim + col_base + 0) = h0_01;
            *reinterpret_cast<half2*>(o_base_ptr + global_q_r0 * head_dim + col_base + 8) = h0_89;
        }

        if (global_q_r1 < seq_len) {
            half2 h1_01 = make_half2(__float2half(acc_o[d].x[2]), __float2half(acc_o[d].x[3]));
            half2 h1_89 = make_half2(__float2half(acc_o[d].x[6]), __float2half(acc_o[d].x[7]));
            *reinterpret_cast<half2*>(o_base_ptr + global_q_r1 * head_dim + col_base + 0) = h1_01;
            *reinterpret_cast<half2*>(o_base_ptr + global_q_r1 * head_dim + col_base + 8) = h1_89;
        }
    }
}

// ============================================================================
// Kernel 2: FlashAttention-2 WMMA Br=128 Bc=64 (Optimized for N >= 2048)
// 256 threads (8 warps), 36.8 KB SRAM, cuts global memory traffic in half
// ============================================================================
template <bool IsCausal = false>
__global__ void __launch_bounds__(256, 2) flash_attn2_wmma_br128_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float sm_scale_log2
) {
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE; // 0..7 (8 warps)
    const int lane_id = tid % WARP_SIZE; // 0..31

    const int block_m_idx = blockIdx.x;  // Query tile index (Br=128 chunk)
    const int head_idx    = blockIdx.y;
    const int batch_idx   = blockIdx.z;

    if (batch_idx >= batch_size || head_idx >= num_heads) return;

    // Shared Memory: Br=128, Bc=64 (36.8 KB total)
    constexpr int Br = 128;
    constexpr int Bc = 64;
    struct SmemStorage {
        half s_Q[Br * PADDED_HEAD_DIM]; // 18.4 KB
        half s_K[Bc * PADDED_HEAD_DIM]; // 9.2 KB
        half s_V[Bc * PADDED_HEAD_DIM]; // 9.2 KB
    };
    __shared__ SmemStorage smem;

    const int head_stride = seq_len * head_dim;
    const int batch_head_offset = (batch_idx * num_heads + head_idx) * head_stride;

    const half* q_base_ptr = Q + batch_head_offset;
    const half* k_base_ptr = K + batch_head_offset;
    const half* v_base_ptr = V + batch_head_offset;
    half*       o_base_ptr = O + batch_head_offset;

    // Load Q (128 x 64 = 8192 halfs = 1024 uint4s; 256 threads => 4 steps)
    const int q_start = block_m_idx * Br;
    #pragma unroll
    for (int step = 0; step < 4; ++step) {
        int vec_idx = step * 256 + tid;
        int r = vec_idx >> 3;
        int d = (vec_idx & 7) << 3;
        int global_row = q_start + r;

        uint4 loaded = make_uint4(0, 0, 0, 0);
        if (global_row < seq_len) {
            loaded = *reinterpret_cast<const uint4*>(q_base_ptr + global_row * head_dim + d);
        }
        *reinterpret_cast<uint4*>(smem.s_Q + r * PADDED_HEAD_DIM + d) = loaded;
    }
    __syncthreads();

    // Hoist Query fragments into registers ONCE per warp
    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> frag_q[4];
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        load_matrix_sync(frag_q[k], &smem.s_Q[(warp_id * 16) * PADDED_HEAD_DIM + k * 16], PADDED_HEAD_DIM);
    }

    // Output accumulator
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_o[4];
    #pragma unroll
    for (int d = 0; d < 4; ++d) {
        fill_fragment(acc_o[d], 0.0f);
    }

    const int r0 = lane_id / 4;
    const int r1 = r0 + 8;
    const int c_base = (lane_id % 4) * 2;

    const int global_q_r0 = q_start + warp_id * 16 + r0;
    const int global_q_r1 = q_start + warp_id * 16 + r1;

    float m_prev0 = -1e30f;
    float m_prev1 = -1e30f;
    float l_prev0 = 0.0f;
    float l_prev1 = 0.0f;

    const int num_tiles_c = (seq_len + Bc - 1) / Bc;
    const int max_tiles_c = IsCausal
        ? min(num_tiles_c, (q_start + Br - 1) / Bc + 1)
        : num_tiles_c;

    // Loop over K, V
    for (int j = 0; j < max_tiles_c; ++j) {
        const int k_start = j * Bc;

        // Load K and V (64 x 64 = 4096 halfs = 512 uint4s; 256 threads => 2 steps)
        #pragma unroll
        for (int step = 0; step < 2; ++step) {
            int vec_idx = step * 256 + tid;
            int r = vec_idx >> 3;
            int d = (vec_idx & 7) << 3;
            int global_row = k_start + r;

            uint4 k_val = make_uint4(0, 0, 0, 0);
            uint4 v_val = make_uint4(0, 0, 0, 0);
            if (global_row < seq_len) {
                k_val = *reinterpret_cast<const uint4*>(k_base_ptr + global_row * head_dim + d);
                v_val = *reinterpret_cast<const uint4*>(v_base_ptr + global_row * head_dim + d);
            }
            *reinterpret_cast<uint4*>(smem.s_K + r * PADDED_HEAD_DIM + d) = k_val;
            *reinterpret_cast<uint4*>(smem.s_V + r * PADDED_HEAD_DIM + d) = v_val;
        }
        __syncthreads();

        // GEMM 1: S = Q * K^T
        fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_s[4];
        #pragma unroll
        for (int n = 0; n < 4; ++n) {
            fill_fragment(acc_s[n], 0.0f);
        }

        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            #pragma unroll
            for (int n = 0; n < 4; ++n) {
                fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, col_major> frag_k;
                load_matrix_sync(frag_k, &smem.s_K[(n * 16) * PADDED_HEAD_DIM + k * 16], PADDED_HEAD_DIM);
                mma_sync(acc_s[n], frag_q[k], frag_k, acc_s[n]);
            }
        }

        // Softmax scaling & Masking (Partitioned: Full Tile vs Diagonal/Boundary Tile)
        const bool is_causal_diag = IsCausal && (k_start + Bc > q_start);
        const bool is_boundary    = (k_start + Bc > seq_len);
        const bool full_tile      = (!IsCausal || (k_start + Bc <= q_start)) && !is_boundary && (q_start + Br <= seq_len);

        if (full_tile) {
            #pragma unroll
            for (int n = 0; n < 4; ++n) {
                acc_s[n].x[0] *= sm_scale_log2;
                acc_s[n].x[1] *= sm_scale_log2;
                acc_s[n].x[2] *= sm_scale_log2;
                acc_s[n].x[3] *= sm_scale_log2;
                acc_s[n].x[4] *= sm_scale_log2;
                acc_s[n].x[5] *= sm_scale_log2;
                acc_s[n].x[6] *= sm_scale_log2;
                acc_s[n].x[7] *= sm_scale_log2;
            }
        } else {
            #pragma unroll
            for (int n = 0; n < 4; ++n) {
                const int col0 = n * 16 + c_base + 0;
                const int col1 = n * 16 + c_base + 1;
                const int col2 = n * 16 + c_base + 8;
                const int col3 = n * 16 + c_base + 9;

                const int g_c0 = k_start + col0;
                const int g_c1 = k_start + col1;
                const int g_c2 = k_start + col2;
                const int g_c3 = k_start + col3;

                bool v0_r0 = (g_c0 < seq_len) && (!is_causal_diag || g_c0 <= global_q_r0) && (global_q_r0 < seq_len);
                bool v1_r0 = (g_c1 < seq_len) && (!is_causal_diag || g_c1 <= global_q_r0) && (global_q_r0 < seq_len);
                bool v2_r0 = (g_c2 < seq_len) && (!is_causal_diag || g_c2 <= global_q_r0) && (global_q_r0 < seq_len);
                bool v3_r0 = (g_c3 < seq_len) && (!is_causal_diag || g_c3 <= global_q_r0) && (global_q_r0 < seq_len);

                acc_s[n].x[0] = v0_r0 ? (acc_s[n].x[0] * sm_scale_log2) : -1e30f;
                acc_s[n].x[1] = v1_r0 ? (acc_s[n].x[1] * sm_scale_log2) : -1e30f;
                acc_s[n].x[4] = v2_r0 ? (acc_s[n].x[4] * sm_scale_log2) : -1e30f;
                acc_s[n].x[5] = v3_r0 ? (acc_s[n].x[5] * sm_scale_log2) : -1e30f;

                bool v0_r1 = (g_c0 < seq_len) && (!is_causal_diag || g_c0 <= global_q_r1) && (global_q_r1 < seq_len);
                bool v1_r1 = (g_c1 < seq_len) && (!is_causal_diag || g_c1 <= global_q_r1) && (global_q_r1 < seq_len);
                bool v2_r1 = (g_c2 < seq_len) && (!is_causal_diag || g_c2 <= global_q_r1) && (global_q_r1 < seq_len);
                bool v3_r1 = (g_c3 < seq_len) && (!is_causal_diag || g_c3 <= global_q_r1) && (global_q_r1 < seq_len);

                acc_s[n].x[2] = v0_r1 ? (acc_s[n].x[2] * sm_scale_log2) : -1e30f;
                acc_s[n].x[3] = v1_r1 ? (acc_s[n].x[3] * sm_scale_log2) : -1e30f;
                acc_s[n].x[6] = v2_r1 ? (acc_s[n].x[6] * sm_scale_log2) : -1e30f;
                acc_s[n].x[7] = v3_r1 ? (acc_s[n].x[7] * sm_scale_log2) : -1e30f;
            }
        }

        // Row max
        float row_max0 = -1e30f;
        float row_max1 = -1e30f;
        #pragma unroll
        for (int n = 0; n < 4; ++n) {
            row_max0 = fmaxf(row_max0, fmaxf(acc_s[n].x[0], acc_s[n].x[1]));
            row_max0 = fmaxf(row_max0, fmaxf(acc_s[n].x[4], acc_s[n].x[5]));

            row_max1 = fmaxf(row_max1, fmaxf(acc_s[n].x[2], acc_s[n].x[3]));
            row_max1 = fmaxf(row_max1, fmaxf(acc_s[n].x[6], acc_s[n].x[7]));
        }

        row_max0 = fmaxf(row_max0, __shfl_xor_sync(0xffffffff, row_max0, 1));
        row_max0 = fmaxf(row_max0, __shfl_xor_sync(0xffffffff, row_max0, 2));

        row_max1 = fmaxf(row_max1, __shfl_xor_sync(0xffffffff, row_max1, 1));
        row_max1 = fmaxf(row_max1, __shfl_xor_sync(0xffffffff, row_max1, 2));

        float m_new0 = fmaxf(m_prev0, row_max0);
        float rescale0 = (m_prev0 > -1e20f) ? fast_exp2(m_prev0 - m_new0) : 0.0f;

        float m_new1 = fmaxf(m_prev1, row_max1);
        float rescale1 = (m_prev1 > -1e20f) ? fast_exp2(m_prev1 - m_new1) : 0.0f;

        float row_sum0 = 0.0f;
        float row_sum1 = 0.0f;
        fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> frag_p[4];

        if (full_tile) {
            #pragma unroll
            for (int n = 0; n < 4; ++n) {
                float p0 = fast_exp2(acc_s[n].x[0] - m_new0);
                float p1 = fast_exp2(acc_s[n].x[1] - m_new0);
                float p4 = fast_exp2(acc_s[n].x[4] - m_new0);
                float p5 = fast_exp2(acc_s[n].x[5] - m_new0);

                row_sum0 += (p0 + p1 + p4 + p5);

                float p2 = fast_exp2(acc_s[n].x[2] - m_new1);
                float p3 = fast_exp2(acc_s[n].x[3] - m_new1);
                float p6 = fast_exp2(acc_s[n].x[6] - m_new1);
                float p7 = fast_exp2(acc_s[n].x[7] - m_new1);

                row_sum1 += (p2 + p3 + p6 + p7);

                frag_p[n].x[0] = __float2half(p0);
                frag_p[n].x[1] = __float2half(p1);
                frag_p[n].x[2] = __float2half(p2);
                frag_p[n].x[3] = __float2half(p3);
                frag_p[n].x[4] = __float2half(p4);
                frag_p[n].x[5] = __float2half(p5);
                frag_p[n].x[6] = __float2half(p6);
                frag_p[n].x[7] = __float2half(p7);

                frag_p[n].x[8]  = frag_p[n].x[0];
                frag_p[n].x[9]  = frag_p[n].x[1];
                frag_p[n].x[10] = frag_p[n].x[2];
                frag_p[n].x[11] = frag_p[n].x[3];
                frag_p[n].x[12] = frag_p[n].x[4];
                frag_p[n].x[13] = frag_p[n].x[5];
                frag_p[n].x[14] = frag_p[n].x[6];
                frag_p[n].x[15] = frag_p[n].x[7];
            }
        } else {
            #pragma unroll
            for (int n = 0; n < 4; ++n) {
                float p0 = (acc_s[n].x[0] > -1e20f) ? fast_exp2(acc_s[n].x[0] - m_new0) : 0.0f;
                float p1 = (acc_s[n].x[1] > -1e20f) ? fast_exp2(acc_s[n].x[1] - m_new0) : 0.0f;
                float p4 = (acc_s[n].x[4] > -1e20f) ? fast_exp2(acc_s[n].x[4] - m_new0) : 0.0f;
                float p5 = (acc_s[n].x[5] > -1e20f) ? fast_exp2(acc_s[n].x[5] - m_new0) : 0.0f;

                row_sum0 += (p0 + p1 + p4 + p5);

                float p2 = (acc_s[n].x[2] > -1e20f) ? fast_exp2(acc_s[n].x[2] - m_new1) : 0.0f;
                float p3 = (acc_s[n].x[3] > -1e20f) ? fast_exp2(acc_s[n].x[3] - m_new1) : 0.0f;
                float p6 = (acc_s[n].x[6] > -1e20f) ? fast_exp2(acc_s[n].x[6] - m_new1) : 0.0f;
                float p7 = (acc_s[n].x[7] > -1e20f) ? fast_exp2(acc_s[n].x[7] - m_new1) : 0.0f;

                row_sum1 += (p2 + p3 + p6 + p7);

                frag_p[n].x[0] = __float2half(p0);
                frag_p[n].x[1] = __float2half(p1);
                frag_p[n].x[2] = __float2half(p2);
                frag_p[n].x[3] = __float2half(p3);
                frag_p[n].x[4] = __float2half(p4);
                frag_p[n].x[5] = __float2half(p5);
                frag_p[n].x[6] = __float2half(p6);
                frag_p[n].x[7] = __float2half(p7);

                frag_p[n].x[8]  = frag_p[n].x[0];
                frag_p[n].x[9]  = frag_p[n].x[1];
                frag_p[n].x[10] = frag_p[n].x[2];
                frag_p[n].x[11] = frag_p[n].x[3];
                frag_p[n].x[12] = frag_p[n].x[4];
                frag_p[n].x[13] = frag_p[n].x[5];
                frag_p[n].x[14] = frag_p[n].x[6];
                frag_p[n].x[15] = frag_p[n].x[7];
            }
        }

        row_sum0 += __shfl_xor_sync(0xffffffff, row_sum0, 1);
        row_sum0 += __shfl_xor_sync(0xffffffff, row_sum0, 2);

        row_sum1 += __shfl_xor_sync(0xffffffff, row_sum1, 1);
        row_sum1 += __shfl_xor_sync(0xffffffff, row_sum1, 2);

        l_prev0 = l_prev0 * rescale0 + row_sum0;
        l_prev1 = l_prev1 * rescale1 + row_sum1;
        m_prev0 = m_new0;
        m_prev1 = m_new1;

        #pragma unroll
        for (int d = 0; d < 4; ++d) {
            rescale_acc(acc_o[d], rescale0, rescale1);
        }

        // GEMM 2: O = P * V
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            #pragma unroll
            for (int d = 0; d < 4; ++d) {
                fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, row_major> frag_v;
                load_matrix_sync(frag_v, &smem.s_V[(k * 16) * PADDED_HEAD_DIM + d * 16], PADDED_HEAD_DIM);
                mma_sync(acc_o[d], frag_p[k], frag_v, acc_o[d]);
            }
        }

        __syncthreads();
    }

    // Step 4: Direct In-Register Global Writeout
    float inv_l0 = (l_prev0 > 0.0f) ? (1.0f / l_prev0) : 0.0f;
    float inv_l1 = (l_prev1 > 0.0f) ? (1.0f / l_prev1) : 0.0f;

    #pragma unroll
    for (int d = 0; d < 4; ++d) {
        rescale_acc(acc_o[d], inv_l0, inv_l1);

        int col_base = d * 16 + c_base;

        if (global_q_r0 < seq_len) {
            half2 h0_01 = make_half2(__float2half(acc_o[d].x[0]), __float2half(acc_o[d].x[1]));
            half2 h0_89 = make_half2(__float2half(acc_o[d].x[4]), __float2half(acc_o[d].x[5]));
            *reinterpret_cast<half2*>(o_base_ptr + global_q_r0 * head_dim + col_base + 0) = h0_01;
            *reinterpret_cast<half2*>(o_base_ptr + global_q_r0 * head_dim + col_base + 8) = h0_89;
        }

        if (global_q_r1 < seq_len) {
            half2 h1_01 = make_half2(__float2half(acc_o[d].x[2]), __float2half(acc_o[d].x[3]));
            half2 h1_89 = make_half2(__float2half(acc_o[d].x[6]), __float2half(acc_o[d].x[7]));
            *reinterpret_cast<half2*>(o_base_ptr + global_q_r1 * head_dim + col_base + 0) = h1_01;
            *reinterpret_cast<half2*>(o_base_ptr + global_q_r1 * head_dim + col_base + 8) = h1_89;
        }
    }
}

// ============================================================================
// Host Entrypoint & Adaptive Dispatcher
// ============================================================================
torch::Tensor flash_attn_forward_flash2(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    double sm_scale,
    bool is_causal,
    std::optional<torch::Tensor> out
) {
    // 1. Tensor Input Validations
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "All tensors must be on CUDA");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(), "All tensors must be contiguous");
    TORCH_CHECK(q.dtype() == torch::kHalf && k.dtype() == torch::kHalf && v.dtype() == torch::kHalf,
                "All tensors must be FP16 (Half)");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "Tensors must be 4D [B, H, N, D]");

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

    torch::Tensor out_tensor;
    if (out.has_value() && out.value().defined()) {
        out_tensor = out.value();
        TORCH_CHECK(out_tensor.is_cuda() && out_tensor.is_contiguous(), "out tensor must be contiguous CUDA tensor");
        TORCH_CHECK(out_tensor.sizes() == q.sizes(), "out tensor shape must match q");
        TORCH_CHECK(out_tensor.dtype() == torch::kHalf, "out tensor dtype must be FP16");
    } else {
        out_tensor = torch::empty_like(q);
    }

    if (sm_scale == 0.0) {
        sm_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    }
    const float sm_scale_log2 = static_cast<float>(sm_scale * LOG2_E);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    // Configure 100 KB shared memory carveout for maximum occupancy
    static bool carveout_configured = false;
    if (!carveout_configured) {
        cudaFuncSetAttribute(flash_attn2_wmma_br64_kernel<false>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
        cudaFuncSetAttribute(flash_attn2_wmma_br64_kernel<true>,  cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
        cudaFuncSetAttribute(flash_attn2_wmma_br128_kernel<false>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
        cudaFuncSetAttribute(flash_attn2_wmma_br128_kernel<true>,  cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
        carveout_configured = true;
    }

    // Adaptive Dispatch:
    // For N >= 2048: Use Br=128 (256 threads) to cut global memory traffic in half (17+ TFLOPS)
    // For N < 2048:  Use Br=64 (128 threads) to maximize wave granularity across 20 SMs (13+ TFLOPS)
    if (seq_len >= 2048) {
        constexpr int Br = 128;
        const int grid_m = (seq_len + Br - 1) / Br;
        dim3 grid(grid_m, num_heads, batch_size);
        dim3 block(256);

        if (is_causal) {
            flash_attn2_wmma_br128_kernel<true><<<grid, block, 0, stream>>>(
                reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
                reinterpret_cast<const half*>(k.data_ptr<at::Half>()),
                reinterpret_cast<const half*>(v.data_ptr<at::Half>()),
                reinterpret_cast<half*>(out_tensor.data_ptr<at::Half>()),
                batch_size, num_heads, seq_len, head_dim, sm_scale_log2
            );
        } else {
            flash_attn2_wmma_br128_kernel<false><<<grid, block, 0, stream>>>(
                reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
                reinterpret_cast<const half*>(k.data_ptr<at::Half>()),
                reinterpret_cast<const half*>(v.data_ptr<at::Half>()),
                reinterpret_cast<half*>(out_tensor.data_ptr<at::Half>()),
                batch_size, num_heads, seq_len, head_dim, sm_scale_log2
            );
        }
    } else {
        constexpr int Br = 64;
        const int grid_m = (seq_len + Br - 1) / Br;
        dim3 grid(grid_m, num_heads, batch_size);
        dim3 block(128);

        if (is_causal) {
            flash_attn2_wmma_br64_kernel<true><<<grid, block, 0, stream>>>(
                reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
                reinterpret_cast<const half*>(k.data_ptr<at::Half>()),
                reinterpret_cast<const half*>(v.data_ptr<at::Half>()),
                reinterpret_cast<half*>(out_tensor.data_ptr<at::Half>()),
                batch_size, num_heads, seq_len, head_dim, sm_scale_log2
            );
        } else {
            flash_attn2_wmma_br64_kernel<false><<<grid, block, 0, stream>>>(
                reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
                reinterpret_cast<const half*>(k.data_ptr<at::Half>()),
                reinterpret_cast<const half*>(v.data_ptr<at::Half>()),
                reinterpret_cast<half*>(out_tensor.data_ptr<at::Half>()),
                batch_size, num_heads, seq_len, head_dim, sm_scale_log2
            );
        }
    }

    CUDA_CHECK(cudaGetLastError());
    return out_tensor;
}

} // namespace flash_attn
