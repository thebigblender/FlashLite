#pragma once

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>

#ifdef __CUDACC__
#include <mma.h>
#endif

// Error checking helper macro for CUDA runtime calls
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            TORCH_CHECK(false, "CUDA Error at ", __FILE__, ":", __LINE__,     \
                        " - ", cudaGetErrorString(err));                      \
        }                                                                     \
    } while (0)

namespace flash_attn {

// ============================================================================
// Tile and Hardware Configuration Constants (Targeting Ampere sm_86)
// ============================================================================
// Ampere warp-level matrix multiply and accumulate (WMMA) standard tile sizes:
// M=16, N=16, K=16 for __half precision
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

constexpr int WARP_SIZE = 32;
constexpr int NUM_WARPS = 4;                           // 4 warps per CTA (Threadblock)
constexpr int BLOCK_THREADS = NUM_WARPS * WARP_SIZE;  // 128 threads per threadblock

// FlashAttention-2 Block Tiling Defaults
constexpr int BLOCK_M = 64;   // Query sequence tile size (Br)
constexpr int BLOCK_N = 64;   // Key/Value sequence tile size (Bc)
constexpr int DEFAULT_HEAD_DIM = 64; // Standard head dimension

#ifdef __CUDACC__
// ============================================================================
// WMMA Typedefs & Fragment Abstractions
// ============================================================================
namespace wmma_stubs {
    using namespace nvcuda::wmma;

    // WMMA fragments for FP16 Matrix Multiply: C = A * B + C
    // Fragment A: [16, 16] tile in row-major layout
    using FragA = fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major>;
    
    // Fragment B: [16, 16] tile in col-major layout (useful for K^T)
    using FragBCol = fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, col_major>;
    
    // Fragment B: [16, 16] tile in row-major layout (useful for V)
    using FragBRow = fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, row_major>;

    // Accumulator fragment: [16, 16] tile with FP32 precision for numerical stability
    using FragAccF32 = fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float>;

    // Accumulator fragment: [16, 16] tile with FP16 precision
    using FragAccF16 = fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, half>;
}

// ============================================================================
// Shared Memory Tile Storage Structs (Stubs)
// ============================================================================
template <int Br = BLOCK_M, int Bc = BLOCK_N, int HeadDim = DEFAULT_HEAD_DIM>
struct Flash2SharedStorage {
    // Shared memory buffers for Q, K, V tiles
    // Dynamic or static allocation depending on launch configuration
    half s_Q[Br * HeadDim];
    half s_K[Bc * HeadDim];
    half s_V[Bc * HeadDim];

    // Shared memory buffer for intermediate score tiles / warp communication
    float s_S[Br * Bc];
};
#endif


// ============================================================================
// Forward Function Declarations
// ============================================================================

/**
 * @brief Forward pass for Naive un-fused Attention (Baseline comparison)
 * @param q Query tensor [Batch, Heads, SeqLen, HeadDim] (FP16, CUDA)
 * @param k Key tensor   [Batch, Heads, SeqLen, HeadDim] (FP16, CUDA)
 * @param v Value tensor [Batch, Heads, SeqLen, HeadDim] (FP16, CUDA)
 * @param sm_scale Softmax scaling factor (typically 1.0 / sqrt(HeadDim))
 * @param is_causal Whether to apply causal masking
 * @return Output tensor [Batch, Heads, SeqLen, HeadDim] (FP16, CUDA)
 */
torch::Tensor flash_attn_forward_naive(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    double sm_scale,
    bool is_causal
);

/**
 * @brief Forward pass for Custom FlashAttention-2 using WMMA
 * @param q Query tensor [Batch, Heads, SeqLen, HeadDim] (FP16, CUDA)
 * @param k Key tensor   [Batch, Heads, SeqLen, HeadDim] (FP16, CUDA)
 * @param v Value tensor [Batch, Heads, SeqLen, HeadDim] (FP16, CUDA)
 * @param sm_scale Softmax scaling factor (typically 1.0 / sqrt(HeadDim))
 * @param is_causal Whether to apply causal masking
 * @return Output tensor [Batch, Heads, SeqLen, HeadDim] (FP16, CUDA)
 */
torch::Tensor flash_attn_forward_flash2(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    double sm_scale,
    bool is_causal
);

} // namespace flash_attn
