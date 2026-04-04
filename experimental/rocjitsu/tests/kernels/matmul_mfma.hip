// Copyright (c) 2026 Advanced Micro Devices, Inc.
// All rights reserved.

#include <hip/hip_runtime.h>

/// Tiled matrix multiplication kernel using Matrix Fused Multiply-Accumulate
/// (MFMA) instructions for multi-workgroup stress testing.
///
/// Uses v_mfma_f32_4x4x1f32 to compute 4x4 tiles of C = A * B.  Each
/// workgroup computes one 4x4 tile, identified by blockIdx.x:
///   tile_row = blockIdx.x / (N/4)
///   tile_col = blockIdx.x % (N/4)
///
/// Within a workgroup, lanes 0-3 cooperate on the tile.  Each lane holds one
/// row of the 4x4 accumulator.  The K-loop iterates over the shared dimension,
/// feeding one column of A and one row of B per iteration via rank-1 updates.
///
/// With N=64 and 256 workgroups of 64 threads each, all 256 CUs across 8 XCDs
/// are utilized (16x16 tiles of 4x4 = 64x64 output elements).
///
/// Kernel arguments (passed via the argument buffer):
///   A - pointer to input matrix A  (N*N floats, row-major)
///   B - pointer to input matrix B  (N*N floats, row-major)
///   C - pointer to output matrix C (N*N floats, row-major)
///   N - matrix dimension (must be a multiple of 4)

using v4f32 = __attribute__((ext_vector_type(4))) float;

extern "C" __global__ void matmul_mfma(const float *__restrict__ A,
                                       const float *__restrict__ B,
                                       float *__restrict__ C, unsigned N) {
  constexpr unsigned TILE = 4;
  unsigned tiles_per_row = N / TILE;
  unsigned tile_row = blockIdx.x / tiles_per_row;
  unsigned tile_col = blockIdx.x % tiles_per_row;
  unsigned lane = threadIdx.x % TILE;

  unsigned row_base = tile_row * TILE;
  unsigned col_base = tile_col * TILE;

  // Initialize the accumulator to zero.  Each lane accumulates one row of the tile.
  v4f32 acc = {0.0f, 0.0f, 0.0f, 0.0f};

  // K-loop: each iteration provides one column of A and one row of B.
  for (unsigned k = 0; k < N; ++k) {
    float a = A[(row_base + lane) * N + k];
    float b = B[k * N + (col_base + lane)];
    acc = __builtin_amdgcn_mfma_f32_4x4x1f32(a, b, acc, 0, 0, 0);
  }

  // Write the accumulated row back to C.
  for (unsigned j = 0; j < TILE; ++j)
    C[(row_base + lane) * N + col_base + j] = acc[j];
}
