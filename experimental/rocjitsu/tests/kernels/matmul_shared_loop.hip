// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>

static __device__ __forceinline__ void matmul_shared_loop_barrier() {
  asm volatile("s_waitcnt lgkmcnt(0)\n\t"
               "s_barrier"
               :
               :
               : "memory");
}

/// Shared-memory matmul loop fixture without MFMA instructions.
///
/// One 256-thread workgroup computes a 16x16 output tile for:
///   C[16x16] = A[16x32] * B[32x16]
///
/// Each outer-loop iteration stages one 16-wide K tile of A and B through LDS.
/// The inner loop performs ordinary scalar floating-point multiply-adds so the
/// fixture exercises loop, LDS, and barrier translation without tensor/MFMA
/// lowering.
extern "C" __global__ void matmul_shared_loop(const float *__restrict__ A,
                                               const float *__restrict__ B,
                                               float *__restrict__ C) {
  constexpr unsigned M = 16;
  constexpr unsigned N = 16;
  constexpr unsigned K = 32;
  constexpr unsigned TILE_K = 16;

  __shared__ float tile_a[M][TILE_K];
  __shared__ float tile_b[TILE_K][N];

  const unsigned lane = threadIdx.x;
  const unsigned row = lane / N;
  const unsigned col = lane & (N - 1);

  float acc = 0.0f;
  for (unsigned k_base = 0; k_base < K; k_base += TILE_K) {
    const unsigned tile_row = lane / TILE_K;
    const unsigned tile_col = lane & (TILE_K - 1);

    tile_a[tile_row][tile_col] = A[tile_row * K + k_base + tile_col];
    tile_b[tile_row][tile_col] = B[(k_base + tile_row) * N + tile_col];
    matmul_shared_loop_barrier();

    for (unsigned kk = 0; kk < TILE_K; ++kk)
      acc += tile_a[row][kk] * tile_b[kk][col];

    matmul_shared_loop_barrier();
  }

  C[row * N + col] = acc;
}
