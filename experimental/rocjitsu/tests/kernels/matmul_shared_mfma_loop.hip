// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

using v4f32 = __attribute__((ext_vector_type(4))) float;
using v4f16 = __attribute__((ext_vector_type(4))) _Float16;

static __device__ __forceinline__ void matmul_shared_mfma_loop_barrier() {
  asm volatile("s_waitcnt lgkmcnt(0)\n\t"
               "s_barrier"
               :
               :
               : "memory");
}

/// Shared-memory matmul loop fixture using MFMA instructions.
///
/// One 64-lane workgroup computes:
///   C[16x16] = A[16x32] * B[32x16]
///
/// Each outer-loop iteration stages a 16-wide K tile of A and B through LDS,
/// then feeds the LDS-loaded fragments to v_mfma_f32_16x16x16_f16. This keeps
/// the same MFMA lane mapping as matmul_mfma_16x16 while adding real loop and
/// shared-memory traffic around the tensor operation.
extern "C" __global__ void matmul_shared_mfma_loop(const _Float16 *__restrict__ A,
                                                    const _Float16 *__restrict__ B,
                                                    float *__restrict__ C) {
  constexpr unsigned M = 16;
  constexpr unsigned N = 16;
  constexpr unsigned K = 32;
  constexpr unsigned TILE_K = 16;

  __shared__ _Float16 tile_a[M][TILE_K];
  __shared__ _Float16 tile_b[TILE_K][N];

  const unsigned lane = threadIdx.x;
  v4f32 acc = {0.0f, 0.0f, 0.0f, 0.0f};

  for (unsigned k_base = 0; k_base < K; k_base += TILE_K) {
    for (unsigned i = 0; i < 4; ++i) {
      const unsigned element = lane * 4 + i;
      const unsigned tile_row = element / TILE_K;
      const unsigned tile_col = element & (TILE_K - 1);

      tile_a[tile_row][tile_col] = A[tile_row * K + k_base + tile_col];
      tile_b[tile_row][tile_col] = B[(k_base + tile_row) * N + tile_col];
    }
    matmul_shared_mfma_loop_barrier();

    const unsigned a_row = lane % M;
    const unsigned a_k = (lane / M) * 4;
    v4f16 src_a = {
        tile_a[a_row][a_k + 0],
        tile_a[a_row][a_k + 1],
        tile_a[a_row][a_k + 2],
        tile_a[a_row][a_k + 3],
    };

    const unsigned b_col = lane % N;
    const unsigned b_k = (lane / N) * 4;
    v4f16 src_b = {
        tile_b[b_k + 0][b_col],
        tile_b[b_k + 1][b_col],
        tile_b[b_k + 2][b_col],
        tile_b[b_k + 3][b_col],
    };

    acc = __builtin_amdgcn_mfma_f32_16x16x16f16(src_a, src_b, acc, 0, 0, 0);
    matmul_shared_mfma_loop_barrier();
  }

  const unsigned col = lane % N;
  const unsigned row_group = lane / N;
  for (unsigned r = 0; r < 4; ++r)
    C[(row_group * 4 + r) * N + col] = acc[r];
}
