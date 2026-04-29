// Copyright (c) 2026 Advanced Micro Devices, Inc.
// All rights reserved.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

/// 16x16 MFMA matrix multiply kernel for DBT translation testing.
///
/// Computes C = A * B using v_mfma_f32_16x16x16_f16 where:
///   A is 16x16 FP16 (row-major)
///   B is 16x16 FP16 (row-major)
///   C is 16x16 FP32 (row-major)
///
/// One workgroup of 64 threads computes the entire 16x16 output.
/// The MFMA instruction distributes the 16x16x16 multiply-accumulate
/// across registers and lanes per the GFX9 MFMA convention:
///   Output: reg = row % 4, lane = (row/4)*16 + col
///   Input:  packed f16 pairs loaded cooperatively by all 64 lanes

using v4f32 = __attribute__((ext_vector_type(4))) float;
using v4f16 = __attribute__((ext_vector_type(4))) _Float16;

extern "C" __global__ void matmul_mfma_16x16(const _Float16 *__restrict__ A,
                                              const _Float16 *__restrict__ B,
                                              float *__restrict__ C) {
  constexpr unsigned M = 16;
  constexpr unsigned N = 16;
  constexpr unsigned K = 16;
  unsigned lane = threadIdx.x;

  // Load A matrix into src0 (v4f16 = 64-bit = 4 packed f16).
  // Lane mapping for 16x16x16 MFMA src0:
  //   lane%16 → matrix row, (lane/16)*4 → K offset base
  //   Each lane holds 4 consecutive f16 from A[row][k_base..k_base+3]
  unsigned a_row = lane % M;
  unsigned a_k = (lane / M) * 4;
  v4f16 src_a = {
      (a_k + 0 < K) ? A[a_row * K + a_k + 0] : (_Float16)0,
      (a_k + 1 < K) ? A[a_row * K + a_k + 1] : (_Float16)0,
      (a_k + 2 < K) ? A[a_row * K + a_k + 2] : (_Float16)0,
      (a_k + 3 < K) ? A[a_row * K + a_k + 3] : (_Float16)0,
  };

  // Load B matrix into src1 (v4f16 = 64-bit = 4 packed f16).
  unsigned b_col = lane % N;
  unsigned b_k = (lane / N) * 4;
  v4f16 src_b = {
      (b_k + 0 < K) ? B[(b_k + 0) * N + b_col] : (_Float16)0,
      (b_k + 1 < K) ? B[(b_k + 1) * N + b_col] : (_Float16)0,
      (b_k + 2 < K) ? B[(b_k + 2) * N + b_col] : (_Float16)0,
      (b_k + 3 < K) ? B[(b_k + 3) * N + b_col] : (_Float16)0,
  };

  // Accumulator initialized to zero (4 x f32 per lane = 128-bit)
  v4f32 acc = {0.0f, 0.0f, 0.0f, 0.0f};

  // MFMA: C += A * B
  acc = __builtin_amdgcn_mfma_f32_16x16x16f16(src_a, src_b, acc, 0, 0, 0);

  // Store: MFMA 16x16 output layout:
  //   For each lane: col = lane % 16, row_group = lane / 16
  //   acc[r] holds C[row_group*4 + r][col]
  unsigned col = lane % N;
  unsigned row_group = lane / N;
  for (unsigned r = 0; r < 4; ++r)
    C[(row_group * 4 + r) * N + col] = acc[r];
}
