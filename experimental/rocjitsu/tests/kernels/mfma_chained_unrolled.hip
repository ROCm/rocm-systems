// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

using v4f32 = __attribute__((ext_vector_type(4))) float;
using v4f16 = __attribute__((ext_vector_type(4))) _Float16;

/// Minimal chained-MFMA fixture for DBT tests.
///
/// The important shape here is the instruction stream, not the math workload:
/// LLVM should emit two back-to-back v_mfma_f32_16x16x16_f16 instructions where
/// the second consumes the accumulator produced by the first. This exercises
/// AccVGPR liveness, MFMA hazard insertion, and MFMA->WMMA lowering without a
/// loop hiding the dependency.
extern "C" __global__ void mfma_chained_unrolled(const _Float16 *__restrict__ A,
                                                  const _Float16 *__restrict__ B,
                                                  float *__restrict__ C) {
  constexpr unsigned M = 16;
  constexpr unsigned N = 16;
  constexpr unsigned K = 16;
  unsigned lane = threadIdx.x;

  unsigned a_row = lane % M;
  unsigned a_k = (lane / M) * 4;
  v4f16 src_a = {
      (a_k + 0 < K) ? A[a_row * K + a_k + 0] : (_Float16)0,
      (a_k + 1 < K) ? A[a_row * K + a_k + 1] : (_Float16)0,
      (a_k + 2 < K) ? A[a_row * K + a_k + 2] : (_Float16)0,
      (a_k + 3 < K) ? A[a_row * K + a_k + 3] : (_Float16)0,
  };

  unsigned b_col = lane % N;
  unsigned b_k = (lane / N) * 4;
  v4f16 src_b = {
      (b_k + 0 < K) ? B[(b_k + 0) * N + b_col] : (_Float16)0,
      (b_k + 1 < K) ? B[(b_k + 1) * N + b_col] : (_Float16)0,
      (b_k + 2 < K) ? B[(b_k + 2) * N + b_col] : (_Float16)0,
      (b_k + 3 < K) ? B[(b_k + 3) * N + b_col] : (_Float16)0,
  };

  v4f32 acc = {0.0f, 0.0f, 0.0f, 0.0f};
  acc = __builtin_amdgcn_mfma_f32_16x16x16f16(src_a, src_b, acc, 0, 0, 0);
  acc = __builtin_amdgcn_mfma_f32_16x16x16f16(src_a, src_b, acc, 0, 0, 0);

  unsigned col = lane % N;
  unsigned row_group = lane / N;
  for (unsigned r = 0; r < 4; ++r)
    C[(row_group * 4 + r) * N + col] = acc[r];
}
