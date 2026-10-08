// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>

/// Executes one CDNA4 packed FP32-to-BF16 conversion per lane.
///
/// The DBT dispatch test feeds one wave of FP32 bit patterns and checks the
/// packed BF16 output exactly.  Keeping the conversion in inline assembly makes
/// this fixture a direct probe of the `v_cvt_pk_bf16_f32` semantic lowering
/// instead of a compiler-selected BF16 conversion sequence.
extern "C" __global__ void cvt_pk_bf16_f32_kernel(const float *__restrict__ input,
                                                   unsigned *__restrict__ output) {
  const unsigned lane = threadIdx.x;
  const float lo = input[lane * 2 + 0];
  const float hi = input[lane * 2 + 1];

  unsigned packed;
  __asm__ volatile("v_cvt_pk_bf16_f32 %0, %1, %2"
                   : "=v"(packed)
                   : "v"(lo), "v"(hi));
  output[lane] = packed;
}
