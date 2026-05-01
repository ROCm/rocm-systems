// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>

/// Memory-focused global load/store kernel for CDNA4->RDNA3 DBT coverage.
///
/// Each lane copies A to C and computes D = 0.5*A + B. This covers multiple
/// pointer arguments, multiple global loads, and separate output streams.
extern "C" __global__ void memory_stream(const float *__restrict__ A,
                                         const float *__restrict__ B, float *__restrict__ C,
                                         float *__restrict__ D, unsigned N) {
  constexpr unsigned BLOCK_SIZE = 64;
  unsigned gid = blockIdx.x * BLOCK_SIZE + threadIdx.x;

  if (gid < N) {
    C[gid] = A[gid];
    D[gid] = 0.5f * A[gid] + B[gid];
  }
}
