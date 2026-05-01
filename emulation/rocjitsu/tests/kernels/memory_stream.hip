// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>

/// Memory-focused global load/store kernel for CDNA4->RDNA3 DBT coverage.
///
/// Each lane copies one stream from A to C. This keeps the representative
/// kernel focused on global loads/stores without relying on matrix, AccVGPR, or
/// multi-output address patterns.
extern "C" __global__ void memory_stream(const float *__restrict__ A, float *__restrict__ C,
                                         unsigned N) {
  constexpr unsigned BLOCK_SIZE = 64;
  unsigned gid = blockIdx.x * BLOCK_SIZE + threadIdx.x;

  if (gid < N) {
    C[gid] = A[gid];
  }
}
