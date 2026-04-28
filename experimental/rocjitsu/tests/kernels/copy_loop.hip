// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>

/// Grid-stride copy fixture for DBT control-flow tests.
///
/// The dispatch intentionally uses fewer work-items than elements so every
/// live lane takes the loop backedge multiple times.
extern "C" __global__ void copy_loop(const float *__restrict__ A, float *__restrict__ C,
                                     unsigned N) {
  unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned stride = gridDim.x * blockDim.x;

  for (unsigned i = gid; i < N; i += stride)
    C[i] = A[i];
}
