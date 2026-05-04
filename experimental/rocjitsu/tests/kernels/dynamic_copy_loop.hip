// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>

/// Runtime-sized copy kernel used by CDNA4->CDNA3 DBT dispatch tests.
///
/// The grid is intentionally allowed to be smaller than N. Each lane walks an
/// explicit-stride loop, so the host test can dispatch the same kernel object
/// for many dynamic sizes without recompiling or changing the workgroup shape.
///
/// The stride is passed as a normal kernarg instead of reading gridDim.x. These
/// tests dispatch the code object through raw HSA, not the HIP runtime, and raw
/// HSA launchers do not provide every HIP implicit runtime field.
extern "C" __global__ void dynamic_copy_loop(const unsigned *__restrict__ src,
                                             unsigned *__restrict__ dst, unsigned n,
                                             unsigned stride) {
  constexpr unsigned kBlockSize = 64;
  const unsigned first = blockIdx.x * kBlockSize + threadIdx.x;

  for (unsigned i = first; i < n; i += stride)
    dst[i] = src[i];
}
