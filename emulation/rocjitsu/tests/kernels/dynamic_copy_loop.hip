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
extern "C" __global__
void dynamic_copy_loop(const unsigned *__restrict__ src, unsigned *__restrict__ dst, unsigned n,
                       unsigned workgroup_size, unsigned stride) {
  // The raw HSA dispatch path does not reliably materialize HIP's blockDim.x
  // implicit field, so the host passes the packet workgroup size explicitly.
  const unsigned first = blockIdx.x * workgroup_size + threadIdx.x;

  for (unsigned i = first; i < n; i += stride)
    dst[i] = src[i];
}
