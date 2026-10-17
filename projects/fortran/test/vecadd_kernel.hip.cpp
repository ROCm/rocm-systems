// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>

__global__ void vector_add(double *out, double *a, double *b, int n)
{
  if (n <= 0)
    return;
  const size_t count = static_cast<size_t>(n);
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;

  for (size_t i = index; i < count; i += stride)
    out[i] = a[i] + b[i];
}
