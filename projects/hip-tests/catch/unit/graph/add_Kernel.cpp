/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime.h>
extern "C" __global__ void Add(int* a, int* b, int* c) {
  size_t tx = (blockIdx.x * blockDim.x + threadIdx.x);
  c[tx] = a[tx] + b[tx];
}
