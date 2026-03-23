/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime.h>

extern "C" __global__ void SimpleKernel(int* a, int* b) {
  int tx = threadIdx.x;
  b[tx] = a[tx];
}
