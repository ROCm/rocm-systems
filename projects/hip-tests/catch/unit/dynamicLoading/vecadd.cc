/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime.h>

__global__ void kerAdd(int* in_a, int* in_b, int* out_c, int nelem) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= nelem) {
    return;
  }
  out_c[id] = in_a[id] + in_b[id];
}
