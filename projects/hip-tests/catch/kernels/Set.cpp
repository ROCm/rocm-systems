/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <kernels.hh>

__global__ void Set(int* Ad, int val) {
  const unsigned int tx = threadIdx.x + blockIdx.x * blockDim.x;
  Ad[tx] = val;
}
