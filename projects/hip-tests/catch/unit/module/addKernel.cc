/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime.h>

__device__ int globalDevData = 10;

extern "C" __global__ void addKernel(int* a, int size) {
  int offset = blockDim.x * blockIdx.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = offset; i < size; i += stride) {
    a[i] += 2;
  }
}

texture<float, 2> tex;

extern "C" __global__ void sampleModuleKernel() {}
