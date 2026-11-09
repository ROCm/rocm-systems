// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>

extern "C" {
__device__ int value = 7;

__global__ void selected(int *output) {
  __syncthreads();
  output[threadIdx.x] = value;
}

__global__ void sibling(int *output) {
  output[threadIdx.x] = value;
}
}
