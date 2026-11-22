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

// Launch bounds below the device maximum, for the occupancy block size suggestion tests. 100 is
// not a multiple of any wavefront size and 96 is not a multiple of 64.
extern "C" __global__ void __launch_bounds__(128) BoundedKernel128(int* a) {
  a[blockIdx.x * blockDim.x + threadIdx.x] = 1;
}

extern "C" __global__ void __launch_bounds__(100) BoundedKernel100(int* a) {
  a[blockIdx.x * blockDim.x + threadIdx.x] = 1;
}

extern "C" __global__ void __launch_bounds__(96) BoundedKernel96(int* a) {
  a[blockIdx.x * blockDim.x + threadIdx.x] = 1;
}
