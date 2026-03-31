/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "hip/hip_runtime.h"

#define ARRAY_SIZE (16)

__device__ float myDeviceGlobal;
__device__ float myDeviceGlobalArray[16];

extern "C" __global__ void hello_world(const float* a, float* b) {
  int tx = threadIdx.x;
  b[tx] = a[tx];
}

extern "C" __global__ void test_globals(const float* a, float* b) {
  int tx = threadIdx.x;
  b[tx] = a[tx] + myDeviceGlobal + myDeviceGlobalArray[tx % ARRAY_SIZE];
}
