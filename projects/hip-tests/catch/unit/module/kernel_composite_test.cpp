/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime.h>
constexpr int GLOBAL_BUF_SIZE = 2048;

__device__ float deviceGlobalFloat;
__device__ int deviceGlobalInt1;
__device__ int deviceGlobalInt2;
__device__ short deviceGlobalShort;  // NOLINT
__device__ char deviceGlobalChar;

__device__ int getSquareOfGlobalFloat() {
  return static_cast<int>(deviceGlobalFloat * deviceGlobalFloat);
}

extern "C" __global__ void testWeightedCopy(int* a, int* b) {
  int tx = threadIdx.x;
  b[tx] = deviceGlobalInt1 * a[tx] + deviceGlobalInt2 + static_cast<int>(deviceGlobalShort) +
          static_cast<int>(deviceGlobalChar) + getSquareOfGlobalFloat();
}
