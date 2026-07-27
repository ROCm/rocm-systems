/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Source for oob_copyKernelCompressed.code: a valid compressed offload bundle
// loaded by hipFatBinaryBounds.cc's false-positive guard.

#include <hip/hip_runtime.h>

extern "C" __global__ void copy_ker(int* Ad, int* Bd, size_t size) {
  int myId = threadIdx.x + blockDim.x * blockIdx.x;
  if (myId < size) {
    Bd[myId] = Ad[myId];
  }
}
