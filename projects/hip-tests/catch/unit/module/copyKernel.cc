/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime.h>

extern "C" __global__ void copy_ker(int* Ad, int* Bd, size_t size) {
  int myId = threadIdx.x + blockDim.x * blockIdx.x;
  if (myId < size) {
    Bd[myId] = Ad[myId];
  }
}
