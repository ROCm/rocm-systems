/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime.h>
#include "./defs.h"

/**
 * This kernel allocates memory in thread 0.
 */
extern "C" __global__ void ker_Alloc_MultCodeObj(int** dev_mem, int test_type) {
  int myId = threadIdx.x + blockDim.x * blockIdx.x;
  // Allocate memory in thread 0 of block 0
  if (0 == myId) {
    if (test_type == TEST_MALLOC_FREE) {
      *dev_mem = reinterpret_cast<int*>(malloc(blockDim.x * gridDim.x * sizeof(int)));
    } else {
      *dev_mem = reinterpret_cast<int*>(new int[blockDim.x * gridDim.x]);
    }
  }
}
