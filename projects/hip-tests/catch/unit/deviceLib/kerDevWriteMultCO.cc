/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime.h>
#include "./defs.h"

/**
 * This kernel writes to memory allocated in ker_Alloc_MultCodeObj<<<>>>.
 */
extern "C" __global__ void ker_Write_MultCodeObj(int** dev_mem, int value) {
  int myId = threadIdx.x + blockDim.x * blockIdx.x;
  // Check allocated memory in all threads in block before access
  if (*dev_mem == nullptr) {
    printf("Device Allocation Failed in thread = %d \n", myId);
    return;
  }
  // Copy to buffer
  (*dev_mem)[myId] = value;
}
