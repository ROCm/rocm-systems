/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime.h>
#include "./defs.h"

/**
 * This kernel copies the contents of memory allocated in
 * ker_Alloc_MultCodeObj<<<>>> to host and deletes the memory
 * from thread 0.
 */
extern "C" __global__ void ker_Free_MultCodeObj(int* outputBuf, int** dev_mem, int test_type) {
  int myId = threadIdx.x + blockDim.x * blockIdx.x;
  // Check allocated memory in all threads in block before access
  if (*dev_mem == nullptr) {
    printf("Device Allocation Failed in thread = %d \n", myId);
    return;
  }

  if (0 == myId) {
    for (size_t idx = 0; idx < (blockDim.x * gridDim.x); idx++) {
      outputBuf[idx] = (*dev_mem)[idx];
    }
    if (test_type == TEST_MALLOC_FREE) {
      free(*dev_mem);
    } else {
      delete[] (*dev_mem);
    }
  }
}
