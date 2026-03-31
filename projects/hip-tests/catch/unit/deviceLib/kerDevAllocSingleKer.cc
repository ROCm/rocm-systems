/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime.h>
#include "./defs.h"
/**
 * This kernel allocates and deallocates memory in every thread.
 */
extern "C" __global__ void ker_TestDynamicAllocInAllThreads_CodeObj(int* outputBuf, int test_type,
                                                                    int value,
                                                                    size_t perThreadSize) {
  int myId = threadIdx.x + blockDim.x * blockIdx.x;
  // Allocate
  size_t size = 0;
  int* ptr = nullptr;
  if (test_type == TEST_MALLOC_FREE) {
    size = perThreadSize * sizeof(int);
    ptr = reinterpret_cast<int*>(malloc(size));
  } else {
    size = perThreadSize;
    ptr = new int[perThreadSize];
  }
  if (ptr == nullptr) {
    printf("Device Allocation in thread %d Failed! \n", myId);
    return;
  }
  // Set memory
  for (size_t idx = 0; idx < perThreadSize; idx++) {
    ptr[idx] = value;
  }
  // Copy to output buffer
  for (size_t idx = 0; idx < perThreadSize; idx++) {
    outputBuf[myId * perThreadSize + idx] = ptr[idx];
  }
  // Free memory
  if (test_type == TEST_MALLOC_FREE) {
    free(ptr);
  } else {
    delete[] ptr;
  }
}
