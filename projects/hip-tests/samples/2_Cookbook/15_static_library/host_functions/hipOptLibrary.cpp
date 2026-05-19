/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <iostream>
#include <stdexcept>
#include <cassert>

#define HIP_ASSERT(status)                                                                         \
  {                                                                                                \
    if ((status != hipSuccess)) {                                                                  \
      std::cerr << "Failed in: " << __LINE__ << " on hip call: " #status << std::endl;             \
      throw std::runtime_error("generic failure");                                                 \
    }                                                                                              \
  }
#define LEN 512

__global__ void copy(uint32_t* A, uint32_t* B) {
  size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  B[tid] = A[tid];
}

void run_test1() {
  uint32_t *A_h, *B_h, *A_d, *B_d;
  size_t valbytes = LEN * sizeof(uint32_t);

  A_h = (uint32_t*)malloc(valbytes);
  B_h = (uint32_t*)malloc(valbytes);
  for (uint32_t i = 0; i < LEN; i++) {
    A_h[i] = i;
    B_h[i] = 0;
  }

  HIP_ASSERT(hipMalloc((void**)&A_d, valbytes));
  HIP_ASSERT(hipMalloc((void**)&B_d, valbytes));

  HIP_ASSERT(hipMemcpy(A_d, A_h, valbytes, hipMemcpyHostToDevice));
  hipLaunchKernelGGL(copy, dim3(LEN / 64), dim3(64), 0, 0, A_d, B_d);
  HIP_ASSERT(hipMemcpy(B_h, B_d, valbytes, hipMemcpyDeviceToHost));

  for (uint32_t i = 0; i < LEN; i++) {
    assert(A_h[i] == B_h[i]);
  }

  HIP_ASSERT(hipFree(A_d));
  HIP_ASSERT(hipFree(B_d));
  free(A_h);
  free(B_h);
  std::cout << "Test Passed!\n";
}
