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

extern __device__ int square_me(int);

__global__ void square_and_save(int* A, int* B) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  B[tid] = square_me(A[tid]);
}

void run_test2() {
  int *A_h, *B_h, *A_d, *B_d;
  A_h = new int[LEN];
  B_h = new int[LEN];
  for (unsigned i = 0; i < LEN; i++) {
    A_h[i] = i;
    B_h[i] = 0;
  }
  size_t valbytes = LEN * sizeof(int);

  HIP_ASSERT(hipMalloc((void**)&A_d, valbytes));
  HIP_ASSERT(hipMalloc((void**)&B_d, valbytes));

  HIP_ASSERT(hipMemcpy(A_d, A_h, valbytes, hipMemcpyHostToDevice));
  hipLaunchKernelGGL(square_and_save, dim3(LEN / 64), dim3(64), 0, 0, A_d, B_d);
  HIP_ASSERT(hipMemcpy(B_h, B_d, valbytes, hipMemcpyDeviceToHost));

  for (unsigned i = 0; i < LEN; i++) {
    assert(A_h[i] * A_h[i] == B_h[i]);
  }

  HIP_ASSERT(hipFree(A_d));
  HIP_ASSERT(hipFree(B_d));
  delete[] A_h;
  delete[] B_h;
  std::cout << "Test Passed!\n";
}

int main() {
  // Run test that generates static lib with ar
  run_test2();
}
