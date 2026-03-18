/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

#define LEN 512
#define SIZE 2048

class A {
 public:
  __device__ A() { a = threadIdx.x + blockIdx.x * blockDim.x; }

 private:
  int a;
};

static __global__ void kernel(int* Ad) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  new (Ad + tid) A();
}

TEST_CASE(Unit_hipTest_DeviceNewOperator) {
  int *A, *Ad;
  A = new int[LEN];
  for (unsigned i = 0; i < LEN; i++) {
    A[i] = 0;
  }

  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Ad), SIZE));
  hipLaunchKernelGGL(kernel, dim3(1, 1, 1), dim3(LEN, 1, 1), 0, 0, Ad);
  HIP_CHECK(hipMemcpy(A, Ad, SIZE, hipMemcpyDeviceToHost));

  // Validation
  for (unsigned i = 0; i < LEN; i++) {
    REQUIRE(i == A[i]);
  }

  HIP_CHECK(hipFree(Ad));
  delete[] A;
}
