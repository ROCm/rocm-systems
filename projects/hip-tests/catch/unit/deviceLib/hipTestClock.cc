/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_kernels.hh>
#include <hip_test_common.hh>
#include <hip_test_checkers.hh>

#define LEN 512
#define SIZE (LEN * sizeof(int64_t))

static __global__ void kernel1(int64_t* Ad) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  Ad[tid] = clock() + clock64() + __clock() + __clock64();
}

static __global__ void kernel2(int64_t* Ad) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  Ad[tid] = clock() + clock64() + __clock() + __clock64() - Ad[tid];
}

HIP_TEST_CASE(Unit_hipTestClock) {
  int64_t *A, *Ad;
  A = new int64_t[LEN];
  for (unsigned i = 0; i < LEN; i++) {
    A[i] = 0;
  }
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Ad), SIZE));
  HIP_CHECK(hipMemcpy(Ad, A, SIZE, hipMemcpyHostToDevice));
  hipLaunchKernelGGL(kernel1, dim3(1, 1, 1), dim3(LEN, 1, 1), 0, 0, Ad);
  hipLaunchKernelGGL(kernel2, dim3(1, 1, 1), dim3(LEN, 1, 1), 0, 0, Ad);
  HIP_CHECK(hipMemcpy(A, Ad, SIZE, hipMemcpyDeviceToHost));
  for (unsigned i = 0; i < LEN; i++) {
    assert(0 != A[i]);
  }

  HIP_CHECK(hipFree(Ad));
  delete[] A;
}
