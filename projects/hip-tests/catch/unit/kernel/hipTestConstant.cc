/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>


#define LEN 512
#define SIZE 2048

__constant__ int Value[LEN];

static __global__ void Get(int* Ad) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  Ad[tid] = Value[tid];
}
/**
* @addtogroup hipLaunchKernelGGL
* @{
* @ingroup KernelTest
* `void hipLaunchKernelGGL(F kernel, const dim3& numBlocks, const dim3& dimBlocks,
   std::uint32_t sharedMemBytes, hipStream_t stream, Args... args)` -
* Method to invocate kernel functions
*/

/**
 * Test Description
 * ------------------------
 * - Test case to check const variable via kernel call.

 * Test source
 * ------------------------
 * - catch/unit/kernel/hipTestConstant.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 5.6
 */

TEST_CASE(Unit_kernel_chkConstantViaKernel) {
  int *A, *B, *Ad;
  A = new int[LEN];
  B = new int[LEN];
  for (unsigned i = 0; i < LEN; i++) {
    A[i] = -1 * i;
    B[i] = 0;
  }

  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Ad), SIZE));

  HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(Value), A, SIZE, 0, hipMemcpyHostToDevice));
  hipLaunchKernelGGL(Get, dim3(1, 1, 1), dim3(LEN, 1, 1), 0, 0, Ad);
  HIP_CHECK(hipMemcpy(B, Ad, SIZE, hipMemcpyDeviceToHost));

  for (unsigned i = 0; i < LEN; i++) {
    REQUIRE(A[i] == B[i]);
  }
  delete[] A;
  delete[] B;
  HIP_CHECK(hipFree(Ad));
}


/**
 * End doxygen group KernelTest.
 * @}
 */
