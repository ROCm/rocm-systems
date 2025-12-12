/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>


#define LEN 512
#define SIZE 2048
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
 * - Test case to check constant global variable and global array via kernel call.

 * Test source
 * ------------------------
 * - catch/unit/kernel/hipTestGlobalVariable.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 5.6
 */
__constant__ int ConstantGlobalVar = 123;

static __global__ void kernel(int* Ad) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  Ad[tid] = ConstantGlobalVar;
}

void runTestConstantGlobalVar() {
  int *A, *Ad;
  A = new int[LEN];
  for (unsigned i = 0; i < LEN; i++) {
    A[i] = 0;
  }
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Ad), SIZE));
  hipLaunchKernelGGL(kernel, dim3(1, 1, 1), dim3(LEN, 1, 1), 0, 0, Ad);
  HIP_CHECK(hipMemcpy(A, Ad, SIZE, hipMemcpyDeviceToHost));

  for (unsigned i = 0; i < LEN; i++) {
    REQUIRE(123 == A[i]);
  }
  delete[] A;
  HIP_CHECK(hipFree(Ad));
}

__device__ int GlobalArray[LEN];
static __global__ void kernelWrite() {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  GlobalArray[tid] = tid;
}
static __global__ void kernelRead(int* Ad) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  Ad[tid] = GlobalArray[tid];
}

void runTestGlobalArray() {
  int *A, *Ad;
  A = new int[LEN];
  for (unsigned i = 0; i < LEN; i++) {
    A[i] = 0;
  }

  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Ad), SIZE));
  hipLaunchKernelGGL(kernelWrite, dim3(1, 1, 1), dim3(LEN, 1, 1), 0, 0);
  hipLaunchKernelGGL(kernelRead, dim3(1, 1, 1), dim3(LEN, 1, 1), 0, 0, Ad);
  HIP_CHECK(hipMemcpy(A, Ad, SIZE, hipMemcpyDeviceToHost));

  for (unsigned i = 0; i < LEN; i++) {
    REQUIRE(i == A[i]);
  }
  delete[] A;
  HIP_CHECK(hipFree(Ad));
}

HIP_TEST_CASE(Unit_kernel_chkGlobalArrAndGlobalVaribleViaKernelFn) {
  runTestConstantGlobalVar();
  runTestGlobalArray();
}

/**
 * End doxygen group KernelTest.
 * @}
 */
