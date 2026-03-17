/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_kernels.hh>
#include <hip_test_checkers.hh>
#include <hip_test_common.hh>


#define LEN (16 * 1024)
#define SIZE (LEN * sizeof(float))

__global__ void vectorAdd(float* Ad, float* Bd) {
  extern __shared__ float sBd[];
  int tx = threadIdx.x;
  for (int i = 0; i < LEN / 64; i++) {
    sBd[tx + i * 64] = Ad[tx + i * 64] + 1.0f;
    Bd[tx + i * 64] = sBd[tx + i * 64];
  }
}

/**
* @addtogroup hipLaunchKernelGGL hipLaunchKernelGGL
* @{
* @ingroup KernelTest
* `void hipLaunchKernelGGL(F kernel, const dim3& numBlocks, const dim3& dimBlocks,
   std::uint32_t sharedMemBytes, hipStream_t stream, Args... args)` -
* Method to invocate kernel functions
*/

/**
 * Test Description
 * ------------------------
 *    - Assign max dynamic shared memory to kernel function and
 * verify the results.

 * Test source
 * ------------------------
 *    - catch/unit/kernel/hipDynamicShared2.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.5
 */

HIP_TEST_CASE(Unit_hipDynamicShared2) {
  float *A, *B, *Ad, *Bd;
  A = new float[LEN];
  B = new float[LEN];
  for (int i = 0; i < LEN; i++) {
    A[i] = 1.0f;
    B[i] = 1.0f;
  }
  HIP_CHECK(hipMalloc(&Ad, SIZE));
  HIP_CHECK(hipMalloc(&Bd, SIZE));
  HIP_CHECK(hipMemcpy(Ad, A, SIZE, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(Bd, B, SIZE, hipMemcpyHostToDevice));

  hipError_t ret = hipFuncSetAttribute(reinterpret_cast<const void*>(&vectorAdd),
                                       hipFuncAttributeMaxDynamicSharedMemorySize, SIZE);

  REQUIRE(ret == hipSuccess);
  hipLaunchKernelGGL(vectorAdd, dim3(1, 1, 1), dim3(64, 1, 1), SIZE, 0, Ad, Bd);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMemcpy(B, Bd, SIZE, hipMemcpyDeviceToHost));
  for (int i = 0; i < LEN; i++) {
    assert(B[i] > 1.0f && B[i] < 3.0f);
  }
  HIP_CHECK(hipFree(Ad));
  HIP_CHECK(hipFree(Bd));

  delete[] A;
  delete[] B;
}

/**
 * End doxygen group KernelTest.
 * @}
 */
