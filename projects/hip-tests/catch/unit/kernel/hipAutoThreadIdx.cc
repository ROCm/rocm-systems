/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

/**
* @addtogroup hipLaunchKernelGGL
* @{
* @ingroup KernelTest
* `void hipLaunchKernelGGL(F kernel, const dim3& numBlocks, const dim3& dimBlocks,
   std::uint32_t sharedMemBytes, hipStream_t stream, Args... args)` -
* Method to invocate kernel functions
*/

static __global__ void vecSqrSingBlk(int* A_d, size_t NELEM) {
  if (0 == blockIdx.x) {
    for (auto i = threadIdx.x; i < NELEM; i += blockDim.x) {
      A_d[i] = A_d[i] * A_d[i];
    }
  }
}

/**
 * Test Description
 * ------------------------
 * - Test case to check usage of C++11 auto for kernel variables.
 * Test source
 * ------------------------
 * - catch/unit/kernel/hipAutoThreadIdx.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 6.1
 */
HIP_TEST_CASE(Unit_kernel_Assign_threadIdx_to_auto) {
  int* A_d;
  const unsigned blocks = 256;
  const unsigned threadsPerBlock = 128;
  size_t N = (blocks * threadsPerBlock);
  std::vector<int> A_h(N), C_h(N);
  size_t Nbytes = N * sizeof(float);

  // Fill with data
  for (size_t i = 0; i < N; i++) {
    A_h[i] = i;
  }

  // Transfer data and perform operations on GPU
  HIP_CHECK(hipMalloc(&A_d, Nbytes));
  HIP_CHECK(hipMemcpy(A_d, A_h.data(), Nbytes, hipMemcpyHostToDevice));
  hipLaunchKernelGGL(vecSqrSingBlk, dim3(blocks), dim3(threadsPerBlock), 0, 0, A_d, N);
  HIP_CHECK(hipMemcpy(C_h.data(), A_d, Nbytes, hipMemcpyDeviceToHost));
  HIP_CHECK(hipDeviceSynchronize());
  for (size_t i = 0; i < N; i++) {
    REQUIRE(C_h[i] == (i * i));
  }
  HIP_CHECK(hipFree(A_d));
}
