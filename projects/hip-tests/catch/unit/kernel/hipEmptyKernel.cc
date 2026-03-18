/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_kernels.hh>
#include <hip_test_checkers.hh>
#include <hip_test_common.hh>


#pragma clang diagnostic ignored "-Wunused-parameter"

__global__ void Empty(int param) {}

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
 *    - pass empty Kernel function.

 * Test source
 * ------------------------
 *    - catch/unit/kernel/hipEmptyKernel.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.5
 */

TEST_CASE(Unit_hipEmptyKernel) {
  hipLaunchKernelGGL(HIP_KERNEL_NAME(Empty), dim3(1), dim3(1), 0, 0, 0);
  HIP_CHECK(hipDeviceSynchronize());
}

/**
 * End doxygen group KernelTest.
 * @}
 */
