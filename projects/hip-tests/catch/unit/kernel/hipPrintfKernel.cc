/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <vector>
#include <cstring>
#include "../kernel/printf_common.h"

#define HIP_ENABLE_PRINTF

__global__ void run_printf() { printf("Hello World"); }
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
 * - Test case to check printf function via kernel call.

 * Test source
 * ------------------------
 * - catch/unit/kernel/hipPrintfKernel.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 5.6
 */
TEST_CASE(Unit_kernel_ChkPrintf) {
  int device_count = 0;
  HIP_CHECK(hipGetDeviceCount(&device_count));
  CaptureStream capture;
  std::string check = "Hello World";
  for (int i = 0; i < device_count; ++i) {
    HIP_CHECK(hipSetDevice(i));
    if (!HipTest::isPcieAtomicSupported()) continue;

    capture.beginCapture();
    hipLaunchKernelGGL(run_printf, dim3(1), dim3(1), 0, 0);
    HIP_CHECK(hipDeviceSynchronize());
    capture.endCapture();

    auto CapturedData = capture.getCapturedData();
    int result = check.compare(CapturedData);
    REQUIRE(result == 0);
  }
}

/**
 * End doxygen group KernelTest.
 * @}
 */
