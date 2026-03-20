/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>


// Kernel Function
__global__ void run_printf(int* count) { *count = printf("Hello World"); }
/**
 * @addtogroup printf printf
 * @{
 * @ingroup PrintfTest
 * `int printf()` -
 * Method to print the content on output device.
 */
/**
 * Test Description
 * ------------------------
 * - Test case to verify the printf return value(Number of Characters)for -mprintf-kind=hostcall
 * compiler option Test source
 * ------------------------
 * - catch/unit/printf/printfHost.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 5.7
 */
HIP_TEST_CASE(Unit_Host_Printf) {
  int pcieAtomic = 0;
  HIP_CHECK(hipDeviceGetAttribute(&pcieAtomic, hipDeviceAttributeHostNativeAtomicSupported, 0));
  if (!pcieAtomic) {
    HipTest::HIP_SKIP_TEST("Device doesn't support pcie atomic, Skipped");
    return;
  }
  int *count{nullptr}, *count_d{nullptr};
  count = reinterpret_cast<int*>(malloc(sizeof(int)));
  HIP_CHECK(hipMalloc(&count_d, sizeof(int)));

  hipLaunchKernelGGL(run_printf, dim3(1), dim3(1), 0, 0, count_d);

  HIP_CHECK(hipMemcpy(count, count_d, sizeof(int), hipMemcpyDeviceToHost));

  std::string str = "Hello World";
  int length = str.length();
#if HT_AMD
  REQUIRE(length == *count);
#else
  REQUIRE(*count == 0);
#endif
  free(count);
  HIP_CHECK(hipFree(count_d));
}

/**
 * End doxygen group PrintfTest.
 * @}
 */
