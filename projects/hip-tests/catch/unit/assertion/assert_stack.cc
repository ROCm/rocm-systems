/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// The tests build with -DNDEBUG, which removes assert(); this file needs it in the kernel.
#undef NDEBUG

#include <cassert>
#include <hip_test_common.hh>

/**
 * @addtogroup assert assert
 * @{
 * @ingroup DeviceLanguageTest
 */

__global__ void AssertOnlyKernel(int* x) { assert(x[threadIdx.x] != 12345); }

/**
 * Test Description
 * ------------------------
 *  - A kernel whose only stack use is an assert must not carry the assertion message format on
 *    its stack: __assert_fail keeps the format in static storage, so the kernel's private memory
 *    stays below the format's 48 bytes.
 * Test source
 * ------------------------
 *  - unit/assertion/assert_stack.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.14
 */
HIP_TEST_CASE(Unit_Assert_Positive_FormatNotOnStack) {
  constexpr size_t kFormatSize = sizeof("%s:%u: %s: Device-side assertion `%s' failed.\n");
  hipFuncAttributes attributes{};
  HIP_CHECK(hipFuncGetAttributes(&attributes, reinterpret_cast<const void*>(AssertOnlyKernel)));
  INFO("localSizeBytes " << attributes.localSizeBytes);
  REQUIRE(attributes.localSizeBytes > 0);  // the assert is compiled in
  REQUIRE(attributes.localSizeBytes < kFormatSize);
}

/**
 * End doxygen group DeviceLanguageTest.
 * @}
 */
