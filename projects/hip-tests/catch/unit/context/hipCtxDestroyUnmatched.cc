/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

#if HT_AMD
/**
 * Test Description
 * ------------------------
 *  - Validates that hipCtxDestroy rejects a handle the caller never created:
 *    -# When the handle comes from hipCtxGetCurrent
 *      - Expected output: return `hipErrorInvalidContext`
 *    -# The device context remains usable afterwards
 *      - Expected output: `hipMalloc` returns `hipSuccess`
 *  - Nothing is allocated before the destroy. A live allocation holds its own
 *    reference on the device context and would hide an erroneous release.
 * Test source
 * ------------------------
 *  - unit/context/hipCtxDestroyUnmatched.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hipCtxDestroy_WithoutCreate_Negative) {
  HIP_CHECK(hipSetDevice(0));

  hipCtx_t context = nullptr;
  HIP_CHECK(hipCtxGetCurrent(&context));
  REQUIRE(context != nullptr);

  REQUIRE(hipCtxDestroy(context) == hipErrorInvalidContext);

  void* ptr = nullptr;
  HIP_CHECK(hipMalloc(&ptr, 1024));
  REQUIRE(ptr != nullptr);
  HIP_CHECK(hipFree(ptr));
}

/**
 * Test Description
 * ------------------------
 *  - Validates that hipCtxDestroy rejects a primary context handle:
 *    -# When the handle comes from hipDevicePrimaryCtxRetain
 *      - Expected output: return `hipErrorInvalidContext`
 *    -# The primary context is still reported active
 *      - Expected output: is_active = 1
 *    -# The device context remains usable afterwards
 *      - Expected output: `hipMalloc` returns `hipSuccess`
 * Test source
 * ------------------------
 *  - unit/context/hipCtxDestroyUnmatched.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hipCtxDestroy_AfterPrimaryCtxRetain_Negative) {
  HIP_CHECK(hipSetDevice(0));

  hipDevice_t device = 0;
  HIP_CHECK(hipDeviceGet(&device, 0));

  hipCtx_t context = nullptr;
  HIP_CHECK(hipDevicePrimaryCtxRetain(&context, device));
  REQUIRE(context != nullptr);

  REQUIRE(hipCtxDestroy(context) == hipErrorInvalidContext);

  unsigned int flags = 0;
  int is_active = 0;
  HIP_CHECK(hipDevicePrimaryCtxGetState(device, &flags, &is_active));
  REQUIRE(is_active == 1);

  void* ptr = nullptr;
  HIP_CHECK(hipMalloc(&ptr, 1024));
  REQUIRE(ptr != nullptr);
  HIP_CHECK(hipFree(ptr));

  HIP_CHECK(hipDevicePrimaryCtxRelease(device));
}
#endif  // HT_AMD
