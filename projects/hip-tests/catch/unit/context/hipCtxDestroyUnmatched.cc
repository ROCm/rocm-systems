/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

/**
 * Test Description
 * ------------------------
 *  - Validates that hipCtxDestroy rejects a handle the caller never created:
 *    -# When the handle comes from hipCtxGetCurrent
 *      - Expected output: return `hipErrorInvalidContext`
 *    -# When a matched hipCtxCreate/hipCtxDestroy pair is destroyed a second time
 *      - Expected output: return `hipErrorInvalidContext`
 *  - Nothing is allocated beforehand; a live allocation masks the failure.
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

  // hipCtxCreate makes the new context current, so restore the original before
  // destroying it and leaving no current context behind.
  hipCtx_t created = nullptr;
  HIP_CHECK(hipCtxCreate(&created, 0, 0));
  HIP_CHECK(hipCtxSetCurrent(context));
  HIP_CHECK(hipCtxDestroy(created));
  REQUIRE(hipCtxDestroy(created) == hipErrorInvalidContext);

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
 *      - Expected output: return `hipErrorInvalidContext`, context stays active
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
