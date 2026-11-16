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
