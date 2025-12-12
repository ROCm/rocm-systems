/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup hipExecutionCtx hipExecutionCtx
 * @{
 * @ingroup ExecutionContextTest
 * `hipExecutionCtxStreamCreate` - stream create, negative, and detach tests
 */

#include <hip_test_common.hh>
#include "hip_executionctx_common.hh"

/**
 * Test Description
 * ------------------------
 *  - Creates a green context and its stream using SM resources
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hipExecutionCtxStreamCreate_Sanity) {
  HIP_CHECK(hipSetDevice(0));
  hipDevResourceDesc_t desc{};
  hipError_t ret = GetSmResourceDesc(&desc);
  REQUIRE(ret == hipSuccess);

  hipExecutionCtx_t green_ctx = nullptr;
  HIP_CHECK(hipGreenCtxCreate(&green_ctx, desc, 0, 0));
  REQUIRE(green_ctx != nullptr);

  hipStream_t stream = nullptr;
  HIP_CHECK(hipExecutionCtxStreamCreate(&stream, green_ctx, hipStreamNonBlocking, 0x0));
  REQUIRE(stream != nullptr);

  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipExecutionCtxDestroy(green_ctx));
}

/**
 * Test Description
 * ------------------------
 *  - Negative parameter validation for hipExecutionCtxStreamCreate and hipStreamGetExecutionCtx
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hipExecutionCtxStreamCreate_Negative) {
  HIP_CHECK(hipSetDevice(0));
  hipDevResourceDesc_t desc{};
  hipError_t ret = GetSmResourceDesc(&desc);
  REQUIRE(ret == hipSuccess);

  hipExecutionCtx_t green_ctx = nullptr;
  HIP_CHECK(hipGreenCtxCreate(&green_ctx, desc, 0, 0));
  REQUIRE(green_ctx != nullptr);
  
  hipStream_t valid_stream = nullptr;
  HIP_CHECK(hipExecutionCtxStreamCreate(&valid_stream, green_ctx, hipStreamNonBlocking, 0x0));
  REQUIRE(valid_stream != nullptr);

  hipStream_t stream = nullptr;

  SECTION("stream create with null stream pointer") {
    HIP_CHECK_ERROR(hipExecutionCtxStreamCreate(nullptr, green_ctx, hipStreamNonBlocking, 0x0), hipErrorInvalidValue);
  }

  SECTION("stream create with invalid green context") {
    hipExecutionCtx_t invalid_green_ctx = nullptr;
    HIP_CHECK_ERROR(hipExecutionCtxStreamCreate(&stream, invalid_green_ctx, hipStreamNonBlocking, 0x0),
                     hipErrorInvalidValue);
  }

  SECTION("stream create with invalid flags") {
    constexpr unsigned int kInvalidFlags = 0xFFFFFFFF;
    HIP_CHECK_ERROR(hipExecutionCtxStreamCreate(&stream, green_ctx, kInvalidFlags, 0x0),
                    hipErrorInvalidValue);
  }

  HIP_CHECK(hipStreamDestroy(valid_stream));
  HIP_CHECK(hipExecutionCtxDestroy(green_ctx));
}

/**
 * Test Description
 * ------------------------
 *  - Verifies that after hipExecutionCtxDestroy, stream-accepting APIs
 *    return hipErrorStreamDetached while hipStreamDestroy still succeeds.
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hipExecutionCtxStreamDestroy_Negative) {
  HIP_CHECK(hipSetDevice(0));
  hipDevResourceDesc_t desc{};
  hipError_t ret = GetSmResourceDesc(&desc);
  REQUIRE(ret == hipSuccess);

  hipExecutionCtx_t green_ctx = nullptr;
  HIP_CHECK(hipGreenCtxCreate(&green_ctx, desc, 0, 0));
  REQUIRE(green_ctx != nullptr);

  hipStream_t stream = nullptr;
  HIP_CHECK(hipExecutionCtxStreamCreate(&stream, green_ctx, hipStreamNonBlocking, 0x0));
  REQUIRE(stream != nullptr);

  HIP_CHECK(hipStreamSynchronize(stream));

  HIP_CHECK(hipExecutionCtxDestroy(green_ctx));

  SECTION("hipStreamSynchronize returns hipErrorStreamDetached") {
    HIP_CHECK_ERROR(hipStreamSynchronize(stream), hipErrorStreamDetached);
  }

  SECTION("hipStreamQuery returns hipErrorStreamDetached") {
    HIP_CHECK_ERROR(hipStreamQuery(stream), hipErrorStreamDetached);
  }

  SECTION("hipStreamDestroy succeeds on detached stream") {
    HIP_CHECK(hipStreamDestroy(stream));
    stream = nullptr;
  }

  if (stream != nullptr) {
    HIP_CHECK(hipStreamDestroy(stream));
  }
}

/**
 * End doxygen group hipExecutionCtx.
 * @}
 */
