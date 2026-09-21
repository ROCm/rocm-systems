/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup hipExecutionCtx hipExecutionCtx
 * @{
 * @ingroup ExecutionContextTest
 * `hipExecutionCtx*` APIs - basic sanity tests
 */

#include <hip_test_common.hh>
#include <hip_test_kernels.hh>
#include "hip_executionctx_common.hh"

/**
 * Test Description
 * ------------------------
 *  - Creates and destroys a green context using SM resources
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hipGreenCtxCreateDestroy_Sanity) {
  HIP_CHECK(hipSetDevice(0));
  hipDevResourceDesc_t desc{};
  hipError_t ret = GetSmResourceDesc(&desc);
  REQUIRE(ret == hipSuccess);

  hipExecutionCtx_t green_ctx = nullptr;
  HIP_CHECK(hipGreenCtxCreate(&green_ctx, desc, 0, 0));
  REQUIRE(green_ctx != nullptr);
  HIP_CHECK(hipExecutionCtxDestroy(green_ctx));
}

/**
 * Test Description
 * ------------------------
 *  - Launches a vectorADD kernel on a execution context stream
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hipGreenCtx_kernelLaunch_Basic) {
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

  constexpr size_t kNumElements = 1024;
  const size_t kBytes = kNumElements * sizeof(int);
  int* h_a = reinterpret_cast<int*>(malloc(kBytes));
  int* h_b = reinterpret_cast<int*>(malloc(kBytes));
  int* h_c = reinterpret_cast<int*>(malloc(kBytes));
  REQUIRE(h_a != nullptr);
  REQUIRE(h_b != nullptr);
  REQUIRE(h_c != nullptr);

  for (size_t i = 0; i < kNumElements; ++i) {
    h_a[i] = static_cast<int>(i);
    h_b[i] = static_cast<int>(2 * i);
    h_c[i] = 0;
  }

  int* d_a = nullptr;
  int* d_b = nullptr;
  int* d_c = nullptr;
  HIP_CHECK(hipMalloc(&d_a, kBytes));
  HIP_CHECK(hipMalloc(&d_b, kBytes));
  HIP_CHECK(hipMalloc(&d_c, kBytes));

  HIP_CHECK(hipMemcpyAsync(d_a, h_a, kBytes, hipMemcpyHostToDevice, stream));
  HIP_CHECK(hipMemcpyAsync(d_b, h_b, kBytes, hipMemcpyHostToDevice, stream));

  constexpr int kThreads = 256;
  const int blocks = static_cast<int>((kNumElements + kThreads - 1) / kThreads);
  HipTest::vectorADD<<<blocks, kThreads, 0, stream>>>(d_a, d_b, d_c, kNumElements);
  HIP_CHECK(hipGetLastError());

  HIP_CHECK(hipMemcpyAsync(h_c, d_c, kBytes, hipMemcpyDeviceToHost, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  for (size_t i = 0; i < kNumElements; ++i) {
    REQUIRE(h_c[i] == h_a[i] + h_b[i]);
  }

  HIP_CHECK(hipFree(d_a));
  HIP_CHECK(hipFree(d_b));
  HIP_CHECK(hipFree(d_c));
  free(h_a);
  free(h_b);
  free(h_c);
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipExecutionCtxDestroy(green_ctx));
}

/**
 * Test Description
 * ------------------------
 *  - Negative parameter validation for hipGreenCtxCreate
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hipGreenCtxCreate_Negative) {
  HIP_CHECK(hipSetDevice(0));
  hipDevResourceDesc_t desc{};
  hipExecutionCtx_t green_ctx = nullptr;
  hipError_t ret = GetSmResourceDesc(&desc);
  REQUIRE(ret == hipSuccess);

  SECTION("Invalid green context output pointer") {
    HIP_CHECK_ERROR(hipGreenCtxCreate(nullptr, desc, 0, 0),
                    hipErrorInvalidValue);
  }

  SECTION("Invalid flags") {
    constexpr unsigned int kInvalidFlags = 0xFFFFFFFF;
    HIP_CHECK_ERROR(hipGreenCtxCreate(&green_ctx, desc, 0, kInvalidFlags), hipErrorInvalidValue);
  }

  SECTION("Invalid Device") {
    HIP_CHECK_ERROR(hipGreenCtxCreate(&green_ctx, desc, -1, 0), hipErrorInvalidDevice);
  }

  // The descriptor is owned by the green context on a successful create and
  // freed by hipExecutionCtxDestroy. Consume it here so it is not leaked, since
  // the negative cases above never transfer ownership.
  HIP_CHECK(hipGreenCtxCreate(&green_ctx, desc, 0, 0));
  REQUIRE(green_ctx != nullptr);
  HIP_CHECK(hipExecutionCtxDestroy(green_ctx));
}

/**
 * End doxygen group hipExecutionCtx.
 * @}
 */
