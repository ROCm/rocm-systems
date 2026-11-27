/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip/hip_runtime_api.h>
#include <utils.hh>

#include <atomic>
#include <cstring>
#include <string>

/**
 * @addtogroup __hipOnError __hipOnError
 * @{
 * @ingroup ErrorHandlingTest
 * `void __hipOnError(const void* err_info)` -
 * No-op hook the runtime calls when an API fails, for debugger breakpoints.
 */

namespace {
struct HipErrInfoV1 {
  uint32_t version;
  uint32_t err_no;
  const char* err_name;
  const char* err_string;
};

std::atomic<int> g_calls{0};
HipErrInfoV1 g_last{};
}  // namespace

// Interposes the runtime's exported no-op, which it calls through the PLT.
extern "C" void __hipOnError(const void* err_info) {
  g_last = *static_cast<const HipErrInfoV1*>(err_info);
  g_calls++;
}

/**
 * Test Description
 * ------------------------
 *  - A failing API calls __hipOnError once with version 1, the error code, name and string.
 *  - Successful calls and hipErrorNotReady don't call it.
 * Test source
 * ------------------------
 *  - unit/errorHandling/hipOnError.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit___hipOnError_CalledOnFailure) {
  HIP_CHECK(hipFree(nullptr));
  (void)hipGetLastError();
  g_calls = 0;

  REQUIRE(hipMalloc(nullptr, 64) == hipErrorInvalidValue);
  REQUIRE(g_calls == 1);
  REQUIRE(g_last.version == 1);
  REQUIRE(g_last.err_no == hipErrorInvalidValue);
  REQUIRE(std::string(g_last.err_name) == hipGetErrorName(hipErrorInvalidValue));
  REQUIRE(std::string(g_last.err_string) == hipGetErrorString(hipErrorInvalidValue));
  (void)hipGetLastError();

  void* ptr = nullptr;
  HIP_CHECK(hipMalloc(&ptr, 64));
  HIP_CHECK(hipFree(ptr));
  REQUIRE(g_calls == 1);

  // A busy stream reports hipErrorNotReady, which isn't a failure.
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  LaunchDelayKernel(std::chrono::milliseconds(200), stream);
  REQUIRE(hipStreamQuery(stream) == hipErrorNotReady);
  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipStreamDestroy(stream));
  REQUIRE(g_calls == 1);
}

/**
 * End doxygen group ErrorHandlingTest.
 * @}
 */
