/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <resource_guards.hh>

/**
 * @addtogroup hipKernelNameRefByPtr hipKernelNameRefByPtr
 * @{
 * @ingroup CallbackTest
 * `hipKernelNameRefByPtr(const void* hostFunction, hipStream_t stream)` -
 * returns the name of passed function pointer on desired stream
 */

__global__ void test_kernel() { return; }

/**
 * Test Description
 * ------------------------
 *  - Creates new stream and a function pointer
 *  - Verifies that valid API name is returned
 * Test source
 * ------------------------
 *  - unit/callback/hipKernelNameRefByPtr.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 *  - Platform specific (AMD)
 */
TEST_CASE(Unit_hipKernelNameRefByPtr_Positive_Basic) {
  const void* kernel_ptr{reinterpret_cast<const void*>(&test_kernel)};
  StreamGuard stream_guard{Streams::created};

  const char* kernel_name{hipKernelNameRefByPtr(kernel_ptr, stream_guard.stream())};
  REQUIRE(kernel_name != nullptr);
  REQUIRE(std::strlen(kernel_name) > 0);
}

/**
 * Test Description
 * ------------------------
 *  - Passes `nullptr` stream while function pointer is valid
 *  - Verifies that the returned value is not `nullptr`
 * Test source
 * ------------------------
 *  - unit/callback/hipKernelNameRefByPtr.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 *  - Platform specific (AMD)
 */
TEST_CASE(Unit_hipKernelNameRefByPtr_Positive_StreamNullptr) {
  const void* kernel_ptr{reinterpret_cast<const void*>(&test_kernel)};
  StreamGuard stream_guard{Streams::nullstream};

  const char* kernel_name{hipKernelNameRefByPtr(kernel_ptr, stream_guard.stream())};
  REQUIRE(kernel_name != nullptr);
  REQUIRE(std::strlen(kernel_name) > 0);
}

/**
 * Test Description
 * ------------------------
 *  - Performs validation when the function pointer is `nullptr`
 *    -# When stream is `nullptr`
 *      - Expected output: return `nullptr`
 *    -# When stream is valid
 *      - Expected output: return `nullptr`
 * Test source
 * ------------------------
 *  - unit/callback/hipKernelNameRefByPtr.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 *  - Platform specific (AMD)
 */
TEST_CASE(Unit_hipKernelNameRefByPtr_Negative_KernelNullptr) {
  const void* kernel_ptr{nullptr};

  SECTION("stream is nullptr") {
    StreamGuard stream_guard{Streams::nullstream};
    REQUIRE(hipKernelNameRefByPtr(kernel_ptr, stream_guard.stream()) == nullptr);
  }

  SECTION("stream is created") {
    StreamGuard stream_guard{Streams::created};
    REQUIRE(hipKernelNameRefByPtr(kernel_ptr, stream_guard.stream()) == nullptr);
  }
}

/**
 * End doxygen group CallbackTest.
 * @}
 */
