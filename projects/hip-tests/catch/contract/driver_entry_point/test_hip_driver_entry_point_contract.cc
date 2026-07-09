/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>

namespace {
void RequireDevice() {
  int device_count = 0;
  HIP_CHECK(hipGetDeviceCount(&device_count));
  if (device_count <= 0) {
    HIP_SKIP_TEST(HipTest::SkipReason::kNoGpuDevice);
  }
}
}  // namespace

HIP_TEST_CASE(Contract_DriverEntryPoint_ResolvesKnownSymbol_IsCallable) {
  RequireDevice();

  void* function_ptr = nullptr;
  hipDriverEntryPointQueryResult query_status = hipDriverEntryPointSymbolNotFound;
  HIP_CHECK(hipGetDriverEntryPoint("hipGetDeviceCount", &function_ptr, hipEnableDefault,
                                   &query_status));

  REQUIRE(function_ptr != nullptr);
  REQUIRE(query_status == hipDriverEntryPointSuccess);

  auto resolved = reinterpret_cast<hipError_t (*)(int*)>(function_ptr);
  int resolved_count = -1;
  HIP_CHECK(resolved(&resolved_count));

  int direct_count = -1;
  HIP_CHECK(hipGetDeviceCount(&direct_count));
  REQUIRE(resolved_count == direct_count);
}

HIP_TEST_CASE(Contract_DriverEntryPoint_UnknownSymbol_ReportsNotFound) {
  RequireDevice();

  void* function_ptr = reinterpret_cast<void*>(0x1);
  hipDriverEntryPointQueryResult query_status = hipDriverEntryPointSuccess;
  const hipError_t error = hipGetDriverEntryPoint("hipThisSymbolDoesNotExist", &function_ptr,
                                                  hipEnableDefault, &query_status);

  const bool reported_not_found = (error != hipSuccess) ||
                                  (query_status == hipDriverEntryPointSymbolNotFound) ||
                                  (function_ptr == nullptr);
  REQUIRE(reported_not_found);

  const bool resolved_successfully = (error == hipSuccess) &&
                                     (query_status == hipDriverEntryPointSuccess) &&
                                     (function_ptr != nullptr);
  REQUIRE_FALSE(resolved_successfully);
}

HIP_TEST_CASE(Contract_DriverEntryPoint_EmptySymbol_IsRejected) {
  RequireDevice();

  void* function_ptr = nullptr;
  hipDriverEntryPointQueryResult query_status = hipDriverEntryPointSuccess;
  REQUIRE(hipGetDriverEntryPoint("", &function_ptr, hipEnableDefault, &query_status) != hipSuccess);
}

HIP_TEST_CASE(Contract_DriverEntryPoint_NullFuncPtr_IsRejected) {
  RequireDevice();

  hipDriverEntryPointQueryResult query_status = hipDriverEntryPointSuccess;
  HIP_CHECK_ERROR(hipGetDriverEntryPoint("hipGetDeviceCount", nullptr, hipEnableDefault,
                                         &query_status),
                  hipErrorInvalidValue);
}

HIP_TEST_CASE(Contract_DriverEntryPoint_InvalidFlag_IsRejected) {
  RequireDevice();

  void* function_ptr = nullptr;
  hipDriverEntryPointQueryResult query_status = hipDriverEntryPointSuccess;
  REQUIRE(hipGetDriverEntryPoint("hipGetDeviceCount", &function_ptr, ~0ull, &query_status) !=
          hipSuccess);
}
