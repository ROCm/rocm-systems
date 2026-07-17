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

// hipGetDriverEntryPoint resolves names in the driver-symbol namespace of the
// active backend: HIP runtime names on AMD, but the underlying CUDA driver names
// on NVIDIA (where it maps to cudaGetDriverEntryPoint). Query a name each backend
// can actually resolve. cuDeviceGetCount is the driver-API equivalent of the
// hipGetDeviceCount entry point resolved on AMD, and has the same int* signature.
#ifdef __HIP_PLATFORM_AMD__
constexpr char const kKnownSymbol[] = "hipGetDeviceCount";
#else
constexpr char const kKnownSymbol[] = "cuDeviceGetCount";
#endif
}  // namespace

HIP_TEST_CASE(Contract_DriverEntryPoint_ResolvesKnownSymbol_IsCallable) {
  RequireDevice();

  void* function_ptr = nullptr;
  hipDriverEntryPointQueryResult query_status = hipDriverEntryPointSymbolNotFound;
  HIP_CHECK(hipGetDriverEntryPoint(kKnownSymbol, &function_ptr, hipEnableDefault, &query_status));

  REQUIRE(function_ptr != nullptr);
  REQUIRE(query_status == hipDriverEntryPointSuccess);

  // The resolved entry point takes an int* out-count and returns 0 on success on
  // both backends (hipGetDeviceCount / cuDeviceGetCount), so it can be called
  // through a common signature and its result compared against the direct query.
  auto resolved = reinterpret_cast<int (*)(int*)>(function_ptr);
  int resolved_count = -1;
  REQUIRE(resolved(&resolved_count) == 0);

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
  const hipError_t error =
      hipGetDriverEntryPoint("", &function_ptr, hipEnableDefault, &query_status);

  // An empty symbol must not resolve to a usable pointer. Backends report this
  // differently: the AMD runtime returns a non-success status, while NVIDIA
  // (cudaGetDriverEntryPoint) returns success but signals the failure through the
  // query result (symbol-not-found) and a null pointer. Any of these is a valid
  // rejection; what must never happen is a successful resolution to a non-null
  // entry point.
  const bool resolved_successfully = (error == hipSuccess) &&
                                     (query_status == hipDriverEntryPointSuccess) &&
                                     (function_ptr != nullptr);
  REQUIRE_FALSE(resolved_successfully);
  (void)hipGetLastError();
}

HIP_TEST_CASE(Contract_DriverEntryPoint_NullFuncPtr_IsRejected) {
  RequireDevice();

  // A null output pointer is a caller error that must be rejected with
  // hipErrorInvalidValue. This is only exercised on AMD: the NVIDIA path
  // (cudaGetDriverEntryPoint) does not null-check the output pointer and
  // dereferences it, faulting instead of returning a defined error, so the
  // contract cannot be safely evaluated there.
#ifdef __HIP_PLATFORM_AMD__
  hipDriverEntryPointQueryResult query_status = hipDriverEntryPointSuccess;
  HIP_CHECK_ERROR(hipGetDriverEntryPoint(kKnownSymbol, nullptr, hipEnableDefault, &query_status),
                  hipErrorInvalidValue);
#else
  HIP_SKIP_TEST("hipGetDriverEntryPoint does not null-check the output pointer on the NVIDIA "
                "backend; the null-pointer rejection contract cannot be exercised safely.");
#endif
}

HIP_TEST_CASE(Contract_DriverEntryPoint_InvalidFlag_IsRejected) {
  RequireDevice();

  void* function_ptr = nullptr;
  hipDriverEntryPointQueryResult query_status = hipDriverEntryPointSuccess;
  REQUIRE(hipGetDriverEntryPoint(kKnownSymbol, &function_ptr, ~0ull, &query_status) != hipSuccess);
}
