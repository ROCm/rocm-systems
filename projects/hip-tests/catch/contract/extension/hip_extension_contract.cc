/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <cstring>
#include <string.h>

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>

namespace {
// Skips the test when no device is visible so that the extension and
// proc-address contracts are only exercised against a provisioned runtime.
void RequireDevice() {
  int device_count = 0;
  HIP_CHECK(hipGetDeviceCount(&device_count));
  if (device_count <= 0) {
    HIP_SKIP_TEST(HipTest::SkipReason::kNoGpuDevice);
  }
}

// Derives the queryable HIP version in the documented 100*major+minor form
// from the live runtime rather than hardcoding a release number, so the
// proc-address contracts track whatever runtime the test is linked against.
int RuntimeQueryVersion() {
  int runtime_version = 0;
  HIP_CHECK(hipRuntimeGetVersion(&runtime_version));
  REQUIRE(runtime_version > 0);

  const int major = runtime_version / 10000000;
  const int minor = (runtime_version / 100000) % 100;
  return 100 * major + minor;
}
}  // namespace

HIP_TEST_CASE(Contract_Extension_GetProcAddress_ResolvesKnownSymbol) {
  RequireDevice();

  const int hip_version = RuntimeQueryVersion();

  void* pfn = nullptr;
  hipDriverProcAddressQueryResult symbol_status = HIP_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
  HIP_CHECK(hipGetProcAddress("hipRuntimeGetVersion", &pfn, hip_version,
                              HIP_GET_PROC_ADDRESS_DEFAULT, &symbol_status));

  // A symbol that exists in the linked runtime must resolve to a non-null
  // function pointer and report a found status.
  REQUIRE(pfn != nullptr);
  REQUIRE(symbol_status == HIP_GET_PROC_ADDRESS_SUCCESS);

  // The resolved pointer must be callable through the public signature and
  // return the same runtime version the runtime reports directly.
  auto resolved = reinterpret_cast<hipError_t (*)(int*)>(pfn);
  int resolved_version = 0;
  HIP_CHECK(resolved(&resolved_version));

  int direct_version = 0;
  HIP_CHECK(hipRuntimeGetVersion(&direct_version));
  REQUIRE(resolved_version == direct_version);
}

HIP_TEST_CASE(Contract_Extension_GetProcAddress_UnknownSymbol_ReportsNotFound) {
  RequireDevice();

  const int hip_version = RuntimeQueryVersion();

  void* pfn = nullptr;
  hipDriverProcAddressQueryResult symbol_status = HIP_GET_PROC_ADDRESS_SUCCESS;
  const hipError_t error =
      hipGetProcAddress("hipThisSymbolDoesNotExist", &pfn, hip_version,
                        HIP_GET_PROC_ADDRESS_DEFAULT, &symbol_status);

  // An unknown symbol must not resolve. The runtime reports this through the
  // optional status enumeration; the returned error code is vendor-specific so
  // it is only required to be a non-success value.
  REQUIRE(error != hipSuccess);
  REQUIRE(pfn == nullptr);
  REQUIRE(symbol_status == HIP_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND);
}

HIP_TEST_CASE(Contract_Extension_GetProcAddress_NullArgs_AreRejected) {
  RequireDevice();

  const int hip_version = RuntimeQueryVersion();

  // A null symbol name is invalid input.
  void* pfn = nullptr;
  HIP_CHECK_ERROR(hipGetProcAddress(nullptr, &pfn, hip_version,
                                    HIP_GET_PROC_ADDRESS_DEFAULT, nullptr),
                  hipErrorInvalidValue);

  // A null output pointer is invalid input.
  HIP_CHECK_ERROR(hipGetProcAddress("hipRuntimeGetVersion", nullptr, hip_version,
                                    HIP_GET_PROC_ADDRESS_DEFAULT, nullptr),
                  hipErrorInvalidValue);
}

#if HT_AMD
HIP_TEST_CASE(Contract_Extension_ApiName_ReturnsNonEmptyString) {
  RequireDevice();

  // hipApiName maps a callback/activity API id to its name. Id 0 is always a
  // valid entry in the activity table, so it must yield a non-empty,
  // NUL-terminated string; the exact text is backend-specific.
  const char* name = hipApiName(0);
  REQUIRE(name != nullptr);
  REQUIRE(std::strlen(name) > 0);
}

HIP_TEST_CASE(Contract_Extension_GetStreamDeviceId_MatchesCurrentDevice) {
  RequireDevice();

  HIP_CHECK(hipSetDevice(0));

  int current_device = -1;
  HIP_CHECK(hipGetDevice(&current_device));

  // The null (default) stream belongs to the current device, so its reported
  // device id must match the active ordinal.
  const int stream_device = hipGetStreamDeviceId(nullptr);
  REQUIRE(stream_device == current_device);
}

HIP_TEST_CASE(Contract_Extension_ExtGetLastError_TracksErrorState) {
  RequireDevice();

  // Clear any residual error from prior runtime calls in this thread.
  HIP_CHECK(hipGetLastError());

  // Provoke a deterministic error through a public API.
  const hipError_t error = hipMalloc(nullptr, 1);
  REQUIRE(error != hipSuccess);

  // hipExtGetLastError must report the stored error and then reset the thread
  // error state to hipSuccess, mirroring hipGetLastError semantics.
  HIP_CHECK_ERROR(hipExtGetLastError(), error);
  HIP_CHECK(hipExtGetLastError());
}
#endif  // HT_AMD
