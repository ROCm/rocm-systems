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

// Saves the current device on construction, switches to a requested ordinal,
// and restores the original device on destruction so tests that force a specific
// device do not leak current-device state into later tests when several run in
// one process. The destructor cannot use Catch assertions, so it ignores the
// restore status; failures on the switch-in path still surface through
// HIP_CHECK.
class ScopedDevice {
 public:
  explicit ScopedDevice(int next) {
    HIP_CHECK(hipGetDevice(&previous_));
    HIP_CHECK(hipSetDevice(next));
  }
  ~ScopedDevice() { static_cast<void>(hipSetDevice(previous_)); }

  ScopedDevice(const ScopedDevice&) = delete;
  ScopedDevice& operator=(const ScopedDevice&) = delete;

 private:
  int previous_ = 0;
};

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
  HIP_CHECK(hipGetProcAddress("hipGetDevice", &pfn, hip_version,
                              HIP_GET_PROC_ADDRESS_DEFAULT, &symbol_status));

  // A symbol that exists in the linked runtime must resolve to a non-null
  // function pointer and report a found status.
  REQUIRE(pfn != nullptr);
  REQUIRE(symbol_status == HIP_GET_PROC_ADDRESS_SUCCESS);

  // The resolved pointer must be callable through the public signature and
  // report the same current device the runtime reports directly.
  auto resolved = reinterpret_cast<hipError_t (*)(int*)>(pfn);
  int resolved_device = -1;
  HIP_CHECK(resolved(&resolved_device));

  int direct_device = -1;
  HIP_CHECK(hipGetDevice(&direct_device));
  REQUIRE(resolved_device == direct_device);
}

HIP_TEST_CASE(Contract_Extension_GetProcAddress_UnknownSymbol_ReportsNotFound) {
  RequireDevice();

  const int hip_version = RuntimeQueryVersion();

  void* pfn = nullptr;
  hipDriverProcAddressQueryResult symbol_status = HIP_GET_PROC_ADDRESS_SUCCESS;
  const hipError_t error =
      hipGetProcAddress("hipThisSymbolDoesNotExist", &pfn, hip_version,
                        HIP_GET_PROC_ADDRESS_DEFAULT, &symbol_status);

  // An unknown symbol must not produce a valid callable lookup. Backends differ
  // in how they signal this, so the contract accepts any of the equivalent
  // not-found indications: a non-success return, the not-found status, or a null
  // function pointer.
  const bool reported_not_found = (error != hipSuccess) ||
                                  (symbol_status == HIP_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND) ||
                                  (pfn == nullptr);
  REQUIRE(reported_not_found);

  // Regardless of how the not-found state is signaled, the lookup must not
  // simultaneously claim success and hand back a callable pointer.
  const bool resolved_successfully =
      (error == hipSuccess) && (symbol_status == HIP_GET_PROC_ADDRESS_SUCCESS) &&
      (pfn != nullptr);
  REQUIRE_FALSE(resolved_successfully);
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

  const ScopedDevice scoped_device(0);

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
