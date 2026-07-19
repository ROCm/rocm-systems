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

// @asserts: hipGetProcAddress - resolves a known runtime symbol to a non-null, callable pointer with a success status
HIP_TEST_CASE(Contract_Extension_GetProcAddress_ResolvesKnownSymbol) {
  RequireDevice();

  const int hip_version = RuntimeQueryVersion();

  void* pfn = nullptr;
  // The HIP_GET_PROC_ADDRESS_* status macros expand to plain HIP enumerators on
  // AMD but to CUDA CUdriverProcAddressQueryResult enumerators on the NVIDIA
  // backend, which cannot initialize a hipDriverProcAddressQueryResult variable
  // directly. static_cast bridges both (an identity conversion on AMD).
  hipDriverProcAddressQueryResult symbol_status =
      static_cast<hipDriverProcAddressQueryResult>(HIP_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND);
  // BACKEND-DIFF: hipGetProcAddress resolves different symbol namespaces per
  // backend. On AMD it resolves runtime-level hip* symbols by name; on NVIDIA it
  // forwards to cuGetProcAddress, which resolves driver-level cu* symbols (a hip*
  // name returns a null pfn), so the two branches query different symbols and the
  // NVIDIA branch cannot call the result through a hip* prototype. Parity would
  // require the NVIDIA HIP layer to resolve hip*/cuda* runtime symbols by name.
#if HT_AMD
  // On AMD hipGetProcAddress resolves runtime-level hip* symbols by name.
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
#else
  // On the NVIDIA backend hipGetProcAddress forwards to cuGetProcAddress, which
  // resolves driver-level cu* symbols by name (a runtime "hipGetDevice" name
  // returns a null pfn). Query the driver entry point instead and assert only
  // the resolution contract; the resolved pointer is a driver function with a
  // different signature, so it is not invoked through a hip* prototype here.
  //
  // cuGetProcAddress version-gates each symbol against the CUDA version it was
  // introduced in, so it must be passed the real CUDA_VERSION. The AMD-style
  // 100*major+minor form that RuntimeQueryVersion() derives collapses to 0 for a
  // CUDA runtime version (e.g. 13010), which cuGetProcAddress rejects as
  // version-not-sufficient and returns a null pfn.
  static_cast<void>(hip_version);
  HIP_CHECK(hipGetProcAddress("cuCtxGetDevice", &pfn, CUDA_VERSION,
                              HIP_GET_PROC_ADDRESS_DEFAULT, &symbol_status));

  REQUIRE(pfn != nullptr);
  REQUIRE(symbol_status ==
          static_cast<hipDriverProcAddressQueryResult>(HIP_GET_PROC_ADDRESS_SUCCESS));
#endif
}

// @asserts: hipGetProcAddress - an unknown symbol reports not-found and never yields a success+callable-pointer result
HIP_TEST_CASE(Contract_Extension_GetProcAddress_UnknownSymbol_ReportsNotFound) {
  RequireDevice();

  const int hip_version = RuntimeQueryVersion();

  void* pfn = nullptr;
  hipDriverProcAddressQueryResult symbol_status =
      static_cast<hipDriverProcAddressQueryResult>(HIP_GET_PROC_ADDRESS_SUCCESS);
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

// @asserts: hipGetProcAddress - rejects a null symbol name or null output pointer with hipErrorInvalidValue (AMD)
HIP_TEST_CASE(Contract_Extension_GetProcAddress_NullArgs_AreRejected) {
  RequireDevice();

  const int hip_version = RuntimeQueryVersion();

  // BACKEND-DIFF: The null-argument rejection contract is only exercised on AMD.
  // On the NVIDIA backend hipGetProcAddress forwards to cuGetProcAddress, which
  // does not validate its arguments and dereferences a null symbol name / output
  // pointer, faulting (SIGSEGV) instead of returning hipErrorInvalidValue, so the
  // rejection cannot be evaluated safely there. Parity would require matching
  // null-argument validation before the dereference.
#if HT_AMD
  // A null symbol name is invalid input.
  void* pfn = nullptr;
  HIP_CHECK_ERROR(hipGetProcAddress(nullptr, &pfn, hip_version,
                                    HIP_GET_PROC_ADDRESS_DEFAULT, nullptr),
                  hipErrorInvalidValue);

  // A null output pointer is invalid input.
  HIP_CHECK_ERROR(hipGetProcAddress("hipRuntimeGetVersion", nullptr, hip_version,
                                    HIP_GET_PROC_ADDRESS_DEFAULT, nullptr),
                  hipErrorInvalidValue);
#else
  HIP_SKIP_TEST("hipGetProcAddress forwards to cuGetProcAddress, which does not validate "
                "null arguments on the NVIDIA backend; the rejection contract cannot be "
                "exercised safely.");
#endif
}

// BACKEND-DIFF: the following AMD extension APIs (hipApiName,
// hipGetStreamDeviceId, hipExtGetLastError) have no NVIDIA-backend equivalent, so
// these contracts build only on AMD. The portable hipGetProcAddress contracts
// above run on both backends. Parity would require NVIDIA-side equivalents.
#if HT_AMD
// @asserts: hipApiName - maps API id 0 to a non-null, non-empty NUL-terminated name string
HIP_TEST_CASE(Contract_Extension_ApiName_ReturnsNonEmptyString) {
  RequireDevice();

  // hipApiName maps a callback/activity API id to its name. Id 0 is always a
  // valid entry in the activity table, so it must yield a non-empty,
  // NUL-terminated string; the exact text is backend-specific.
  const char* name = hipApiName(0);
  REQUIRE(name != nullptr);
  REQUIRE(std::strlen(name) > 0);
}

// @asserts: hipGetStreamDeviceId - the null/default stream reports the device id of the current active device
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

// @asserts: hipExtGetLastError - reports the stored thread error then resets the thread error state to hipSuccess
HIP_TEST_CASE(Contract_Extension_ExtGetLastError_TracksErrorState) {
  RequireDevice();

  // Clear any residual error from prior runtime calls in this thread. This must
  // discard, not assert on, the residual: a sibling test running earlier in the
  // process can leave a sticky thread-local error (the tests share one thread),
  // and clearing it is the whole point here, so HIP_CHECK would wrongly fail on
  // that leaked state instead of resetting it.
  (void)hipGetLastError();

  // Provoke a deterministic error through a public API.
  const hipError_t error = hipMalloc(nullptr, 1);
  REQUIRE(error != hipSuccess);

  // hipExtGetLastError must report the stored error and then reset the thread
  // error state to hipSuccess, mirroring hipGetLastError semantics.
  HIP_CHECK_ERROR(hipExtGetLastError(), error);
  HIP_CHECK(hipExtGetLastError());
}
#endif  // HT_AMD
