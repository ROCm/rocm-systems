/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>

namespace {
// Requires at least one visible device, skipping the test when none is present
// so that driver-style context configuration is only exercised against a real
// ordinal.
void RequireDevice() {
  int device_count = 0;
  HIP_CHECK(hipGetDeviceCount(&device_count));
  if (device_count <= 0) {
    HIP_SKIP_TEST(HipTest::SkipReason::kNoGpuDevice);
  }
}

// Resolves the driver-style handle for ordinal zero, which every context
// configuration contract builds on.
hipDevice_t DeviceForOrdinalZero() {
  hipDevice_t device = 0;
  HIP_CHECK(hipDeviceGet(&device, 0));
  return device;
}

// Reports whether a cache-config value is one of the four documented
// hipFuncCache_t enumerators; exact selection is backend-specific.
bool IsKnownCacheConfig(hipFuncCache_t config) {
  return config == hipFuncCachePreferNone || config == hipFuncCachePreferShared ||
         config == hipFuncCachePreferL1 || config == hipFuncCachePreferEqual;
}

// Reports whether a shared-memory bank-size value is one of the three
// documented hipSharedMemConfig enumerators.
bool IsKnownSharedMemConfig(hipSharedMemConfig config) {
  return config == hipSharedMemBankSizeDefault || config == hipSharedMemBankSizeFourByte ||
         config == hipSharedMemBankSizeEightByte;
}

// Saves the thread-current driver-style context on construction and restores it
// on destruction so tests that mutate the current context do not leak that state
// into later tests when several run in one process. The destructor cannot use
// Catch assertions, so it ignores the restore status; tests still perform an
// explicit, asserted restore before scope exit so failures on the restore path
// stay visible.
class ScopedCurrentContext {
 public:
  ScopedCurrentContext() { HIP_CHECK(hipCtxGetCurrent(&previous_)); }
  ~ScopedCurrentContext() { static_cast<void>(hipCtxSetCurrent(previous_)); }

  ScopedCurrentContext(const ScopedCurrentContext&) = delete;
  ScopedCurrentContext& operator=(const ScopedCurrentContext&) = delete;

  hipCtx_t previous() const { return previous_; }

 private:
  hipCtx_t previous_ = nullptr;
};
}  // namespace

HIP_TEST_CASE(Contract_ContextConfig_GetCacheConfig_ReturnsEnumOrNotSupported) {
  RequireDevice();

  hipFuncCache_t config = hipFuncCachePreferNone;
  const hipError_t status = hipCtxGetCacheConfig(&config);
  if (status == hipErrorNotSupported) {
    // Driver-style cache-config query is an optional capability on some
    // backends; an unsupported report is contract-compliant.
    return;
  }
  HIP_CHECK(status);

  // When supported, the queried preference must be a documented enumerator; the
  // specific value is device- and backend-dependent and therefore not asserted.
  REQUIRE(IsKnownCacheConfig(config));
}

HIP_TEST_CASE(Contract_ContextConfig_SetCacheConfig_IsAcceptedOrUnsupported_RejectsInvalid) {
  RequireDevice();

  // Save the current preference so the context configuration is restored even if
  // the set path is honored, avoiding cross-test contamination. The save is only
  // attempted when the get path is supported.
  hipFuncCache_t saved = hipFuncCachePreferNone;
  bool have_saved = false;
  const hipError_t get_status = hipCtxGetCacheConfig(&saved);
  if (get_status == hipSuccess) {
    have_saved = true;
  } else if (get_status != hipErrorNotSupported) {
    HIP_CHECK(get_status);
  }

  const hipError_t set_status = hipCtxSetCacheConfig(hipFuncCachePreferNone);
  if (set_status != hipErrorNotSupported) {
    HIP_CHECK(set_status);
    // The runtime may coerce the request; the exact post-state is not part of
    // the contract, so only the saved preference is restored when known.
    if (have_saved) {
      HIP_CHECK(hipCtxSetCacheConfig(saved));
    }
  }

  // An out-of-range enumerator must not be silently accepted. Backends may map
  // this to a specific invalid-value error or to a general non-success code; the
  // contract only requires that the query does not succeed.
  const hipError_t invalid_status =
      hipCtxSetCacheConfig(static_cast<hipFuncCache_t>(0x100));
  REQUIRE(invalid_status != hipSuccess);
}

HIP_TEST_CASE(Contract_ContextConfig_GetSharedMemConfig_ReturnsEnumOrNotSupported) {
  RequireDevice();

  hipSharedMemConfig config = hipSharedMemBankSizeDefault;
  const hipError_t status = hipCtxGetSharedMemConfig(&config);
  if (status == hipErrorNotSupported) {
    // Driver-style shared-memory bank configuration is an optional capability on
    // some backends; an unsupported report is contract-compliant. AMD may report
    // success here, which is equally compliant and is not assumed to be
    // unsupported.
    return;
  }
  HIP_CHECK(status);

  // When supported, the reported value must be a documented enumerator.
  REQUIRE(IsKnownSharedMemConfig(config));
}

HIP_TEST_CASE(Contract_ContextConfig_SetSharedMemConfig_IsAcceptedOrUnsupported) {
  RequireDevice();

  // Save the current bank size so the context configuration is restored even if
  // the set path is honored. The save is only attempted when the get path is
  // supported.
  hipSharedMemConfig saved = hipSharedMemBankSizeDefault;
  bool have_saved = false;
  const hipError_t get_status = hipCtxGetSharedMemConfig(&saved);
  if (get_status == hipSuccess) {
    have_saved = true;
  } else if (get_status != hipErrorNotSupported) {
    HIP_CHECK(get_status);
  }

  const hipError_t set_status = hipCtxSetSharedMemConfig(hipSharedMemBankSizeDefault);
  if (set_status != hipErrorNotSupported) {
    HIP_CHECK(set_status);
    // The runtime may coerce the request; only the saved bank size is restored
    // when it is known.
    if (have_saved) {
      HIP_CHECK(hipCtxSetSharedMemConfig(saved));
    }
  }
}

HIP_TEST_CASE(Contract_ContextConfig_GetFlags_ReturnsScheduleOrNotSupported) {
  RequireDevice();

  unsigned int flags = 0;
  const hipError_t status = hipCtxGetFlags(&flags);
  if (status == hipErrorNotSupported) {
    // Driver-style context flag query is an optional capability on some
    // backends; an unsupported report is contract-compliant.
    return;
  }
  HIP_CHECK(status);

  // The schedule subfield must resolve to one of the documented scheduling
  // modes. Backends may report additional device-specific flag bits, so only the
  // schedule subfield is constrained rather than rejecting the whole flags word.
  const unsigned int schedule = flags & hipDeviceScheduleMask;
  REQUIRE((schedule == hipDeviceScheduleAuto || schedule == hipDeviceScheduleSpin ||
           schedule == hipDeviceScheduleYield || schedule == hipDeviceScheduleBlockingSync));
}

HIP_TEST_CASE(Contract_ContextConfig_PeerAccessSelf_IsRejectedOrUnsupportedOrNoOp) {
  RequireDevice();

  const hipDevice_t device = DeviceForOrdinalZero();
  const ScopedCurrentContext scoped_context;

  hipCtx_t context = nullptr;
  HIP_CHECK(hipCtxCreate(&context, 0, device));
  REQUIRE(context != nullptr);

  hipCtx_t current = nullptr;
  HIP_CHECK(hipCtxGetCurrent(&current));
  REQUIRE(current != nullptr);

  // Enabling peer access from a context to itself is not a real cross-device
  // mapping. Backends legitimately differ here: the request may succeed, report
  // that peer access is already enabled, reject the self/context/value as
  // invalid, or report the API unsupported. Any of these documented, portable
  // outcomes is contract-compliant; a real multi-device mapping is deliberately
  // avoided.
  const hipError_t enable_status = hipCtxEnablePeerAccess(current, 0);
  REQUIRE((enable_status == hipSuccess ||
           enable_status == hipErrorPeerAccessAlreadyEnabled ||
           enable_status == hipErrorInvalidDevice ||
           enable_status == hipErrorInvalidValue ||
           enable_status == hipErrorInvalidContext ||
           enable_status == hipErrorNotSupported));

  if (enable_status == hipSuccess) {
    // If enable was honored, tearing the same mapping down must land in the
    // matching portable outcome set rather than leaking peer-access state.
    const hipError_t disable_status = hipCtxDisablePeerAccess(current);
    REQUIRE((disable_status == hipSuccess ||
             disable_status == hipErrorPeerAccessNotEnabled ||
             disable_status == hipErrorInvalidValue ||
             disable_status == hipErrorInvalidContext ||
             disable_status == hipErrorNotSupported));
  }

  // Restore the previously current context explicitly before destroying the
  // context created here, so the restore is an asserted step rather than relying
  // solely on the RAII guard's silent restore.
  HIP_CHECK(hipCtxSetCurrent(scoped_context.previous()));
  HIP_CHECK(hipCtxDestroy(context));
}
