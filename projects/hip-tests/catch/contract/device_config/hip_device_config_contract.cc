/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>

namespace {
// Skips the test when no device is visible so that the device configuration and
// limit contracts are only exercised against a provisioned runtime.
void RequireDevice() {
  int device_count = 0;
  HIP_CHECK(hipGetDeviceCount(&device_count));
  if (device_count <= 0) {
    HIP_SKIP_TEST(HipTest::SkipReason::kNoGpuDevice);
  }
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
}  // namespace

HIP_TEST_CASE(Contract_DeviceConfig_GetCacheConfig_ReturnsEnumValue) {
  RequireDevice();

  hipFuncCache_t config = hipFuncCachePreferNone;
  HIP_CHECK(hipDeviceGetCacheConfig(&config));

  // The queried preference must be a documented enumerator; the specific value
  // is device- and backend-dependent and therefore not asserted.
  REQUIRE(IsKnownCacheConfig(config));
}

HIP_TEST_CASE(Contract_DeviceConfig_SetCacheConfig_IsAcceptedOrUnsupported) {
  RequireDevice();

  // Save the current preference so the device configuration is restored even if
  // the set path is honored, avoiding cross-test contamination.
  hipFuncCache_t saved = hipFuncCachePreferNone;
  HIP_CHECK(hipDeviceGetCacheConfig(&saved));

  const hipError_t status = hipDeviceSetCacheConfig(hipFuncCachePreferNone);
  if (status == hipErrorNotSupported) {
    // Setting the cache preference is an optional capability; an unsupported
    // report is a contract-compliant outcome, not a failure.
    return;
  }
  HIP_CHECK(status);

  // The runtime may coerce the request; the exact post-state is not part of the
  // contract, so only the saved preference is restored.
  HIP_CHECK(hipDeviceSetCacheConfig(saved));
}

HIP_TEST_CASE(Contract_DeviceConfig_GetSharedMemConfig_ReturnsEnumValue) {
  RequireDevice();

  hipSharedMemConfig config = hipSharedMemBankSizeDefault;
  const hipError_t status = hipDeviceGetSharedMemConfig(&config);
  if (status == hipErrorNotSupported) {
    // Shared-memory bank configuration is an optional capability on some
    // backends; an unsupported report is contract-compliant.
    return;
  }
  HIP_CHECK(status);

  // When supported, the reported value must be a documented enumerator.
  REQUIRE(IsKnownSharedMemConfig(config));
}

HIP_TEST_CASE(Contract_DeviceConfig_GetLimit_ReportsStackAndHeapAndRejectsInvalid) {
  RequireDevice();

  // Stack size and malloc heap size are the two portably queryable limits; each
  // either resolves to a well-defined size_t or is reported unsupported.
  for (const hipLimit_t limit : {hipLimitStackSize, hipLimitMallocHeapSize}) {
    size_t value = 0;
    const hipError_t status = hipDeviceGetLimit(&value, limit);
    if (status == hipErrorUnsupportedLimit) {
      continue;
    }
    HIP_CHECK(status);
    // value is a well-defined size_t; its magnitude is device-dependent and is
    // therefore not asserted against any specific bound.
    (void)value;
  }

  // An out-of-range hipLimit_t must not succeed. Backends may report the
  // specific hipErrorUnsupportedLimit or another non-success error; the
  // contract only requires that the query does not silently succeed.
  size_t invalid_value = 0;
  const hipError_t invalid_status =
      hipDeviceGetLimit(&invalid_value, static_cast<hipLimit_t>(hipLimitRange));
  REQUIRE(invalid_status != hipSuccess);
}

HIP_TEST_CASE(Contract_DeviceConfig_SetLimit_RoundTripsOrIsUnsupported) {
  RequireDevice();

  // Query the current heap-size limit first; if the limit itself is
  // unsupported, there is nothing to round-trip and the test skips.
  size_t current = 0;
  const hipError_t get_status = hipDeviceGetLimit(&current, hipLimitMallocHeapSize);
  if (get_status == hipErrorUnsupportedLimit) {
    return;
  }
  HIP_CHECK(get_status);

  // Setting the limit back to its current value performs no global mutation and
  // must either succeed or report the limit as unsupported for set.
  const hipError_t set_status = hipDeviceSetLimit(hipLimitMallocHeapSize, current);
  if (set_status == hipErrorUnsupportedLimit) {
    return;
  }
  HIP_CHECK(set_status);
}

HIP_TEST_CASE(Contract_DeviceConfig_GetDeviceFlagsAndStreamPriorityRange_AreConsistent) {
  RequireDevice();

  unsigned int flags = 0;
  HIP_CHECK(hipGetDeviceFlags(&flags));

  // The device flags word is composed of the documented schedule mask plus the
  // map-host and lmem-resize bits; no undocumented bits should be reported.
  const unsigned int documented_flags =
      hipDeviceScheduleMask | hipDeviceMapHost | hipDeviceLmemResizeToMax;
  REQUIRE((flags & ~documented_flags) == 0u);

  int least_priority = 0;
  int greatest_priority = 0;
  HIP_CHECK(hipDeviceGetStreamPriorityRange(&least_priority, &greatest_priority));

  // HIP orders stream priorities so that the greatest (highest) priority is a
  // numerically smaller-or-equal value than the least priority.
  REQUIRE(greatest_priority <= least_priority);
}
