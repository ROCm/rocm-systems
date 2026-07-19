/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <vector>

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>
#include <contract_cleanup.hh>

namespace {
// Accepts a status that is either success or one of the documented
// "cannot apply right now" outcomes for the deprecated device/primary-context
// configuration entry points. Any other status is a contract violation.
void RequireAcceptedOrBenign(hipError_t status) {
  if (status == hipSuccess || status == hipErrorNotSupported ||
      status == hipErrorContextAlreadyInUse ||
      status == hipErrorSetOnActiveProcess) {
    return;
  }
  HIP_CHECK(status);
}

hipDevice_t DeviceForOrdinalZero() {
  // Prime a primary context before the driver-API query. On the NVIDIA backend
  // hipDeviceGet forwards to cuDeviceGet, which needs an initialized context;
  // when this is the first HIP call in the process (e.g. run in isolation under
  // ctest) no context exists yet and the call fails with an initialization/
  // invalid-context error. hipFree(0) is the canonical no-op that establishes
  // it. On AMD this is a harmless success.
  HIP_CHECK(hipFree(0));
  hipDevice_t device = 0;
  HIP_CHECK(hipDeviceGet(&device, 0));
  return device;
}
}  // namespace

// @asserts: hipSetDeviceFlags - writing back the currently-active device flags is accepted or reported not-settable
HIP_TEST_CASE(Contract_DeviceLifecycle_SetDeviceFlags_AcceptsCurrentFlags) {
  // Reading the active device flags and setting the very same value back must be
  // accepted (or reported as not-settable on an already-active process). Writing
  // the current value keeps global device state unchanged for sibling tests.
  unsigned int flags = 0;
  HIP_CHECK(hipGetDeviceFlags(&flags));

  RequireAcceptedOrBenign(hipSetDeviceFlags(flags));
}

// @asserts: hipDeviceSetSharedMemConfig - setting back the currently-reported shared-mem config is accepted or unsupported
HIP_TEST_CASE(Contract_DeviceLifecycle_SetSharedMemConfig_RoundTripsCurrentConfig) {
  // The shared-memory bank configuration setter (deprecated) must accept the
  // configuration the device currently reports. Setting the current value back
  // leaves the device configuration unchanged.
  hipSharedMemConfig config = hipSharedMemBankSizeDefault;
  const hipError_t get_status = hipDeviceGetSharedMemConfig(&config);
  if (get_status == hipErrorNotSupported) {
    HIP_SKIP_TEST("Shared-memory bank configuration is not supported by this runtime path.");
  }
  HIP_CHECK(get_status);

  RequireAcceptedOrBenign(hipDeviceSetSharedMemConfig(config));
}

// @asserts: hipSetValidDevices - presenting the full set of visible device ordinals is accepted or reported unsupported
HIP_TEST_CASE(Contract_DeviceLifecycle_SetValidDevices_AcceptsFullDeviceList) {
  int count = 0;
  HIP_CHECK(hipGetDeviceCount(&count));
  REQUIRE(count > 0);

  // Presenting the complete set of visible device ordinals to the deprecated
  // valid-device selector must be accepted (or reported unsupported). The full
  // list does not remove any device from later tests in this process.
  std::vector<int> devices(static_cast<size_t>(count));
  for (int i = 0; i < count; ++i) {
    devices[static_cast<size_t>(i)] = i;
  }

  RequireAcceptedOrBenign(hipSetValidDevices(devices.data(), count));
}

// @asserts: hipDevicePrimaryCtxSetFlags - setting primary-context flags is accepted or reports context-already-in-use
HIP_TEST_CASE(Contract_DeviceLifecycle_PrimaryCtxSetFlags_IsAcceptedOrInUse) {
  const hipDevice_t device = DeviceForOrdinalZero();

  // Setting primary-context flags (deprecated) is only permitted while the
  // primary context is not already in use; once the runtime has activated it,
  // the call reports hipErrorContextAlreadyInUse. Either outcome satisfies the
  // contract; an unrelated failure does not.
  RequireAcceptedOrBenign(hipDevicePrimaryCtxSetFlags(device, hipDeviceScheduleAuto));
}

// @asserts: hipDevicePrimaryCtxReset - after resetting the primary context the device still serves fresh allocations
HIP_TEST_CASE(Contract_DeviceLifecycle_PrimaryCtxReset_LeavesDeviceUsable) {
  const hipDevice_t device = DeviceForOrdinalZero();

  // Resetting the primary context (deprecated) destroys allocations on it, but
  // the device must remain usable afterward: the runtime re-establishes a
  // primary context on the next allocation. A tiny alloc/free round-trip proves
  // the device is not left in a broken state.
  const hipError_t reset_status = hipDevicePrimaryCtxReset(device);
  if (reset_status == hipErrorNotSupported) {
    HIP_SKIP_TEST("Primary-context reset is not supported by this runtime path.");
  }
  HIP_CHECK(reset_status);

  hip::contract::ContractCleanup cleanup;
  void* ptr = nullptr;
  HIP_CHECK(hipMalloc(&ptr, 64));
  cleanup.Add([ptr] { (void)hipFree(ptr); });
  REQUIRE(ptr != nullptr);
}
