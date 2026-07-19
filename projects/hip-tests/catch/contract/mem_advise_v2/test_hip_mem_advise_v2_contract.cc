/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <cstddef>

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>
#include <contract_cleanup.hh>

namespace {
constexpr size_t kRangeBytes = 4096;

int CurrentDevice() {
  int device = 0;
  HIP_CHECK(hipGetDevice(&device));
  return device;
}

hipMemLocation CurrentDeviceLocation() {
  hipMemLocation location{};
  location.type = hipMemLocationTypeDevice;
  location.id = CurrentDevice();
  return location;
}

bool ManagedMemorySupported() {
  void* ptr = nullptr;
  const hipError_t status = hipMallocManaged(&ptr, kRangeBytes, hipMemAttachGlobal);
  if (status == hipSuccess) {
    HIP_CHECK(hipFree(ptr));
    return true;
  }
  if (status == hipErrorNotSupported || status == hipErrorOutOfMemory) {
    return false;
  }
  HIP_CHECK(status);
  return false;
}

void SkipIfManagedMemoryUnsupported() {
  if (!ManagedMemorySupported()) {
    HIP_SKIP_TEST(HipTest::SkipReason::kManagedMemoryUnsupported);
  }
}
}  // namespace

// @asserts: hipMemAdvise_v2 - the location-based set-read-mostly hint on managed memory is accepted or reported unsupported
HIP_TEST_CASE(Contract_MemAdviseV2_SetReadMostly_IsAcceptedOrUnsupported) {
  SkipIfManagedMemoryUnsupported();
  hip::contract::ContractCleanup cleanup;

  void* ptr = nullptr;
  HIP_CHECK(hipMallocManaged(&ptr, kRangeBytes, hipMemAttachGlobal));
  cleanup.Add([ptr] { (void)hipFree(ptr); });

  // The location-based advise must either accept the read-mostly hint or report
  // that the path is unsupported on this runtime. Any other status is a contract
  // violation. The attribute is not read back because the advice is advisory and
  // not required to be observable through the range-attribute query on every
  // path (see the mem_advise domain, which gates those assertions).
  const hipMemLocation location = CurrentDeviceLocation();
  const hipError_t status = hipMemAdvise_v2(ptr, kRangeBytes, hipMemAdviseSetReadMostly, location);
  if (status != hipSuccess && status != hipErrorNotSupported) {
    HIP_CHECK(status);
  }
}

// @asserts: hipMemAdvise_v2 - setting then unsetting the preferred location for a range is each accepted or reported unsupported
HIP_TEST_CASE(Contract_MemAdviseV2_SetAndUnsetPreferredLocation_IsAcceptedOrUnsupported) {
  SkipIfManagedMemoryUnsupported();
  hip::contract::ContractCleanup cleanup;

  void* ptr = nullptr;
  HIP_CHECK(hipMallocManaged(&ptr, kRangeBytes, hipMemAttachGlobal));
  cleanup.Add([ptr] { (void)hipFree(ptr); });

  const hipMemLocation location = CurrentDeviceLocation();

  // Setting then unsetting the preferred location for the range must each be
  // accepted or reported unsupported.
  const hipError_t set_status =
      hipMemAdvise_v2(ptr, kRangeBytes, hipMemAdviseSetPreferredLocation, location);
  if (set_status != hipSuccess && set_status != hipErrorNotSupported) {
    HIP_CHECK(set_status);
  }

  const hipError_t unset_status =
      hipMemAdvise_v2(ptr, kRangeBytes, hipMemAdviseUnsetPreferredLocation, location);
  if (unset_status != hipSuccess && unset_status != hipErrorNotSupported) {
    HIP_CHECK(unset_status);
  }
}

// @asserts: hipMemPrefetchAsync_v2 - a location-based prefetch either completes after stream sync or is reported unsupported
HIP_TEST_CASE(Contract_MemAdviseV2_PrefetchAsync_IsAcceptedOrUnsupported) {
  SkipIfManagedMemoryUnsupported();
  hip::contract::ContractCleanup cleanup;

  void* ptr = nullptr;
  hipStream_t stream = nullptr;
  HIP_CHECK(hipMallocManaged(&ptr, kRangeBytes, hipMemAttachGlobal));
  cleanup.Add([ptr] { (void)hipFree(ptr); });
  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([stream] { (void)hipStreamDestroy(stream); });

  // The location-based prefetch must either succeed (and complete after the
  // stream is synchronized) or report that prefetch is unsupported.
  const hipMemLocation location = CurrentDeviceLocation();
  const hipError_t status = hipMemPrefetchAsync_v2(ptr, kRangeBytes, location, 0, stream);
  if (status == hipErrorNotSupported) {
    HIP_SKIP_TEST("Location-based prefetch is not supported by this device/runtime path.");
  }
  HIP_CHECK(status);
  HIP_CHECK(hipStreamSynchronize(stream));
}

// @asserts: hipMemAdvise_v2 - advising a null range does not silently succeed and returns a non-success status
HIP_TEST_CASE(Contract_MemAdviseV2_NullPointer_IsRejected) {
  SkipIfManagedMemoryUnsupported();

  // Advising a null range must not silently succeed. The exact error code is
  // backend-specific, so only a non-success status is required.
  const hipMemLocation location = CurrentDeviceLocation();
  const hipError_t status =
      hipMemAdvise_v2(nullptr, kRangeBytes, hipMemAdviseSetReadMostly, location);
  REQUIRE(status != hipSuccess);
}
