/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>
#include <contract_cleanup.hh>

namespace {
bool MemoryPoolsSupported() {
  int device = 0;
  int supported = 0;
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipDeviceGetAttribute(&supported, hipDeviceAttributeMemoryPoolsSupported, device));
  return supported != 0;
}

void SkipIfMemoryPoolsUnsupported() {
  if (!MemoryPoolsSupported()) {
    HIP_SKIP_TEST("HIP memory pools are not supported by this device/runtime path.");
  }
}

hipMemLocation CurrentDeviceLocation() {
  int device = 0;
  HIP_CHECK(hipGetDevice(&device));
  hipMemLocation location{};
  location.type = hipMemLocationTypeDevice;
  location.id = device;
  return location;
}
}  // namespace

// @asserts: hipMemGetMemPool - a location-based pool query returns a non-null pool for a valid device location
HIP_TEST_CASE(Contract_MemLocationPool_GetMemPool_ReturnsPoolForDeviceLocation) {
  SkipIfMemoryPoolsUnsupported();

  // The location-based pool query must return a non-null pool for a valid device
  // location and the pinned allocation type.
  hipMemLocation location = CurrentDeviceLocation();
  hipMemPool_t pool = nullptr;
  HIP_CHECK(hipMemGetMemPool(&pool, &location, hipMemAllocationTypePinned));

  REQUIRE(pool != nullptr);
}

// @asserts: hipMemSetMemPool - a pool set for a location round-trips through a subsequent hipMemGetMemPool query
HIP_TEST_CASE(Contract_MemLocationPool_SetMemPool_RoundTripsThroughGetMemPool) {
  SkipIfMemoryPoolsUnsupported();

  int device = 0;
  HIP_CHECK(hipGetDevice(&device));
  hipMemLocation location = CurrentDeviceLocation();

  // Setting the device's default pool as the current pool for the location must
  // round-trip: a subsequent location query must report the same pool handle.
  hipMemPool_t default_pool = nullptr;
  HIP_CHECK(hipDeviceGetDefaultMemPool(&default_pool, device));
  REQUIRE(default_pool != nullptr);

  HIP_CHECK(hipMemSetMemPool(&location, hipMemAllocationTypePinned, default_pool));

  hipMemPool_t readback = nullptr;
  HIP_CHECK(hipMemGetMemPool(&readback, &location, hipMemAllocationTypePinned));
  REQUIRE(readback == default_pool);
}

// @asserts: hipMemGetAccess - querying access for a pooled allocation's location returns a defined protection flag
HIP_TEST_CASE(Contract_MemLocationPool_GetAccess_ReturnsFlagsForPooledAllocation) {
  SkipIfMemoryPoolsUnsupported();
  hip::contract::ContractCleanup cleanup;

  hipMemLocation location = CurrentDeviceLocation();
  hipStream_t stream = nullptr;
  void* pooled = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([stream] { (void)hipStreamDestroy(stream); });

  const hipError_t alloc_status = hipMallocAsync(&pooled, 256, stream);
  if (alloc_status == hipErrorNotSupported) {
    HIP_SKIP_TEST("Stream-ordered allocation is not supported by this device/runtime path.");
  }
  HIP_CHECK(alloc_status);
  // Free-and-drain on teardown: the async free is enqueued on the stream, then
  // the stream is synchronized so the free completes before the stream-destroy
  // action (registered earlier, so it runs after this one) tears the stream down.
  cleanup.Add([pooled, stream] {
    (void)hipFreeAsync(pooled, stream);
    (void)hipStreamSynchronize(stream);
  });
  HIP_CHECK(hipStreamSynchronize(stream));
  REQUIRE(pooled != nullptr);

  // Querying access for the owning device location must succeed and report one of
  // the defined protection flag values. The exact value is policy-dependent, so
  // only membership in the valid set is asserted.
  unsigned long long flags = 0xFFFFFFFFull;
  HIP_CHECK(hipMemGetAccess(&flags, &location, pooled));
  REQUIRE((flags == hipMemAccessFlagsProtNone || flags == hipMemAccessFlagsProtRead ||
           flags == hipMemAccessFlagsProtReadWrite));
}
