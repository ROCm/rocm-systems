/*
 * Copyright Advanced Micro Devices, Inc.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>

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
HIP_TEST_CASE(Contract_MemLocationPool_HipMemGetMemPool_Default_ReturnsPoolForDeviceLocation) {
#if HT_NVIDIA && defined(CUDA_VERSION) && CUDA_VERSION < 13000
  HIP_SKIP_TEST("Location-based memory-pool queries are not exposed before CUDA 13.0 on the NVIDIA backend.");
#else
  SkipIfMemoryPoolsUnsupported();

  // The location-based pool query must return a non-null pool for a valid device
  // location and the pinned allocation type.
  hipMemLocation location = CurrentDeviceLocation();
  hipMemPool_t pool = nullptr;
  HIP_CHECK(hipMemGetMemPool(&pool, &location, hipMemAllocationTypePinned));

  REQUIRE(pool != nullptr);
#endif  // HT_NVIDIA && CUDA_VERSION < 13000
}

// @asserts: hipMemGetDefaultMemPool - querying the current device location's default memory pool returns a non-null pool when memory pools are supported
HIP_TEST_CASE(Contract_MemLocationPool_HipMemGetDefaultMemPool_CurrentDevice_ReturnsNonNullPool) {
#if HT_NVIDIA
  HIP_SKIP_TEST("hipMemGetDefaultMemPool is not exposed by the NVIDIA backend.");
#else
  SkipIfMemoryPoolsUnsupported();

  hipMemLocation location = CurrentDeviceLocation();
  hipMemPool_t default_pool = nullptr;
  HIP_CHECK(hipMemGetDefaultMemPool(&default_pool, &location, hipMemAllocationTypePinned));
  REQUIRE(default_pool != nullptr);
#endif  // HT_NVIDIA
}

// @asserts: hipMemSetMemPool - a pool set for a location round-trips through a subsequent hipMemGetMemPool query
HIP_TEST_CASE(Contract_MemLocationPool_HipMemSetMemPool_Default_RoundTripsThroughGetMemPool) {
#if HT_NVIDIA && defined(CUDA_VERSION) && CUDA_VERSION < 13000
  HIP_SKIP_TEST("Location-based memory-pool set/query APIs are not exposed before CUDA 13.0 on the NVIDIA backend.");
#else
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
#endif  // HT_NVIDIA && CUDA_VERSION < 13000
}
