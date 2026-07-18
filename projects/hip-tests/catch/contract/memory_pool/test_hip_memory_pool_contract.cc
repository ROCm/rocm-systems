/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <cstdint>

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

hipMemPool_t CurrentDefaultPool() {
  int device = 0;
  hipMemPool_t pool = nullptr;
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipDeviceGetDefaultMemPool(&pool, device));
  return pool;
}
}

HIP_TEST_CASE(Contract_MemoryPool_DeviceGetDefaultMemPool_ReturnsPool) {
  SkipIfMemoryPoolsUnsupported();
  hipMemPool_t pool = CurrentDefaultPool();

  REQUIRE(pool != nullptr);
}

HIP_TEST_CASE(Contract_MemoryPool_DeviceGetMemPool_ReturnsCurrentPool) {
  SkipIfMemoryPoolsUnsupported();
  int device = 0;
  hipMemPool_t pool = nullptr;

  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipDeviceGetMemPool(&pool, device));

  REQUIRE(pool != nullptr);
}

HIP_TEST_CASE(Contract_MemoryPool_SetCurrentMemPool_RoundTripsThroughGetMemPool) {
  SkipIfMemoryPoolsUnsupported();
  int device = 0;
  hipMemPool_t default_pool = nullptr;
  hipMemPool_t current_pool = nullptr;

  HIP_CHECK(hipGetDevice(&device));
  default_pool = CurrentDefaultPool();
  HIP_CHECK(hipDeviceSetMemPool(device, default_pool));
  HIP_CHECK(hipDeviceGetMemPool(&current_pool, device));

  REQUIRE(current_pool == default_pool);
}

HIP_TEST_CASE(Contract_MemoryPool_GetSetReleaseThreshold_RoundTripsValue) {
  SkipIfMemoryPoolsUnsupported();
  hipMemPool_t pool = CurrentDefaultPool();
  uint64_t original_threshold = 0;
  uint64_t requested_threshold = 4096;
  uint64_t readback_threshold = 0;

  HIP_CHECK(hipMemPoolGetAttribute(pool, hipMemPoolAttrReleaseThreshold, &original_threshold));
  HIP_CHECK(hipMemPoolSetAttribute(pool, hipMemPoolAttrReleaseThreshold, &requested_threshold));
  HIP_CHECK(hipMemPoolGetAttribute(pool, hipMemPoolAttrReleaseThreshold, &readback_threshold));

  REQUIRE(readback_threshold == requested_threshold);

  HIP_CHECK(hipMemPoolSetAttribute(pool, hipMemPoolAttrReleaseThreshold, &original_threshold));
}

HIP_TEST_CASE(Contract_MemoryPool_MallocAsyncFreeAsync_SucceedsWhenSupported) {
  SkipIfMemoryPoolsUnsupported();
  hip::contract::ContractCleanup cleanup;
  void* ptr = nullptr;
  hipStream_t stream = nullptr;

  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([stream] { (void)hipStreamDestroy(stream); });
  HIP_CHECK(hipMallocAsync(&ptr, 128, stream));
  // Register the free immediately so a failing REQUIRE or a throwing hipFreeAsync
  // below cannot leak the allocation. The guard frees only if the explicit free
  // has not already run (tracked by nulling ptr), avoiding a double free.
  cleanup.Add([&] {
    if (ptr != nullptr) (void)hipFreeAsync(ptr, stream);
  });
  REQUIRE(ptr != nullptr);
  HIP_CHECK(hipFreeAsync(ptr, stream));
  ptr = nullptr;
  HIP_CHECK(hipStreamSynchronize(stream));
}

HIP_TEST_CASE(Contract_MemoryPool_MallocAsync_MemoryUsableAfterStreamSynchronize) {
  SkipIfMemoryPoolsUnsupported();
  hip::contract::ContractCleanup cleanup;
  void* ptr = nullptr;
  hipStream_t stream = nullptr;
  uint8_t value = 0;

  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([stream] { (void)hipStreamDestroy(stream); });
  HIP_CHECK(hipMallocAsync(&ptr, sizeof(value), stream));
  // Free-and-drain on teardown: enqueue the async free, then synchronize the
  // stream so the free completes before the stream-destroy action (registered
  // earlier, so it runs after this one) tears the stream down.
  cleanup.Add([ptr, stream] {
    (void)hipFreeAsync(ptr, stream);
    (void)hipStreamSynchronize(stream);
  });
  HIP_CHECK(hipMemsetAsync(ptr, 0x5a, sizeof(value), stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipMemcpy(&value, ptr, sizeof(value), hipMemcpyDeviceToHost));

  REQUIRE(value == 0x5a);
}
