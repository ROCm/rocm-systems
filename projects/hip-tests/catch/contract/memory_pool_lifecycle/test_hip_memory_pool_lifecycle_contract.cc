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

hipMemPoolProps CurrentDevicePoolProps() {
  int device = 0;
  HIP_CHECK(hipGetDevice(&device));
  hipMemPoolProps props{};
  props.allocType = hipMemAllocationTypePinned;
  props.handleTypes = hipMemHandleTypeNone;
  props.location.type = hipMemLocationTypeDevice;
  props.location.id = device;
  return props;
}

bool CreatePool(hipMemPool_t* pool) {
  const auto props = CurrentDevicePoolProps();
  const hipError_t status = hipMemPoolCreate(pool, &props);
  if (status == hipSuccess) {
    return true;
  }
  if (status == hipErrorNotSupported) {
    return false;
  }
  HIP_CHECK(status);
  return false;
}
}

// @asserts: hipMemPoolCreate - creating a pool yields a non-null handle that hipMemPoolDestroy releases, or is unsupported
HIP_TEST_CASE(Contract_MemoryPoolLifecycle_CreateDestroy_SucceedsWhenSupported) {
  SkipIfMemoryPoolsUnsupported();
  hipMemPool_t pool = nullptr;

  if (!CreatePool(&pool)) {
    HIP_SKIP_TEST("hipMemPoolCreate is not supported by this device/runtime path.");
  }

  REQUIRE(pool != nullptr);

  HIP_CHECK(hipMemPoolDestroy(pool));
}

// @asserts: hipMemPoolSetAttribute - a release-threshold value set on a pool reads back unchanged via hipMemPoolGetAttribute
HIP_TEST_CASE(Contract_MemoryPoolLifecycle_GetSetReleaseThreshold_RoundTripsValue) {
  SkipIfMemoryPoolsUnsupported();
  hip::contract::ContractCleanup cleanup;
  hipMemPool_t pool = nullptr;
  uint64_t threshold = 4096;
  uint64_t readback = 0;

  if (!CreatePool(&pool)) {
    HIP_SKIP_TEST("hipMemPoolCreate is not supported by this device/runtime path.");
  }
  cleanup.Add([pool] { (void)hipMemPoolDestroy(pool); });

  HIP_CHECK(hipMemPoolSetAttribute(pool, hipMemPoolAttrReleaseThreshold, &threshold));
  HIP_CHECK(hipMemPoolGetAttribute(pool, hipMemPoolAttrReleaseThreshold, &readback));

  REQUIRE(readback == threshold);
}

// @asserts: hipMemPoolTrimTo - trimming an empty pool to zero reserved bytes succeeds
HIP_TEST_CASE(Contract_MemoryPoolLifecycle_TrimTo_SucceedsOnEmptyPool) {
  SkipIfMemoryPoolsUnsupported();
  hip::contract::ContractCleanup cleanup;
  hipMemPool_t pool = nullptr;

  if (!CreatePool(&pool)) {
    HIP_SKIP_TEST("hipMemPoolCreate is not supported by this device/runtime path.");
  }
  cleanup.Add([pool] { (void)hipMemPoolDestroy(pool); });

  HIP_CHECK(hipMemPoolTrimTo(pool, 0));
}

// @asserts: hipMallocFromPoolAsync - an async allocation from a created pool on a stream yields a non-null pointer, or is unsupported
HIP_TEST_CASE(Contract_MemoryPoolLifecycle_MallocFromCreatedPoolAsync_SucceedsWhenSupported) {
  SkipIfMemoryPoolsUnsupported();
  hip::contract::ContractCleanup cleanup;
  hipMemPool_t pool = nullptr;
  hipStream_t stream = nullptr;
  void* ptr = nullptr;

  if (!CreatePool(&pool)) {
    HIP_SKIP_TEST("hipMemPoolCreate is not supported by this device/runtime path.");
  }
  cleanup.Add([pool] { (void)hipMemPoolDestroy(pool); });

  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([stream] { (void)hipStreamDestroy(stream); });
  const hipError_t status = hipMallocFromPoolAsync(&ptr, 128, pool, stream);
  if (status == hipErrorNotSupported) {
    HIP_SKIP_TEST("hipMallocFromPoolAsync is not supported by this device/runtime path.");
  }
  HIP_CHECK(status);
  // Free-and-drain on teardown: enqueue the async free, then synchronize the
  // stream so the free completes before the stream-destroy and pool-destroy
  // actions (registered earlier, so they run after this one) tear down the stream
  // and pool the allocation depends on.
  cleanup.Add([ptr, stream] {
    (void)hipFreeAsync(ptr, stream);
    (void)hipStreamSynchronize(stream);
  });
  REQUIRE(ptr != nullptr);
}
