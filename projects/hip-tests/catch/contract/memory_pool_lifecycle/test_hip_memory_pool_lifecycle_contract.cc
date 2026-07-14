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

HIP_TEST_CASE(Contract_MemoryPoolLifecycle_CreateDestroy_SucceedsWhenSupported) {
  SkipIfMemoryPoolsUnsupported();
  hipMemPool_t pool = nullptr;

  if (!CreatePool(&pool)) {
    HIP_SKIP_TEST("hipMemPoolCreate is not supported by this device/runtime path.");
  }

  REQUIRE(pool != nullptr);

  HIP_CHECK(hipMemPoolDestroy(pool));
}

HIP_TEST_CASE(Contract_MemoryPoolLifecycle_GetSetReleaseThreshold_RoundTripsValue) {
  SkipIfMemoryPoolsUnsupported();
  hip::contract::ContractCleanup cleanup;
  hipMemPool_t pool = nullptr;
  uint64_t threshold = 4096;
  uint64_t readback = 0;

  if (!CreatePool(&pool)) {
    HIP_SKIP_TEST("hipMemPoolCreate is not supported by this device/runtime path.");
  }
  cleanup.Add([&] { (void)hipMemPoolDestroy(pool); });

  HIP_CHECK(hipMemPoolSetAttribute(pool, hipMemPoolAttrReleaseThreshold, &threshold));
  HIP_CHECK(hipMemPoolGetAttribute(pool, hipMemPoolAttrReleaseThreshold, &readback));

  REQUIRE(readback == threshold);
}

HIP_TEST_CASE(Contract_MemoryPoolLifecycle_TrimTo_SucceedsOnEmptyPool) {
  SkipIfMemoryPoolsUnsupported();
  hip::contract::ContractCleanup cleanup;
  hipMemPool_t pool = nullptr;

  if (!CreatePool(&pool)) {
    HIP_SKIP_TEST("hipMemPoolCreate is not supported by this device/runtime path.");
  }
  cleanup.Add([&] { (void)hipMemPoolDestroy(pool); });

  HIP_CHECK(hipMemPoolTrimTo(pool, 0));
}

HIP_TEST_CASE(Contract_MemoryPoolLifecycle_MallocFromCreatedPoolAsync_SucceedsWhenSupported) {
  SkipIfMemoryPoolsUnsupported();
  hip::contract::ContractCleanup cleanup;
  hipMemPool_t pool = nullptr;
  hipStream_t stream = nullptr;
  void* ptr = nullptr;

  if (!CreatePool(&pool)) {
    HIP_SKIP_TEST("hipMemPoolCreate is not supported by this device/runtime path.");
  }
  cleanup.Add([&] { (void)hipMemPoolDestroy(pool); });

  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([&] { (void)hipStreamDestroy(stream); });
  const hipError_t status = hipMallocFromPoolAsync(&ptr, 128, pool, stream);
  if (status == hipErrorNotSupported) {
    HIP_SKIP_TEST("hipMallocFromPoolAsync is not supported by this device/runtime path.");
  }
  HIP_CHECK(status);
  cleanup.Add([&] { (void)hipFreeAsync(ptr, stream); });
  REQUIRE(ptr != nullptr);
}
