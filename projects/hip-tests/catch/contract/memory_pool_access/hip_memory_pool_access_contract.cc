/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
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

hipMemAccessDesc CurrentDeviceAccessDesc(hipMemAccessFlags flags) {
  int device = 0;
  HIP_CHECK(hipGetDevice(&device));
  hipMemAccessDesc desc{};
  desc.location.type = hipMemLocationTypeDevice;
  desc.location.id = device;
  desc.flags = flags;
  return desc;
}
}

HIP_TEST_CASE(Contract_MemoryPoolAccess_SetAccessCurrentDevice_SucceedsWhenSupported) {
  SkipIfMemoryPoolsUnsupported();
  hipMemPool_t pool = nullptr;
  const auto access = CurrentDeviceAccessDesc(hipMemAccessFlagsProtReadWrite);

  if (!CreatePool(&pool)) {
    HIP_SKIP_TEST("hipMemPoolCreate is not supported by this device/runtime path.");
  }

  const hipError_t status = hipMemPoolSetAccess(pool, &access, 1);
  if (status == hipErrorNotSupported) {
    HIP_CHECK(hipMemPoolDestroy(pool));
    HIP_SKIP_TEST("hipMemPoolSetAccess is not supported by this device/runtime path.");
  }
  HIP_CHECK(status);

  HIP_CHECK(hipMemPoolDestroy(pool));
}

HIP_TEST_CASE(Contract_MemoryPoolAccess_GetAccessCurrentDevice_ReturnsGrantedFlags) {
  SkipIfMemoryPoolsUnsupported();
  hipMemPool_t pool = nullptr;
  const auto access = CurrentDeviceAccessDesc(hipMemAccessFlagsProtReadWrite);
  hipMemAccessFlags flags = hipMemAccessFlagsProtNone;

  if (!CreatePool(&pool)) {
    HIP_SKIP_TEST("hipMemPoolCreate is not supported by this device/runtime path.");
  }

  const hipError_t set_status = hipMemPoolSetAccess(pool, &access, 1);
  if (set_status == hipErrorNotSupported) {
    HIP_CHECK(hipMemPoolDestroy(pool));
    HIP_SKIP_TEST("hipMemPoolSetAccess is not supported by this device/runtime path.");
  }
  HIP_CHECK(set_status);
  hipMemLocation location = access.location;
  HIP_CHECK(hipMemPoolGetAccess(&flags, pool, &location));

  REQUIRE((flags & hipMemAccessFlagsProtReadWrite) == hipMemAccessFlagsProtReadWrite);

  HIP_CHECK(hipMemPoolDestroy(pool));
}

HIP_TEST_CASE(Contract_MemoryPoolAccess_SetAccessNoneCurrentDevice_IsRejected) {
  SkipIfMemoryPoolsUnsupported();
  hipMemPool_t pool = nullptr;
  auto access = CurrentDeviceAccessDesc(hipMemAccessFlagsProtReadWrite);

  if (!CreatePool(&pool)) {
    HIP_SKIP_TEST("hipMemPoolCreate is not supported by this device/runtime path.");
  }

  const hipError_t set_status = hipMemPoolSetAccess(pool, &access, 1);
  if (set_status == hipErrorNotSupported) {
    HIP_CHECK(hipMemPoolDestroy(pool));
    HIP_SKIP_TEST("hipMemPoolSetAccess is not supported by this device/runtime path.");
  }
  HIP_CHECK(set_status);

  access.flags = hipMemAccessFlagsProtNone;
  const hipError_t revoke_status = hipMemPoolSetAccess(pool, &access, 1);
  if (revoke_status == hipErrorNotSupported) {
    HIP_CHECK(hipMemPoolDestroy(pool));
    HIP_SKIP_TEST("hipMemPoolSetAccess with hipMemAccessFlagsProtNone is not supported by this device/runtime path.");
  }
  REQUIRE((revoke_status == hipErrorInvalidDevice || revoke_status == hipErrorInvalidValue));

  HIP_CHECK(hipMemPoolDestroy(pool));
}
