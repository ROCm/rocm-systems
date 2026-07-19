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

// @asserts: hipMemPoolSetAccess - granting the current device read-write pool access is accepted-or-unsupported
HIP_TEST_CASE(Contract_MemoryPoolAccess_SetAccessCurrentDevice_SucceedsWhenSupported) {
  SkipIfMemoryPoolsUnsupported();
  hip::contract::ContractCleanup cleanup;
  hipMemPool_t pool = nullptr;
  const auto access = CurrentDeviceAccessDesc(hipMemAccessFlagsProtReadWrite);

  if (!CreatePool(&pool)) {
    HIP_SKIP_TEST("hipMemPoolCreate is not supported by this device/runtime path.");
  }
  cleanup.Add([pool] { (void)hipMemPoolDestroy(pool); });

  const hipError_t status = hipMemPoolSetAccess(pool, &access, 1);
  if (status == hipErrorNotSupported) {
    HIP_SKIP_TEST("hipMemPoolSetAccess is not supported by this device/runtime path.");
  }
  HIP_CHECK(status);
}

// @asserts: hipMemPoolGetAccess - reads back the read-write flags previously granted to the current device via SetAccess
HIP_TEST_CASE(Contract_MemoryPoolAccess_GetAccessCurrentDevice_ReturnsGrantedFlags) {
  SkipIfMemoryPoolsUnsupported();
  hip::contract::ContractCleanup cleanup;
  hipMemPool_t pool = nullptr;
  const auto access = CurrentDeviceAccessDesc(hipMemAccessFlagsProtReadWrite);
  hipMemAccessFlags flags = hipMemAccessFlagsProtNone;

  if (!CreatePool(&pool)) {
    HIP_SKIP_TEST("hipMemPoolCreate is not supported by this device/runtime path.");
  }
  cleanup.Add([pool] { (void)hipMemPoolDestroy(pool); });

  const hipError_t set_status = hipMemPoolSetAccess(pool, &access, 1);
  if (set_status == hipErrorNotSupported) {
    HIP_SKIP_TEST("hipMemPoolSetAccess is not supported by this device/runtime path.");
  }
  HIP_CHECK(set_status);
  hipMemLocation location = access.location;
  HIP_CHECK(hipMemPoolGetAccess(&flags, pool, &location));

  REQUIRE((flags & hipMemAccessFlagsProtReadWrite) == hipMemAccessFlagsProtReadWrite);
}

// @asserts: hipMemPoolSetAccess - revoking the current device's own access (ProtNone) rejects with InvalidDevice or InvalidValue
HIP_TEST_CASE(Contract_MemoryPoolAccess_SetAccessNoneCurrentDevice_IsRejected) {
  SkipIfMemoryPoolsUnsupported();
  hip::contract::ContractCleanup cleanup;
  hipMemPool_t pool = nullptr;
  auto access = CurrentDeviceAccessDesc(hipMemAccessFlagsProtReadWrite);

  if (!CreatePool(&pool)) {
    HIP_SKIP_TEST("hipMemPoolCreate is not supported by this device/runtime path.");
  }
  cleanup.Add([pool] { (void)hipMemPoolDestroy(pool); });

  const hipError_t set_status = hipMemPoolSetAccess(pool, &access, 1);
  if (set_status == hipErrorNotSupported) {
    HIP_SKIP_TEST("hipMemPoolSetAccess is not supported by this device/runtime path.");
  }
  HIP_CHECK(set_status);

  access.flags = hipMemAccessFlagsProtNone;
  const hipError_t revoke_status = hipMemPoolSetAccess(pool, &access, 1);
  if (revoke_status == hipErrorNotSupported) {
    HIP_SKIP_TEST("hipMemPoolSetAccess with hipMemAccessFlagsProtNone is not supported by this device/runtime path.");
  }
  REQUIRE((revoke_status == hipErrorInvalidDevice || revoke_status == hipErrorInvalidValue));
}
