/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <array>
#include <cstddef>
#include <cstdint>

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>
#include <contract_cleanup.hh>

namespace {
constexpr size_t kElementCount = 128;

std::array<uint8_t, kElementCount> MakePattern(uint8_t seed) {
  std::array<uint8_t, kElementCount> pattern{};
  for (size_t i = 0; i < pattern.size(); ++i) {
    pattern[i] = static_cast<uint8_t>(seed + i);
  }
  return pattern;
}

int CurrentDevice() {
  int device = 0;
  HIP_CHECK(hipGetDevice(&device));
  return device;
}

hipMemAllocationProp DeviceAllocationProp() {
  hipMemAllocationProp prop{};
  prop.type = hipMemAllocationTypePinned;
  prop.requestedHandleTypes = hipMemHandleTypeNone;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = CurrentDevice();
  return prop;
}

bool VmmSupported() {
  int supported = 0;
  HIP_CHECK(hipDeviceGetAttribute(&supported, hipDeviceAttributeVirtualMemoryManagementSupported,
                                  CurrentDevice()));
  return supported != 0;
}

void SkipIfVmmUnsupported() {
  if (!VmmSupported()) {
    HIP_SKIP_TEST("HIP virtual memory management is not supported by this device/runtime path.");
  }
}

size_t AllocationGranularity() {
  const auto prop = DeviceAllocationProp();
  size_t granularity = 0;
  HIP_CHECK(hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));
  return granularity;
}

bool CreateAllocationHandle(hipMemGenericAllocationHandle_t* handle, size_t size) {
  const auto prop = DeviceAllocationProp();
  const hipError_t status = hipMemCreate(handle, size, &prop, 0);
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

HIP_TEST_CASE(Contract_Vmm_GetAllocationGranularity_ReturnsPositiveValue) {
  SkipIfVmmUnsupported();
  const size_t granularity = AllocationGranularity();

  REQUIRE(granularity > 0);
}

HIP_TEST_CASE(Contract_Vmm_AddressReserveFree_Succeeds) {
  SkipIfVmmUnsupported();
  hip::contract::ContractCleanup cleanup;
  const size_t size = AllocationGranularity();
  void* address = nullptr;

  HIP_CHECK(hipMemAddressReserve(&address, size, 0, nullptr, 0));
  cleanup.Add([address, size] { (void)hipMemAddressFree(address, size); });

  REQUIRE(address != nullptr);
}

HIP_TEST_CASE(Contract_Vmm_CreateReleaseAllocationHandle_SucceedsWhenSupported) {
  SkipIfVmmUnsupported();
  hip::contract::ContractCleanup cleanup;
  const size_t size = AllocationGranularity();
  hipMemGenericAllocationHandle_t handle{};

  if (!CreateAllocationHandle(&handle, size)) {
    HIP_SKIP_TEST("hipMemCreate is not supported by this device/runtime path.");
  }
  cleanup.Add([handle] { (void)hipMemRelease(handle); });
}

HIP_TEST_CASE(Contract_Vmm_MapUnmap_SucceedsWhenSupported) {
  SkipIfVmmUnsupported();
  hip::contract::ContractCleanup cleanup;
  const size_t size = AllocationGranularity();
  void* address = nullptr;
  hipMemGenericAllocationHandle_t handle{};

  if (!CreateAllocationHandle(&handle, size)) {
    HIP_SKIP_TEST("hipMemCreate is not supported by this device/runtime path.");
  }
  cleanup.Add([handle] { (void)hipMemRelease(handle); });

  HIP_CHECK(hipMemAddressReserve(&address, size, 0, nullptr, 0));
  cleanup.Add([address, size] { (void)hipMemAddressFree(address, size); });

  const hipError_t map_status = hipMemMap(address, size, 0, handle, 0);
  if (map_status == hipErrorNotSupported) {
    HIP_SKIP_TEST("hipMemMap is not supported by this device/runtime path.");
  }
  HIP_CHECK(map_status);
  cleanup.Add([address, size] { (void)hipMemUnmap(address, size); });
}

HIP_TEST_CASE(Contract_Vmm_SetAccess_AllowsRoundTripWhenSupported) {
  SkipIfVmmUnsupported();
  hip::contract::ContractCleanup cleanup;
  const auto src = MakePattern(0x33);
  std::array<uint8_t, kElementCount> dst{};
  const size_t size = AllocationGranularity();
  void* address = nullptr;
  hipMemGenericAllocationHandle_t handle{};

  if (!CreateAllocationHandle(&handle, size)) {
    HIP_SKIP_TEST("hipMemCreate is not supported by this device/runtime path.");
  }
  cleanup.Add([handle] { (void)hipMemRelease(handle); });

  HIP_CHECK(hipMemAddressReserve(&address, size, 0, nullptr, 0));
  cleanup.Add([address, size] { (void)hipMemAddressFree(address, size); });

  const hipError_t map_status = hipMemMap(address, size, 0, handle, 0);
  if (map_status == hipErrorNotSupported) {
    HIP_SKIP_TEST("hipMemMap is not supported by this device/runtime path.");
  }
  HIP_CHECK(map_status);
  cleanup.Add([address, size] { (void)hipMemUnmap(address, size); });

  hipMemAccessDesc access{};
  access.location.type = hipMemLocationTypeDevice;
  access.location.id = CurrentDevice();
  access.flags = hipMemAccessFlagsProtReadWrite;
  HIP_CHECK(hipMemSetAccess(address, size, &access, 1));

  HIP_CHECK(hipMemcpy(address, src.data(), src.size(), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dst.data(), address, dst.size(), hipMemcpyDeviceToHost));

  REQUIRE(dst == src);
}
