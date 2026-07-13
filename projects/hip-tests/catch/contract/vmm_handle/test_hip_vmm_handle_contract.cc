/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <cstddef>

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>

namespace {
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

// A mapped VMM allocation: a physical handle created and mapped at a reserved
// virtual address. Returns false (with cleanup) if any step reports
// hipErrorNotSupported so callers can skip on unsupported runtime paths.
struct MappedAllocation {
  hipMemGenericAllocationHandle_t handle{};
  void* address = nullptr;
  size_t size = 0;
  bool mapped = false;
};

bool CreateMappedAllocation(MappedAllocation* out) {
  out->size = AllocationGranularity();

  const auto prop = DeviceAllocationProp();
  hipError_t status = hipMemCreate(&out->handle, out->size, &prop, 0);
  if (status == hipErrorNotSupported) {
    return false;
  }
  HIP_CHECK(status);

  HIP_CHECK(hipMemAddressReserve(&out->address, out->size, 0, nullptr, 0));

  status = hipMemMap(out->address, out->size, 0, out->handle, 0);
  if (status == hipErrorNotSupported) {
    HIP_CHECK(hipMemAddressFree(out->address, out->size));
    HIP_CHECK(hipMemRelease(out->handle));
    out->address = nullptr;
    return false;
  }
  HIP_CHECK(status);
  out->mapped = true;
  return true;
}

void DestroyMappedAllocation(MappedAllocation* a) {
  if (a->mapped) {
    HIP_CHECK(hipMemUnmap(a->address, a->size));
  }
  if (a->address != nullptr) {
    HIP_CHECK(hipMemAddressFree(a->address, a->size));
  }
  HIP_CHECK(hipMemRelease(a->handle));
}
}  // namespace

HIP_TEST_CASE(Contract_VmmHandle_RetainAllocationHandle_ByAddress_Succeeds) {
  SkipIfVmmUnsupported();

  MappedAllocation alloc;
  if (!CreateMappedAllocation(&alloc)) {
    HIP_SKIP_TEST("VMM create/map is not supported by this device/runtime path.");
  }

  // Retaining the allocation handle for a mapped address must return a usable
  // handle that can be released independently of the original.
  hipMemGenericAllocationHandle_t retained{};
  HIP_CHECK(hipMemRetainAllocationHandle(&retained, alloc.address));
  HIP_CHECK(hipMemRelease(retained));

  DestroyMappedAllocation(&alloc);
}

HIP_TEST_CASE(Contract_VmmHandle_GetAllocationProperties_RoundTripsFromHandle) {
  SkipIfVmmUnsupported();

  MappedAllocation alloc;
  if (!CreateMappedAllocation(&alloc)) {
    HIP_SKIP_TEST("VMM create/map is not supported by this device/runtime path.");
  }

  // The properties queried from the handle must reflect what the allocation was
  // created with: pinned type on the current device location.
  hipMemAllocationProp prop{};
  HIP_CHECK(hipMemGetAllocationPropertiesFromHandle(&prop, alloc.handle));
  REQUIRE(prop.type == hipMemAllocationTypePinned);
  REQUIRE(prop.location.type == hipMemLocationTypeDevice);
  REQUIRE(prop.location.id == CurrentDevice());

  DestroyMappedAllocation(&alloc);
}

HIP_TEST_CASE(Contract_VmmHandle_GetHandleForAddressRange_DmaBufFd_IsQueryableWhenSupported) {
  SkipIfVmmUnsupported();

  MappedAllocation alloc;
  if (!CreateMappedAllocation(&alloc)) {
    HIP_SKIP_TEST("VMM create/map is not supported by this device/runtime path.");
  }

  // Exporting a dma-buf file descriptor for the mapped range is an OS/driver
  // capability. When supported it must yield a non-negative fd; when not, the
  // runtime reports a non-success status and the contract skips rather than
  // failing.
  int fd = -1;
  const hipError_t status = hipMemGetHandleForAddressRange(
      &fd, reinterpret_cast<hipDeviceptr_t>(alloc.address), alloc.size,
      hipMemRangeHandleTypeDmaBufFd, 0);
  if (status != hipSuccess) {
    DestroyMappedAllocation(&alloc);
    HIP_SKIP_TEST("dma-buf handle export is not supported by this device/runtime path.");
  }
  REQUIRE(fd >= 0);

  DestroyMappedAllocation(&alloc);
}
