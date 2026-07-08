/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <cstddef>
#include <cstdint>

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>

namespace {
constexpr size_t kAllocationBytes = 4096;
constexpr size_t kInteriorOffset = 128;

bool IsDeviceMemoryType(hipMemoryType type) {
  return type == hipMemoryTypeDevice || type == hipMemoryTypeUnified;
}

bool IsHostMemoryType(hipMemoryType type) {
  return type == hipMemoryTypeHost || type == hipMemoryTypeUnified;
}

bool IsManagedMemoryType(hipMemoryType type) {
  return type == hipMemoryTypeManaged || type == hipMemoryTypeUnified;
}

bool ManagedMemorySupported() {
  void* ptr = nullptr;
  const hipError_t status = hipMallocManaged(&ptr, sizeof(int), hipMemAttachGlobal);
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
    HIP_SKIP_TEST("hipMallocManaged is not supported by this device/runtime path.");
  }
}

bool QueryPointerAttributeOrSkip(void* data, hipPointer_attribute attribute, hipDeviceptr_t ptr) {
  const hipError_t status = hipPointerGetAttribute(data, attribute, ptr);
  if (status == hipErrorNotSupported) {
    return false;
  }
  HIP_CHECK(status);
  return true;
}
}  // namespace

HIP_TEST_CASE(Contract_PointerInfo_GetAttributes_DeviceAllocation_ReportsDeviceType) {
  int device = 0;
  HIP_CHECK(hipGetDevice(&device));

  void* data = nullptr;
  HIP_CHECK(hipMalloc(&data, kAllocationBytes));

  hipPointerAttribute_t attributes{};
  HIP_CHECK(hipPointerGetAttributes(&attributes, data));

  REQUIRE(IsDeviceMemoryType(attributes.type));
  REQUIRE(attributes.device == device);
  REQUIRE(attributes.devicePointer == data);

  HIP_CHECK(hipFree(data));
}

HIP_TEST_CASE(Contract_PointerInfo_GetAttributes_HostAllocation_ReportsHostType) {
  void* data = nullptr;
  HIP_CHECK(hipHostMalloc(&data, kAllocationBytes, hipHostMallocDefault));

  hipPointerAttribute_t attributes{};
  HIP_CHECK(hipPointerGetAttributes(&attributes, data));

  REQUIRE(IsHostMemoryType(attributes.type));
  REQUIRE(attributes.hostPointer == data);

  HIP_CHECK(hipHostFree(data));
}

HIP_TEST_CASE(Contract_PointerInfo_GetAttributes_ManagedAllocation_ReportsManagedOrUnified) {
  SkipIfManagedMemoryUnsupported();

  int device = 0;
  HIP_CHECK(hipGetDevice(&device));

  void* data = nullptr;
  HIP_CHECK(hipMallocManaged(&data, kAllocationBytes, hipMemAttachGlobal));

  hipPointerAttribute_t attributes{};
  HIP_CHECK(hipPointerGetAttributes(&attributes, data));

  REQUIRE(IsManagedMemoryType(attributes.type));
  REQUIRE(attributes.device == device);

  HIP_CHECK(hipFree(data));
}

HIP_TEST_CASE(Contract_PointerInfo_GetAttribute_MemoryType_MatchesGetAttributes) {
  void* data = nullptr;
  HIP_CHECK(hipMalloc(&data, kAllocationBytes));

  hipPointerAttribute_t attributes{};
  HIP_CHECK(hipPointerGetAttributes(&attributes, data));

  unsigned int memory_type = 0;
  if (!QueryPointerAttributeOrSkip(&memory_type, HIP_POINTER_ATTRIBUTE_MEMORY_TYPE,
                                   reinterpret_cast<hipDeviceptr_t>(data))) {
    HIP_CHECK(hipFree(data));
    HIP_SKIP_TEST("HIP_POINTER_ATTRIBUTE_MEMORY_TYPE query is not supported by this runtime path.");
  }

  REQUIRE(IsDeviceMemoryType(static_cast<hipMemoryType>(memory_type)));
  REQUIRE(IsDeviceMemoryType(attributes.type));

  HIP_CHECK(hipFree(data));
}

HIP_TEST_CASE(Contract_PointerInfo_MemGetAddressRange_ReturnsBaseAndSize) {
  char* data = nullptr;
  HIP_CHECK(hipMalloc(&data, kAllocationBytes));

  hipDeviceptr_t base = 0;
  size_t size = 0;
  HIP_CHECK(hipMemGetAddressRange(&base, &size,
                                  reinterpret_cast<hipDeviceptr_t>(data + kInteriorOffset)));

  REQUIRE(base == reinterpret_cast<hipDeviceptr_t>(data));
  REQUIRE(size >= kAllocationBytes);
  REQUIRE(reinterpret_cast<std::uintptr_t>(data + kInteriorOffset) <
          reinterpret_cast<std::uintptr_t>(base) + size);

  HIP_CHECK(hipFree(data));
}

HIP_TEST_CASE(Contract_PointerInfo_MemGetInfo_FreeNotGreaterThanTotal) {
  size_t free_bytes = 0;
  size_t total_bytes = 0;

  HIP_CHECK(hipMemGetInfo(&free_bytes, &total_bytes));

  REQUIRE(total_bytes > 0);
  REQUIRE(free_bytes <= total_bytes);
}

HIP_TEST_CASE(Contract_PointerInfo_GetAttributes_NullOutput_IsRejected) {
  void* data = nullptr;
  HIP_CHECK(hipMalloc(&data, sizeof(int)));

  REQUIRE(hipPointerGetAttributes(nullptr, data) != hipSuccess);

  HIP_CHECK(hipFree(data));
}
