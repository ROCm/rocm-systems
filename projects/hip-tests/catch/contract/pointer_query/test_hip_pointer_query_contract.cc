/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <cstddef>

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>
#include <contract_cleanup.hh>

namespace {
constexpr size_t kAllocationBytes = 4096;

bool IsDeviceMemoryType(unsigned int type) {
  return type == hipMemoryTypeDevice || type == hipMemoryTypeUnified;
}

bool PointerSetAttributeOrSkip(const void* value, hipPointer_attribute attribute, hipDeviceptr_t ptr) {
  const hipError_t status = hipPointerSetAttribute(value, attribute, ptr);
  if (status == hipErrorNotSupported) {
    return false;
  }
  HIP_CHECK(status);
  return true;
}
}  // namespace

HIP_TEST_CASE(Contract_PointerQuery_DrvGetAttributes_DeviceAllocation_ReportsTypeAndOrdinal) {
  hip::contract::ContractCleanup cleanup;
  int current_device = 0;
  HIP_CHECK(hipGetDevice(&current_device));

  void* data = nullptr;
  HIP_CHECK(hipMalloc(&data, kAllocationBytes));
  cleanup.Add([&] { (void)hipFree(data); });

  unsigned int memory_type = 0;
  int device_ordinal = -1;
  void* results[] = {&memory_type, &device_ordinal};
  hipPointer_attribute attributes[] = {HIP_POINTER_ATTRIBUTE_MEMORY_TYPE,
                                       HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL};

  HIP_CHECK(hipDrvPointerGetAttributes(2, attributes, results,
                                       reinterpret_cast<hipDeviceptr_t>(data)));

  REQUIRE(IsDeviceMemoryType(memory_type));
  REQUIRE(device_ordinal == current_device);
}

HIP_TEST_CASE(Contract_PointerQuery_DrvGetAttributes_MatchesSingleAttributeQuery) {
  hip::contract::ContractCleanup cleanup;
  void* data = nullptr;
  HIP_CHECK(hipMalloc(&data, kAllocationBytes));
  cleanup.Add([&] { (void)hipFree(data); });

  void* batch_device_pointer = nullptr;
  void* results[] = {&batch_device_pointer};
  hipPointer_attribute attributes[] = {HIP_POINTER_ATTRIBUTE_DEVICE_POINTER};

  HIP_CHECK(hipDrvPointerGetAttributes(1, attributes, results,
                                       reinterpret_cast<hipDeviceptr_t>(data)));

  void* scalar_device_pointer = nullptr;
  HIP_CHECK(hipPointerGetAttribute(&scalar_device_pointer, HIP_POINTER_ATTRIBUTE_DEVICE_POINTER,
                                   reinterpret_cast<hipDeviceptr_t>(data)));

  REQUIRE(batch_device_pointer == scalar_device_pointer);
}

HIP_TEST_CASE(Contract_PointerQuery_DrvGetAttributes_InvalidArgs_AreRejected) {
  hip::contract::ContractCleanup cleanup;
  void* data = nullptr;
  HIP_CHECK(hipMalloc(&data, kAllocationBytes));
  cleanup.Add([&] { (void)hipFree(data); });

  unsigned int memory_type = 0;
  void* results[] = {&memory_type};
  hipPointer_attribute attributes[] = {HIP_POINTER_ATTRIBUTE_MEMORY_TYPE};

  REQUIRE(hipDrvPointerGetAttributes(0, attributes, results,
                                     reinterpret_cast<hipDeviceptr_t>(data)) != hipSuccess);
  REQUIRE(hipDrvPointerGetAttributes(1, nullptr, results,
                                     reinterpret_cast<hipDeviceptr_t>(data)) != hipSuccess);
  REQUIRE(hipDrvPointerGetAttributes(1, attributes, nullptr,
                                     reinterpret_cast<hipDeviceptr_t>(data)) != hipSuccess);
}

HIP_TEST_CASE(Contract_PointerQuery_MemPtrGetInfo_ReturnsAllocationSize) {
  hip::contract::ContractCleanup cleanup;
  void* data = nullptr;
  HIP_CHECK(hipMalloc(&data, kAllocationBytes));
  cleanup.Add([&] { (void)hipFree(data); });

  size_t size = 0;
  HIP_CHECK(hipMemPtrGetInfo(data, &size));

  REQUIRE(size >= kAllocationBytes);
}

HIP_TEST_CASE(Contract_PointerQuery_SetAttribute_SyncMemops_SucceedsWhenSupported) {
  hip::contract::ContractCleanup cleanup;
  void* data = nullptr;
  HIP_CHECK(hipMalloc(&data, kAllocationBytes));
  cleanup.Add([&] { (void)hipFree(data); });

  int value = 1;
  if (!PointerSetAttributeOrSkip(&value, HIP_POINTER_ATTRIBUTE_SYNC_MEMOPS,
                                 reinterpret_cast<hipDeviceptr_t>(data))) {
    HIP_SKIP_TEST("HIP_POINTER_ATTRIBUTE_SYNC_MEMOPS is not supported by this runtime path.");
  }
}

HIP_TEST_CASE(Contract_PointerQuery_SetAttribute_InvalidArgs_AreRejected) {
  hip::contract::ContractCleanup cleanup;
  void* data = nullptr;
  HIP_CHECK(hipMalloc(&data, kAllocationBytes));
  cleanup.Add([&] { (void)hipFree(data); });

  int value = 0;
  REQUIRE(hipPointerSetAttribute(nullptr, HIP_POINTER_ATTRIBUTE_SYNC_MEMOPS,
                                 reinterpret_cast<hipDeviceptr_t>(data)) != hipSuccess);
  REQUIRE(hipPointerSetAttribute(&value, static_cast<hipPointer_attribute>(0x7fffffff),
                                 reinterpret_cast<hipDeviceptr_t>(data)) != hipSuccess);
  REQUIRE(hipPointerSetAttribute(&value, HIP_POINTER_ATTRIBUTE_SYNC_MEMOPS, nullptr) !=
          hipSuccess);
}
