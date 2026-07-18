/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <cstddef>
#include <cstdint>

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>
#include <contract_cleanup.hh>

namespace {
constexpr size_t kAllocationBytes = 4096;
constexpr size_t kInteriorOffset = 128;

// Converts a driver device pointer to an integer address for range comparisons.
// hipDeviceptr_t is a pointer (void*) on AMD but an integer (unsigned long long)
// on the NVIDIA backend, so a single reinterpret_cast is not portable: cast
// through the pointer/integer form each backend actually uses.
std::uintptr_t DevPtrToUint(hipDeviceptr_t p) {
#if HT_AMD
  return reinterpret_cast<std::uintptr_t>(p);
#else
  return static_cast<std::uintptr_t>(p);
#endif
}

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
  hip::contract::ContractCleanup cleanup;
  int device = 0;
  HIP_CHECK(hipGetDevice(&device));

  void* data = nullptr;
  HIP_CHECK(hipMalloc(&data, kAllocationBytes));
  cleanup.Add([&] { (void)hipFree(data); });

  hipPointerAttribute_t attributes{};
  HIP_CHECK(hipPointerGetAttributes(&attributes, data));

  REQUIRE(IsDeviceMemoryType(attributes.type));
  REQUIRE(attributes.device == device);
  REQUIRE(attributes.devicePointer == data);
}

HIP_TEST_CASE(Contract_PointerInfo_GetAttributes_HostAllocation_ReportsHostType) {
  hip::contract::ContractCleanup cleanup;
  void* data = nullptr;
  HIP_CHECK(hipHostMalloc(&data, kAllocationBytes, hipHostMallocDefault));
  cleanup.Add([&] { (void)hipHostFree(data); });

  hipPointerAttribute_t attributes{};
  HIP_CHECK(hipPointerGetAttributes(&attributes, data));

  REQUIRE(IsHostMemoryType(attributes.type));
  REQUIRE(attributes.hostPointer == data);
}

HIP_TEST_CASE(Contract_PointerInfo_GetAttributes_ManagedAllocation_ReportsManagedOrUnified) {
  SkipIfManagedMemoryUnsupported();
  hip::contract::ContractCleanup cleanup;

  int device = 0;
  HIP_CHECK(hipGetDevice(&device));

  void* data = nullptr;
  HIP_CHECK(hipMallocManaged(&data, kAllocationBytes, hipMemAttachGlobal));
  cleanup.Add([&] { (void)hipFree(data); });

  hipPointerAttribute_t attributes{};
  HIP_CHECK(hipPointerGetAttributes(&attributes, data));

  REQUIRE(IsManagedMemoryType(attributes.type));
  REQUIRE(attributes.device == device);
}

HIP_TEST_CASE(Contract_PointerInfo_GetAttribute_MemoryType_MatchesGetAttributes) {
  hip::contract::ContractCleanup cleanup;
  void* data = nullptr;
  HIP_CHECK(hipMalloc(&data, kAllocationBytes));
  cleanup.Add([&] { (void)hipFree(data); });

  hipPointerAttribute_t attributes{};
  HIP_CHECK(hipPointerGetAttributes(&attributes, data));

  unsigned int memory_type = 0;
  if (!QueryPointerAttributeOrSkip(&memory_type, HIP_POINTER_ATTRIBUTE_MEMORY_TYPE,
                                   reinterpret_cast<hipDeviceptr_t>(data))) {
    HIP_SKIP_TEST("HIP_POINTER_ATTRIBUTE_MEMORY_TYPE query is not supported by this runtime path.");
  }

  REQUIRE(IsDeviceMemoryType(static_cast<hipMemoryType>(memory_type)));
  REQUIRE(IsDeviceMemoryType(attributes.type));
}

HIP_TEST_CASE(Contract_PointerInfo_MemGetAddressRange_ReturnsBaseAndSize) {
  hip::contract::ContractCleanup cleanup;
  char* data = nullptr;
  HIP_CHECK(hipMalloc(&data, kAllocationBytes));
  cleanup.Add([&] { (void)hipFree(data); });

  hipDeviceptr_t base = 0;
  size_t size = 0;
  HIP_CHECK(hipMemGetAddressRange(&base, &size,
                                  reinterpret_cast<hipDeviceptr_t>(data + kInteriorOffset)));

  REQUIRE(base == reinterpret_cast<hipDeviceptr_t>(data));
  REQUIRE(size >= kAllocationBytes);
  REQUIRE(reinterpret_cast<std::uintptr_t>(data + kInteriorOffset) <
          DevPtrToUint(base) + size);
}

HIP_TEST_CASE(Contract_PointerInfo_MemGetInfo_FreeNotGreaterThanTotal) {
  size_t free_bytes = 0;
  size_t total_bytes = 0;

  HIP_CHECK(hipMemGetInfo(&free_bytes, &total_bytes));

  REQUIRE(total_bytes > 0);
  REQUIRE(free_bytes <= total_bytes);
}

HIP_TEST_CASE(Contract_PointerInfo_GetAttributes_NullOutput_IsRejected) {
  // BACKEND-DIFF: The null-output rejection contract is only exercised on AMD. On
  // NVIDIA hipPointerGetAttributes maps to cudaPointerGetAttributes, which does
  // not validate the output-attributes pointer and dereferences it - a null
  // output faults (SIGSEGV) instead of returning a defined error - so the
  // rejection cannot be evaluated safely there. Parity would require matching
  // null-output validation.
#if HT_AMD
  hip::contract::ContractCleanup cleanup;
  void* data = nullptr;
  HIP_CHECK(hipMalloc(&data, sizeof(int)));
  cleanup.Add([&] { (void)hipFree(data); });

  const hipError_t status = hipPointerGetAttributes(nullptr, data);
  REQUIRE(status != hipSuccess);
  (void)hipGetLastError();
#else
  HIP_SKIP_TEST("hipPointerGetAttributes does not validate the output pointer on the NVIDIA "
                "backend; the null-output rejection contract cannot be exercised safely.");
#endif
}
