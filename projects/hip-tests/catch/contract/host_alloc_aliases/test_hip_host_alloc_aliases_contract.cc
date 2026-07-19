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
constexpr size_t kElementCount = 64;

std::array<uint8_t, kElementCount> MakePattern(uint8_t seed) {
  std::array<uint8_t, kElementCount> pattern{};
  for (size_t i = 0; i < pattern.size(); ++i) {
    pattern[i] = static_cast<uint8_t>(seed + i);
  }
  return pattern;
}

void WriteAndVerifyHostBytes(void* host_ptr, const std::array<uint8_t, kElementCount>& pattern) {
  auto* bytes = static_cast<uint8_t*>(host_ptr);
  for (size_t i = 0; i < pattern.size(); ++i) {
    bytes[i] = pattern[i];
  }
  for (size_t i = 0; i < pattern.size(); ++i) {
    REQUIRE(bytes[i] == pattern[i]);
  }
}
}  // namespace

// @asserts: hipHostAlloc - returns a non-null pinned host pointer that is host-readable/writable
HIP_TEST_CASE(Contract_HostAllocAliases_HostAlloc_ReturnsUsablePinnedPointer) {
  hip::contract::ContractCleanup cleanup;
  void* host_ptr = nullptr;
  const auto pattern = MakePattern(0x12);

  HIP_CHECK(hipHostAlloc(&host_ptr, kElementCount, hipHostAllocDefault));
  cleanup.Add([host_ptr] { (void)hipFreeHost(host_ptr); });

  REQUIRE(host_ptr != nullptr);
  WriteAndVerifyHostBytes(host_ptr, pattern);
}

// @asserts: hipMallocHost - alias returns a non-null host pointer that is host-readable/writable
HIP_TEST_CASE(Contract_HostAllocAliases_MallocHost_ReturnsUsablePointer) {
  hip::contract::ContractCleanup cleanup;
  void* host_ptr = nullptr;
  const auto pattern = MakePattern(0x34);

  HIP_CHECK(hipMallocHost(&host_ptr, kElementCount));
  cleanup.Add([host_ptr] { (void)hipFreeHost(host_ptr); });

  REQUIRE(host_ptr != nullptr);
  WriteAndVerifyHostBytes(host_ptr, pattern);
}

// @asserts: hipMemAllocHost - driver-API alias returns a non-null host pointer that is host-readable/writable
HIP_TEST_CASE(Contract_HostAllocAliases_MemAllocHost_ReturnsUsablePointer) {
  hip::contract::ContractCleanup cleanup;
  void* host_ptr = nullptr;
  const auto pattern = MakePattern(0x56);

  // hipMemAllocHost is the driver-API alias (cuMemAllocHost on NVIDIA) and needs
  // an initialized primary context; prime one so the test passes even when it is
  // the first HIP call in the process (e.g. run in isolation under ctest). The
  // runtime-API siblings above auto-initialize, so only this variant needs it.
  HIP_CHECK(hipFree(0));
  HIP_CHECK(hipMemAllocHost(&host_ptr, kElementCount));
  cleanup.Add([host_ptr] { (void)hipFreeHost(host_ptr); });

  REQUIRE(host_ptr != nullptr);
  WriteAndVerifyHostBytes(host_ptr, pattern);
}

// @asserts: hipFreeHost - accepts a null pointer as success but rejects a non-pinned (stack) pointer with non-success
HIP_TEST_CASE(Contract_HostAllocAliases_FreeHost_NullSucceeds_InvalidPointerRejected) {
  int stack_value = 0;

  HIP_CHECK(hipFreeHost(nullptr));
  REQUIRE(hipFreeHost(&stack_value) != hipSuccess);
}

// @asserts: hipHostAlloc - the pinned-host alloc aliases reject a null out-pointer and an invalid flag with non-success
HIP_TEST_CASE(Contract_HostAllocAliases_InvalidArgs_AreRejected) {
  void* host_ptr = nullptr;

  REQUIRE(hipHostAlloc(nullptr, kElementCount, hipHostAllocDefault) != hipSuccess);
  REQUIRE(hipMallocHost(nullptr, kElementCount) != hipSuccess);
  REQUIRE(hipMemAllocHost(nullptr, kElementCount) != hipSuccess);
  REQUIRE(hipHostAlloc(&host_ptr, kElementCount, 0x7fffffff) != hipSuccess);
}

// BACKEND-DIFF: hipExtMallocWithFlags and the hipDeviceMallocDefault flag are
// AMD extensions with no NVIDIA-backend equivalent, so these two contracts build
// only on AMD. The pinned-host-allocation aliases above are portable. Parity
// would require NVIDIA to expose the ext malloc entry point and flag.
#if HT_AMD
// @asserts: hipExtMallocWithFlags - default flag allocates a non-null device pointer and a zero-size request yields null (AMD only)
HIP_TEST_CASE(Contract_HostAllocAliases_ExtMallocWithFlags_DefaultAllocatesAndZeroSizeIsNull) {
  void* device_ptr = nullptr;

  HIP_CHECK(hipExtMallocWithFlags(&device_ptr, kElementCount, hipDeviceMallocDefault));
  REQUIRE(device_ptr != nullptr);
  HIP_CHECK(hipFree(device_ptr));

  device_ptr = reinterpret_cast<void*>(0x1);
  HIP_CHECK(hipExtMallocWithFlags(&device_ptr, 0, hipDeviceMallocDefault));
  REQUIRE(device_ptr == nullptr);
}

// @asserts: hipExtMallocWithFlags - rejects a null out-pointer and an invalid flag with non-success (AMD only)
HIP_TEST_CASE(Contract_HostAllocAliases_ExtMallocWithFlags_InvalidArgs_AreRejected) {
  void* device_ptr = nullptr;

  REQUIRE(hipExtMallocWithFlags(nullptr, kElementCount, hipDeviceMallocDefault) != hipSuccess);
  REQUIRE(hipExtMallocWithFlags(&device_ptr, kElementCount, 0x1000) != hipSuccess);
}
#endif  // HT_AMD
