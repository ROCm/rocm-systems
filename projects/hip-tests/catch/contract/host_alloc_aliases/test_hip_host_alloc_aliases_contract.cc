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

HIP_TEST_CASE(Contract_HostAllocAliases_HostAlloc_ReturnsUsablePinnedPointer) {
  void* host_ptr = nullptr;
  const auto pattern = MakePattern(0x12);

  HIP_CHECK(hipHostAlloc(&host_ptr, kElementCount, hipHostAllocDefault));

  REQUIRE(host_ptr != nullptr);
  WriteAndVerifyHostBytes(host_ptr, pattern);

  HIP_CHECK(hipFreeHost(host_ptr));
}

HIP_TEST_CASE(Contract_HostAllocAliases_MallocHost_ReturnsUsablePointer) {
  void* host_ptr = nullptr;
  const auto pattern = MakePattern(0x34);

  HIP_CHECK(hipMallocHost(&host_ptr, kElementCount));

  REQUIRE(host_ptr != nullptr);
  WriteAndVerifyHostBytes(host_ptr, pattern);

  HIP_CHECK(hipFreeHost(host_ptr));
}

HIP_TEST_CASE(Contract_HostAllocAliases_MemAllocHost_ReturnsUsablePointer) {
  void* host_ptr = nullptr;
  const auto pattern = MakePattern(0x56);

  HIP_CHECK(hipMemAllocHost(&host_ptr, kElementCount));

  REQUIRE(host_ptr != nullptr);
  WriteAndVerifyHostBytes(host_ptr, pattern);

  HIP_CHECK(hipFreeHost(host_ptr));
}

HIP_TEST_CASE(Contract_HostAllocAliases_FreeHost_NullSucceeds_InvalidPointerRejected) {
  int stack_value = 0;

  HIP_CHECK(hipFreeHost(nullptr));
  REQUIRE(hipFreeHost(&stack_value) != hipSuccess);
}

HIP_TEST_CASE(Contract_HostAllocAliases_InvalidArgs_AreRejected) {
  void* host_ptr = nullptr;

  REQUIRE(hipHostAlloc(nullptr, kElementCount, hipHostAllocDefault) != hipSuccess);
  REQUIRE(hipMallocHost(nullptr, kElementCount) != hipSuccess);
  REQUIRE(hipMemAllocHost(nullptr, kElementCount) != hipSuccess);
  REQUIRE(hipHostAlloc(&host_ptr, kElementCount, 0x7fffffff) != hipSuccess);
}

HIP_TEST_CASE(Contract_HostAllocAliases_ExtMallocWithFlags_DefaultAllocatesAndZeroSizeIsNull) {
  void* device_ptr = nullptr;

  HIP_CHECK(hipExtMallocWithFlags(&device_ptr, kElementCount, hipDeviceMallocDefault));
  REQUIRE(device_ptr != nullptr);
  HIP_CHECK(hipFree(device_ptr));

  device_ptr = reinterpret_cast<void*>(0x1);
  HIP_CHECK(hipExtMallocWithFlags(&device_ptr, 0, hipDeviceMallocDefault));
  REQUIRE(device_ptr == nullptr);
}

HIP_TEST_CASE(Contract_HostAllocAliases_ExtMallocWithFlags_InvalidArgs_AreRejected) {
  void* device_ptr = nullptr;

  REQUIRE(hipExtMallocWithFlags(nullptr, kElementCount, hipDeviceMallocDefault) != hipSuccess);
  REQUIRE(hipExtMallocWithFlags(&device_ptr, kElementCount, 0x1000) != hipSuccess);
}
