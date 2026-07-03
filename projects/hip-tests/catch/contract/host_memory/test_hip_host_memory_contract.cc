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
constexpr size_t kElementCount = 128;

std::array<uint8_t, kElementCount> MakePattern(uint8_t seed) {
  std::array<uint8_t, kElementCount> pattern{};
  for (size_t i = 0; i < pattern.size(); ++i) {
    pattern[i] = static_cast<uint8_t>(seed + i);
  }
  return pattern;
}
}

HIP_TEST_CASE(Contract_HostMemory_HostMalloc_ReturnsUsablePointer) {
  void* host_ptr = nullptr;

  HIP_CHECK(hipHostMalloc(&host_ptr, kElementCount, hipHostMallocDefault));

  REQUIRE(host_ptr != nullptr);

  auto* bytes = static_cast<uint8_t*>(host_ptr);
  const auto pattern = MakePattern(0x19);
  for (size_t i = 0; i < pattern.size(); ++i) {
    bytes[i] = pattern[i];
  }
  for (size_t i = 0; i < pattern.size(); ++i) {
    REQUIRE(bytes[i] == pattern[i]);
  }

  HIP_CHECK(hipHostFree(host_ptr));
}

HIP_TEST_CASE(Contract_HostMemory_HostFree_Succeeds) {
  void* host_ptr = nullptr;

  HIP_CHECK(hipHostMalloc(&host_ptr, kElementCount, hipHostMallocDefault));
  HIP_CHECK(hipHostFree(host_ptr));
}

HIP_TEST_CASE(Contract_HostMemory_HostRegisterUnregister_Succeeds) {
  std::array<uint8_t, kElementCount> host_buffer{};

  HIP_CHECK(hipHostRegister(host_buffer.data(), host_buffer.size(), hipHostRegisterDefault));
  HIP_CHECK(hipHostUnregister(host_buffer.data()));
}

HIP_TEST_CASE(Contract_HostMemory_HostGetDevicePointer_RoundTripsBytes) {
  const auto src = MakePattern(0x41);
  std::array<uint8_t, kElementCount> dst{};
  void* host_ptr = nullptr;
  void* device_visible_ptr = nullptr;

  HIP_CHECK(hipHostMalloc(&host_ptr, src.size(), hipHostMallocMapped));
  HIP_CHECK(hipHostGetDevicePointer(&device_visible_ptr, host_ptr, 0));

  REQUIRE(device_visible_ptr != nullptr);

  HIP_CHECK(hipMemcpy(device_visible_ptr, src.data(), src.size(), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dst.data(), device_visible_ptr, dst.size(), hipMemcpyDeviceToHost));

  REQUIRE(dst == src);

  HIP_CHECK(hipHostFree(host_ptr));
}

HIP_TEST_CASE(Contract_HostMemory_HostGetFlags_IncludesRequestedFlags) {
  void* host_ptr = nullptr;
  unsigned int flags = 0;
  constexpr unsigned int requested_flags = hipHostMallocMapped;

  HIP_CHECK(hipHostMalloc(&host_ptr, kElementCount, requested_flags));
  HIP_CHECK(hipHostGetFlags(&flags, host_ptr));

  REQUIRE((flags & requested_flags) == requested_flags);

  HIP_CHECK(hipHostFree(host_ptr));
}
