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
constexpr size_t kElementCount = 256;

std::array<uint8_t, kElementCount> MakePattern(uint8_t seed) {
  std::array<uint8_t, kElementCount> pattern{};
  for (size_t i = 0; i < pattern.size(); ++i) {
    pattern[i] = static_cast<uint8_t>(seed + i);
  }
  return pattern;
}
}

HIP_TEST_CASE(Contract_Transfer_HostToDeviceToHost_RoundTripsBytes) {
  hip::contract::ContractCleanup cleanup;
  const auto src = MakePattern(0x31);
  std::array<uint8_t, kElementCount> dst{};
  void* device_ptr = nullptr;

  HIP_CHECK(hipMalloc(&device_ptr, src.size()));
  cleanup.Add([&] { (void)hipFree(device_ptr); });

  HIP_CHECK(hipMemcpy(device_ptr, src.data(), src.size(), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dst.data(), device_ptr, dst.size(), hipMemcpyDeviceToHost));

  REQUIRE(dst == src);
}

HIP_TEST_CASE(Contract_Transfer_DeviceToDevice_CopiesBytes) {
  hip::contract::ContractCleanup cleanup;
  const auto src = MakePattern(0x67);
  std::array<uint8_t, kElementCount> dst{};
  void* src_device_ptr = nullptr;
  void* dst_device_ptr = nullptr;

  HIP_CHECK(hipMalloc(&src_device_ptr, src.size()));
  cleanup.Add([&] { (void)hipFree(src_device_ptr); });
  HIP_CHECK(hipMalloc(&dst_device_ptr, dst.size()));
  cleanup.Add([&] { (void)hipFree(dst_device_ptr); });

  HIP_CHECK(hipMemcpy(src_device_ptr, src.data(), src.size(), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dst_device_ptr, src_device_ptr, src.size(), hipMemcpyDeviceToDevice));
  HIP_CHECK(hipMemcpy(dst.data(), dst_device_ptr, dst.size(), hipMemcpyDeviceToHost));

  REQUIRE(dst == src);
}

HIP_TEST_CASE(Contract_Transfer_MemcpyZeroBytes_Succeeds) {
  uint8_t src = 0x1;
  uint8_t dst = 0x2;

  HIP_CHECK(hipMemcpy(&dst, &src, 0, hipMemcpyHostToHost));

  REQUIRE(dst == 0x2);
}

HIP_TEST_CASE(Contract_Transfer_InvalidDirection_ReturnsInvalidMemcpyDirection) {
  hip::contract::ContractCleanup cleanup;
  const auto src = MakePattern(0x9a);
  void* device_ptr = nullptr;

  HIP_CHECK(hipMalloc(&device_ptr, src.size()));
  cleanup.Add([&] { (void)hipFree(device_ptr); });
  HIP_CHECK_ERROR(hipMemcpy(device_ptr, src.data(), src.size(), static_cast<hipMemcpyKind>(-1)),
                  hipErrorInvalidMemcpyDirection);
}
