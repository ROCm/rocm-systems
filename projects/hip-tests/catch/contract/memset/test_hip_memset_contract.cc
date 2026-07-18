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
constexpr size_t kByteCount = 256;
constexpr size_t kWordCount = 128;
constexpr size_t kDwordCount = 64;

void* AllocateDeviceBytes(size_t bytes) {
  void* ptr = nullptr;
  HIP_CHECK(hipMalloc(&ptr, bytes));
  return ptr;
}
}

HIP_TEST_CASE(Contract_Memset_DeviceBuffer_IsFilledWithBytePattern) {
  hip::contract::ContractCleanup cleanup;
  constexpr uint8_t pattern = 0x5a;
  std::array<uint8_t, kByteCount> dst{};
  void* device_ptr = AllocateDeviceBytes(dst.size());
  cleanup.Add([device_ptr] { (void)hipFree(device_ptr); });

  HIP_CHECK(hipMemset(device_ptr, pattern, dst.size()));
  HIP_CHECK(hipMemcpy(dst.data(), device_ptr, dst.size(), hipMemcpyDeviceToHost));

  for (const auto value : dst) {
    REQUIRE(value == pattern);
  }
}

HIP_TEST_CASE(Contract_Memset_ZeroBytes_Succeeds) {
  hip::contract::ContractCleanup cleanup;
  uint8_t value = 0;
  void* device_ptr = AllocateDeviceBytes(sizeof(value));
  cleanup.Add([device_ptr] { (void)hipFree(device_ptr); });

  HIP_CHECK(hipMemset(device_ptr, 0x7f, 0));
}

HIP_TEST_CASE(Contract_MemsetAsync_DeviceBuffer_IsFilledAfterStreamSynchronize) {
  hip::contract::ContractCleanup cleanup;
  constexpr uint8_t pattern = 0xa5;
  std::array<uint8_t, kByteCount> dst{};
  void* device_ptr = AllocateDeviceBytes(dst.size());
  cleanup.Add([device_ptr] { (void)hipFree(device_ptr); });
  hipStream_t stream = nullptr;

  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([stream] { (void)hipStreamDestroy(stream); });
  HIP_CHECK(hipMemsetAsync(device_ptr, pattern, dst.size(), stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipMemcpy(dst.data(), device_ptr, dst.size(), hipMemcpyDeviceToHost));

  for (const auto value : dst) {
    REQUIRE(value == pattern);
  }
}

HIP_TEST_CASE(Contract_MemsetD8_DeviceBuffer_IsFilledWithBytePattern) {
  hip::contract::ContractCleanup cleanup;
  constexpr uint8_t pattern = 0x3c;
  std::array<uint8_t, kByteCount> dst{};
  void* device_ptr = AllocateDeviceBytes(dst.size());
  cleanup.Add([device_ptr] { (void)hipFree(device_ptr); });

  HIP_CHECK(hipMemsetD8(reinterpret_cast<hipDeviceptr_t>(device_ptr), pattern, dst.size()));
  HIP_CHECK(hipMemcpy(dst.data(), device_ptr, dst.size(), hipMemcpyDeviceToHost));

  for (const auto value : dst) {
    REQUIRE(value == pattern);
  }
}

HIP_TEST_CASE(Contract_MemsetD16_DeviceBuffer_IsFilledWithWordPattern) {
  hip::contract::ContractCleanup cleanup;
  constexpr uint16_t pattern = 0x1357;
  std::array<uint16_t, kWordCount> dst{};
  void* device_ptr = AllocateDeviceBytes(dst.size() * sizeof(uint16_t));
  cleanup.Add([device_ptr] { (void)hipFree(device_ptr); });

  HIP_CHECK(hipMemsetD16(reinterpret_cast<hipDeviceptr_t>(device_ptr), pattern, dst.size()));
  HIP_CHECK(hipMemcpy(dst.data(), device_ptr, dst.size() * sizeof(uint16_t), hipMemcpyDeviceToHost));

  for (const auto value : dst) {
    REQUIRE(value == pattern);
  }
}

HIP_TEST_CASE(Contract_MemsetD32_DeviceBuffer_IsFilledWithDwordPattern) {
  hip::contract::ContractCleanup cleanup;
  constexpr int pattern = 0x12345678;
  std::array<int, kDwordCount> dst{};
  void* device_ptr = AllocateDeviceBytes(dst.size() * sizeof(int));
  cleanup.Add([device_ptr] { (void)hipFree(device_ptr); });

  HIP_CHECK(hipMemsetD32(reinterpret_cast<hipDeviceptr_t>(device_ptr), pattern, dst.size()));
  HIP_CHECK(hipMemcpy(dst.data(), device_ptr, dst.size() * sizeof(int), hipMemcpyDeviceToHost));

  for (const auto value : dst) {
    REQUIRE(value == pattern);
  }
}
