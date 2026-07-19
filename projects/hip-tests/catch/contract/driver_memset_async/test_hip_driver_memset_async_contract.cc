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

template <typename T, size_t N>
void RequireAllEqual(const std::array<T, N>& values, T expected) {
  for (const auto value : values) {
    REQUIRE(value == expected);
  }
}
}  // namespace

// @asserts: hipMemsetD8Async - fills device memory with the byte pattern, visible on the host after stream sync
HIP_TEST_CASE(Contract_DriverMemsetAsync_D8_FillsByte_VisibleAfterSync) {
  hip::contract::ContractCleanup cleanup;
  constexpr uint8_t pattern = 0x5a;
  std::array<uint8_t, kByteCount> dst{};
  void* device_ptr = nullptr;
  hipStream_t stream = nullptr;

  HIP_CHECK(hipMalloc(&device_ptr, dst.size()));
  cleanup.Add([device_ptr] { (void)hipFree(device_ptr); });
  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([stream] { (void)hipStreamDestroy(stream); });

  HIP_CHECK(hipMemsetD8Async(reinterpret_cast<hipDeviceptr_t>(device_ptr), pattern, dst.size(),
                             stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipMemcpyDtoH(dst.data(), reinterpret_cast<hipDeviceptr_t>(device_ptr), dst.size()));

  RequireAllEqual(dst, pattern);
}

// @asserts: hipMemsetD16Async - fills device memory with the 16-bit pattern, visible on the host after stream sync
HIP_TEST_CASE(Contract_DriverMemsetAsync_D16_FillsWord_VisibleAfterSync) {
  hip::contract::ContractCleanup cleanup;
  constexpr uint16_t pattern = 0x1357;
  std::array<uint16_t, kWordCount> dst{};
  void* device_ptr = nullptr;
  hipStream_t stream = nullptr;

  HIP_CHECK(hipMalloc(&device_ptr, dst.size() * sizeof(uint16_t)));
  cleanup.Add([device_ptr] { (void)hipFree(device_ptr); });
  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([stream] { (void)hipStreamDestroy(stream); });

  HIP_CHECK(hipMemsetD16Async(reinterpret_cast<hipDeviceptr_t>(device_ptr), pattern, dst.size(),
                              stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipMemcpyDtoH(dst.data(), reinterpret_cast<hipDeviceptr_t>(device_ptr),
                          dst.size() * sizeof(uint16_t)));

  RequireAllEqual(dst, pattern);
}

// @asserts: hipMemsetD32Async - fills device memory with the 32-bit pattern, visible on the host after stream sync
HIP_TEST_CASE(Contract_DriverMemsetAsync_D32_FillsDword_VisibleAfterSync) {
  hip::contract::ContractCleanup cleanup;
  constexpr int pattern = 0x12345678;
  std::array<int, kDwordCount> dst{};
  void* device_ptr = nullptr;
  hipStream_t stream = nullptr;

  HIP_CHECK(hipMalloc(&device_ptr, dst.size() * sizeof(int)));
  cleanup.Add([device_ptr] { (void)hipFree(device_ptr); });
  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([stream] { (void)hipStreamDestroy(stream); });

  HIP_CHECK(hipMemsetD32Async(reinterpret_cast<hipDeviceptr_t>(device_ptr), pattern, dst.size(),
                              stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipMemcpyDtoH(dst.data(), reinterpret_cast<hipDeviceptr_t>(device_ptr),
                          dst.size() * sizeof(int)));

  RequireAllEqual(dst, pattern);
}

// @asserts: hipMemsetD32Async - a null stream argument runs the fill on the default stream
HIP_TEST_CASE(Contract_DriverMemsetAsync_NullStream_UsesDefaultStream) {
  hip::contract::ContractCleanup cleanup;
  constexpr int pattern = 0x76543210;
  std::array<int, kDwordCount> dst{};
  void* device_ptr = nullptr;

  HIP_CHECK(hipMalloc(&device_ptr, dst.size() * sizeof(int)));
  cleanup.Add([device_ptr] { (void)hipFree(device_ptr); });

  HIP_CHECK(
      hipMemsetD32Async(reinterpret_cast<hipDeviceptr_t>(device_ptr), pattern, dst.size(), nullptr));
  HIP_CHECK(hipStreamSynchronize(nullptr));
  HIP_CHECK(hipMemcpyDtoH(dst.data(), reinterpret_cast<hipDeviceptr_t>(device_ptr),
                          dst.size() * sizeof(int)));

  RequireAllEqual(dst, pattern);
}

// @asserts: hipMemsetD8Async - a zero-element count succeeds and leaves existing memory unchanged
HIP_TEST_CASE(Contract_DriverMemsetAsync_ZeroCount_Succeeds) {
  hip::contract::ContractCleanup cleanup;
  constexpr uint8_t original = 0x21;
  constexpr uint8_t replacement = 0x7f;
  std::array<uint8_t, kByteCount> dst{};
  void* device_ptr = nullptr;
  hipStream_t stream = nullptr;

  HIP_CHECK(hipMalloc(&device_ptr, dst.size()));
  cleanup.Add([device_ptr] { (void)hipFree(device_ptr); });
  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([stream] { (void)hipStreamDestroy(stream); });
  HIP_CHECK(hipMemsetD8(reinterpret_cast<hipDeviceptr_t>(device_ptr), original, dst.size()));

  HIP_CHECK(hipMemsetD8Async(reinterpret_cast<hipDeviceptr_t>(device_ptr), replacement, 0, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipMemcpyDtoH(dst.data(), reinterpret_cast<hipDeviceptr_t>(device_ptr), dst.size()));

  RequireAllEqual(dst, original);
}

// @asserts: hipMemsetD8Async - the D8/D16/D32 async variants reject a null destination pointer with a non-success error
HIP_TEST_CASE(Contract_DriverMemsetAsync_NullDestination_IsRejected) {
  hip::contract::ContractCleanup cleanup;
  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([stream] { (void)hipStreamDestroy(stream); });

  REQUIRE(hipMemsetD8Async(hipDeviceptr_t(nullptr), 0x5a, kByteCount, stream) != hipSuccess);
  REQUIRE(hipMemsetD16Async(hipDeviceptr_t(nullptr), 0x1357, kWordCount, stream) != hipSuccess);
  REQUIRE(hipMemsetD32Async(hipDeviceptr_t(nullptr), 0x12345678, kDwordCount, stream) != hipSuccess);
}
