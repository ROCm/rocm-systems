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

HIP_TEST_CASE(Contract_DriverMemsetAsync_D8_FillsByte_VisibleAfterSync) {
  constexpr uint8_t pattern = 0x5a;
  std::array<uint8_t, kByteCount> dst{};
  void* device_ptr = nullptr;
  hipStream_t stream = nullptr;

  HIP_CHECK(hipMalloc(&device_ptr, dst.size()));
  HIP_CHECK(hipStreamCreate(&stream));

  HIP_CHECK(hipMemsetD8Async(reinterpret_cast<hipDeviceptr_t>(device_ptr), pattern, dst.size(),
                             stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipMemcpyDtoH(dst.data(), reinterpret_cast<hipDeviceptr_t>(device_ptr), dst.size()));

  RequireAllEqual(dst, pattern);

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(device_ptr));
}

HIP_TEST_CASE(Contract_DriverMemsetAsync_D16_FillsWord_VisibleAfterSync) {
  constexpr uint16_t pattern = 0x1357;
  std::array<uint16_t, kWordCount> dst{};
  void* device_ptr = nullptr;
  hipStream_t stream = nullptr;

  HIP_CHECK(hipMalloc(&device_ptr, dst.size() * sizeof(uint16_t)));
  HIP_CHECK(hipStreamCreate(&stream));

  HIP_CHECK(hipMemsetD16Async(reinterpret_cast<hipDeviceptr_t>(device_ptr), pattern, dst.size(),
                              stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipMemcpyDtoH(dst.data(), reinterpret_cast<hipDeviceptr_t>(device_ptr),
                          dst.size() * sizeof(uint16_t)));

  RequireAllEqual(dst, pattern);

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(device_ptr));
}

HIP_TEST_CASE(Contract_DriverMemsetAsync_D32_FillsDword_VisibleAfterSync) {
  constexpr int pattern = 0x12345678;
  std::array<int, kDwordCount> dst{};
  void* device_ptr = nullptr;
  hipStream_t stream = nullptr;

  HIP_CHECK(hipMalloc(&device_ptr, dst.size() * sizeof(int)));
  HIP_CHECK(hipStreamCreate(&stream));

  HIP_CHECK(hipMemsetD32Async(reinterpret_cast<hipDeviceptr_t>(device_ptr), pattern, dst.size(),
                              stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipMemcpyDtoH(dst.data(), reinterpret_cast<hipDeviceptr_t>(device_ptr),
                          dst.size() * sizeof(int)));

  RequireAllEqual(dst, pattern);

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(device_ptr));
}

HIP_TEST_CASE(Contract_DriverMemsetAsync_NullStream_UsesDefaultStream) {
  constexpr int pattern = 0x76543210;
  std::array<int, kDwordCount> dst{};
  void* device_ptr = nullptr;

  HIP_CHECK(hipMalloc(&device_ptr, dst.size() * sizeof(int)));

  HIP_CHECK(
      hipMemsetD32Async(reinterpret_cast<hipDeviceptr_t>(device_ptr), pattern, dst.size(), nullptr));
  HIP_CHECK(hipStreamSynchronize(nullptr));
  HIP_CHECK(hipMemcpyDtoH(dst.data(), reinterpret_cast<hipDeviceptr_t>(device_ptr),
                          dst.size() * sizeof(int)));

  RequireAllEqual(dst, pattern);

  HIP_CHECK(hipFree(device_ptr));
}

HIP_TEST_CASE(Contract_DriverMemsetAsync_ZeroCount_Succeeds) {
  constexpr uint8_t original = 0x21;
  constexpr uint8_t replacement = 0x7f;
  std::array<uint8_t, kByteCount> dst{};
  void* device_ptr = nullptr;
  hipStream_t stream = nullptr;

  HIP_CHECK(hipMalloc(&device_ptr, dst.size()));
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipMemsetD8(reinterpret_cast<hipDeviceptr_t>(device_ptr), original, dst.size()));

  HIP_CHECK(hipMemsetD8Async(reinterpret_cast<hipDeviceptr_t>(device_ptr), replacement, 0, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipMemcpyDtoH(dst.data(), reinterpret_cast<hipDeviceptr_t>(device_ptr), dst.size()));

  RequireAllEqual(dst, original);

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(device_ptr));
}

HIP_TEST_CASE(Contract_DriverMemsetAsync_NullDestination_IsRejected) {
  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));

  REQUIRE(hipMemsetD8Async(hipDeviceptr_t(nullptr), 0x5a, kByteCount, stream) != hipSuccess);
  REQUIRE(hipMemsetD16Async(hipDeviceptr_t(nullptr), 0x1357, kWordCount, stream) != hipSuccess);
  REQUIRE(hipMemsetD32Async(hipDeviceptr_t(nullptr), 0x12345678, kDwordCount, stream) != hipSuccess);

  HIP_CHECK(hipStreamDestroy(stream));
}
