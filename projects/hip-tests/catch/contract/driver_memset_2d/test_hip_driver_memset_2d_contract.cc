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
constexpr size_t kWidth = 17;
constexpr size_t kHeight = 9;

template <typename T, size_t N>
void RequireAllEqual(const std::array<T, N>& values, T expected) {
  for (const auto value : values) {
    REQUIRE(value == expected);
  }
}

bool TryMallocPitch(void** device_ptr, size_t* pitch, size_t width, size_t height) {
  const hipError_t status = hipMallocPitch(device_ptr, pitch, width, height);
  if (status == hipErrorOutOfMemory) {
    return false;
  }
  HIP_CHECK(status);
  return true;
}

void SkipPitchedAllocationUnsupported() {
  HIP_SKIP_TEST("hipMallocPitch is not supported by this device/runtime path.");
}
}  // namespace

HIP_TEST_CASE(Contract_DriverMemset2D_D2D8_FillsRows_VisibleAfterCopy) {
  constexpr uint8_t pattern = 0x5a;
  std::array<uint8_t, kWidth * kHeight> dst{};
  void* device_ptr = nullptr;
  size_t pitch = 0;

  if (!TryMallocPitch(&device_ptr, &pitch, kWidth * sizeof(uint8_t), kHeight)) {
    SkipPitchedAllocationUnsupported();
  }

  HIP_CHECK(hipMemsetD2D8(reinterpret_cast<hipDeviceptr_t>(device_ptr), pitch, pattern,
                          kWidth * sizeof(uint8_t), kHeight));
  HIP_CHECK(hipMemcpy2D(dst.data(), kWidth * sizeof(uint8_t), device_ptr, pitch,
                        kWidth * sizeof(uint8_t), kHeight, hipMemcpyDeviceToHost));

  RequireAllEqual(dst, pattern);

  HIP_CHECK(hipFree(device_ptr));
}

HIP_TEST_CASE(Contract_DriverMemset2D_D2D16_FillsWordRows_VisibleAfterCopy) {
  constexpr uint16_t pattern = 0x1357;
  std::array<uint16_t, kWidth * kHeight> dst{};
  void* device_ptr = nullptr;
  size_t pitch = 0;

  if (!TryMallocPitch(&device_ptr, &pitch, kWidth * sizeof(uint16_t), kHeight)) {
    SkipPitchedAllocationUnsupported();
  }

  HIP_CHECK(hipMemsetD2D16(reinterpret_cast<hipDeviceptr_t>(device_ptr), pitch, pattern,
                           kWidth * sizeof(uint16_t), kHeight));
  HIP_CHECK(hipMemcpy2D(dst.data(), kWidth * sizeof(uint16_t), device_ptr, pitch,
                        kWidth * sizeof(uint16_t), kHeight, hipMemcpyDeviceToHost));

  RequireAllEqual(dst, pattern);

  HIP_CHECK(hipFree(device_ptr));
}

HIP_TEST_CASE(Contract_DriverMemset2D_D2D32_FillsDwordRows_VisibleAfterCopy) {
  constexpr int pattern = 0x12345678;
  std::array<int, kWidth * kHeight> dst{};
  void* device_ptr = nullptr;
  size_t pitch = 0;

  if (!TryMallocPitch(&device_ptr, &pitch, kWidth * sizeof(int), kHeight)) {
    SkipPitchedAllocationUnsupported();
  }

  HIP_CHECK(hipMemsetD2D32(reinterpret_cast<hipDeviceptr_t>(device_ptr), pitch, pattern,
                           kWidth * sizeof(int), kHeight));
  HIP_CHECK(hipMemcpy2D(dst.data(), kWidth * sizeof(int), device_ptr, pitch, kWidth * sizeof(int),
                        kHeight, hipMemcpyDeviceToHost));

  RequireAllEqual(dst, pattern);

  HIP_CHECK(hipFree(device_ptr));
}

HIP_TEST_CASE(Contract_DriverMemset2D_AsyncD2D8_VisibleAfterSync) {
  constexpr uint8_t pattern = 0x3c;
  std::array<uint8_t, kWidth * kHeight> dst{};
  void* device_ptr = nullptr;
  size_t pitch = 0;
  hipStream_t stream = nullptr;

  if (!TryMallocPitch(&device_ptr, &pitch, kWidth * sizeof(uint8_t), kHeight)) {
    SkipPitchedAllocationUnsupported();
  }
  HIP_CHECK(hipStreamCreate(&stream));

  HIP_CHECK(hipMemsetD2D8Async(reinterpret_cast<hipDeviceptr_t>(device_ptr), pitch, pattern,
                               kWidth * sizeof(uint8_t), kHeight, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipMemcpy2D(dst.data(), kWidth * sizeof(uint8_t), device_ptr, pitch,
                        kWidth * sizeof(uint8_t), kHeight, hipMemcpyDeviceToHost));

  RequireAllEqual(dst, pattern);

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(device_ptr));
}

HIP_TEST_CASE(Contract_DriverMemset2D_RuntimeMemset2D_FillsRegion) {
  constexpr uint8_t pattern = 0x7b;
  std::array<uint8_t, kWidth * kHeight> dst{};
  void* device_ptr = nullptr;
  size_t pitch = 0;

  if (!TryMallocPitch(&device_ptr, &pitch, kWidth, kHeight)) {
    SkipPitchedAllocationUnsupported();
  }

  HIP_CHECK(hipMemset2D(device_ptr, pitch, pattern, kWidth, kHeight));
  HIP_CHECK(hipMemcpy2D(dst.data(), kWidth, device_ptr, pitch, kWidth, kHeight,
                        hipMemcpyDeviceToHost));

  RequireAllEqual(dst, pattern);

  HIP_CHECK(hipFree(device_ptr));
}

HIP_TEST_CASE(Contract_DriverMemset2D_NullDestination_IsRejected) {
  constexpr size_t pitch = kWidth * sizeof(uint32_t);

  REQUIRE(hipMemsetD2D8(hipDeviceptr_t(nullptr), pitch, 0x5a, kWidth, kHeight) != hipSuccess);
  REQUIRE(hipMemsetD2D16(hipDeviceptr_t(nullptr), pitch, 0x1357, kWidth * sizeof(uint16_t),
                         kHeight) != hipSuccess);
  REQUIRE(hipMemsetD2D32(hipDeviceptr_t(nullptr), pitch, 0x12345678, kWidth * sizeof(int),
                         kHeight) != hipSuccess);
  REQUIRE(hipMemset2D(nullptr, pitch, 0x7b, kWidth, kHeight) != hipSuccess);
}
