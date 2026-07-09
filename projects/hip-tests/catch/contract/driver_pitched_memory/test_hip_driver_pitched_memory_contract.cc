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
constexpr size_t kWidth = 16;
constexpr size_t kHeight = 8;
constexpr unsigned int kElementBytes = sizeof(uint32_t);

std::array<uint32_t, kWidth * kHeight> MakePattern(uint32_t seed) {
  std::array<uint32_t, kWidth * kHeight> pattern{};
  for (size_t i = 0; i < pattern.size(); ++i) {
    pattern[i] = seed + static_cast<uint32_t>(i);
  }
  return pattern;
}

bool TryMemAllocPitch(hipDeviceptr_t* ptr, size_t* pitch, size_t width, size_t height,
                      unsigned int element_size) {
  const hipError_t status = hipMemAllocPitch(ptr, pitch, width, height, element_size);
  if (status == hipErrorOutOfMemory) {
    return false;
  }
  HIP_CHECK(status);
  return true;
}

void SkipPitchedAllocationUnsupported() {
  HIP_SKIP_TEST("hipMemAllocPitch is not supported by this device/runtime path.");
}
}  // namespace

HIP_TEST_CASE(Contract_DriverPitchedMemory_AllocPitch_ReturnsPitchAtLeastWidth) {
  hipDeviceptr_t device_ptr = 0;
  size_t pitch = 0;
  const size_t width_bytes = kWidth * sizeof(uint32_t);

  if (!TryMemAllocPitch(&device_ptr, &pitch, width_bytes, kHeight, kElementBytes)) {
    SkipPitchedAllocationUnsupported();
  }

  REQUIRE(device_ptr != 0);
  REQUIRE(pitch >= width_bytes);

  HIP_CHECK(hipFree(reinterpret_cast<void*>(device_ptr)));
}

HIP_TEST_CASE(Contract_DriverPitchedMemory_Memcpy2D_HostDeviceRoundTripsRows) {
  const auto src = MakePattern(0x10u);
  std::array<uint32_t, kWidth * kHeight> dst{};
  hipDeviceptr_t device_ptr = 0;
  size_t pitch = 0;
  const size_t width_bytes = kWidth * sizeof(uint32_t);

  if (!TryMemAllocPitch(&device_ptr, &pitch, width_bytes, kHeight, kElementBytes)) {
    SkipPitchedAllocationUnsupported();
  }

  HIP_CHECK(hipMemcpy2D(reinterpret_cast<void*>(device_ptr), pitch, src.data(), width_bytes,
                        width_bytes, kHeight, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy2D(dst.data(), width_bytes, reinterpret_cast<void*>(device_ptr), pitch,
                        width_bytes, kHeight, hipMemcpyDeviceToHost));

  REQUIRE(dst == src);

  HIP_CHECK(hipFree(reinterpret_cast<void*>(device_ptr)));
}

HIP_TEST_CASE(Contract_DriverPitchedMemory_MemsetD2D32_RoundTripsWords) {
  constexpr uint32_t pattern = 0x12345678u;
  std::array<uint32_t, kWidth * kHeight> dst{};
  hipDeviceptr_t device_ptr = 0;
  size_t pitch = 0;
  const size_t width_bytes = kWidth * sizeof(uint32_t);

  if (!TryMemAllocPitch(&device_ptr, &pitch, width_bytes, kHeight, kElementBytes)) {
    SkipPitchedAllocationUnsupported();
  }

  HIP_CHECK(hipMemsetD2D32(device_ptr, pitch, static_cast<int>(pattern), kWidth, kHeight));
  HIP_CHECK(hipMemcpy2D(dst.data(), width_bytes, reinterpret_cast<void*>(device_ptr), pitch,
                        width_bytes, kHeight, hipMemcpyDeviceToHost));

  for (const auto value : dst) {
    REQUIRE(value == pattern);
  }

  HIP_CHECK(hipFree(reinterpret_cast<void*>(device_ptr)));
}

HIP_TEST_CASE(Contract_DriverPitchedMemory_FreePitchedAllocation_Succeeds) {
  hipDeviceptr_t device_ptr = 0;
  size_t pitch = 0;
  const size_t width_bytes = kWidth * sizeof(uint32_t);

  if (!TryMemAllocPitch(&device_ptr, &pitch, width_bytes, kHeight, kElementBytes)) {
    SkipPitchedAllocationUnsupported();
  }

  HIP_CHECK(hipFree(reinterpret_cast<void*>(device_ptr)));
}

HIP_TEST_CASE(Contract_DriverPitchedMemory_RejectsInvalidArgs) {
  hipDeviceptr_t device_ptr = 0;
  size_t pitch = 0;
  const size_t width_bytes = kWidth * sizeof(uint32_t);

  REQUIRE(hipMemAllocPitch(nullptr, &pitch, width_bytes, kHeight, kElementBytes) != hipSuccess);
  REQUIRE(hipMemAllocPitch(&device_ptr, nullptr, width_bytes, kHeight, kElementBytes) != hipSuccess);
}
