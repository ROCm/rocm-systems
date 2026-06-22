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
constexpr size_t kWidth = 13;
constexpr size_t kHeight = 7;

std::array<uint8_t, kWidth * kHeight> MakePattern(uint8_t seed) {
  std::array<uint8_t, kWidth * kHeight> pattern{};
  for (size_t i = 0; i < pattern.size(); ++i) {
    pattern[i] = static_cast<uint8_t>(seed + i);
  }
  return pattern;
}
}

HIP_TEST_CASE(Contract_PitchedMemory_MallocPitch_ReturnsPitchAtLeastWidth) {
  void* device_ptr = nullptr;
  size_t pitch = 0;

  HIP_CHECK(hipMallocPitch(&device_ptr, &pitch, kWidth, kHeight));

  REQUIRE(device_ptr != nullptr);
  REQUIRE(pitch >= kWidth);

  HIP_CHECK(hipFree(device_ptr));
}

HIP_TEST_CASE(Contract_PitchedMemory_Memcpy2D_HostDeviceRoundTripsRows) {
  const auto src = MakePattern(0x24);
  std::array<uint8_t, kWidth * kHeight> dst{};
  void* device_ptr = nullptr;
  size_t pitch = 0;

  HIP_CHECK(hipMallocPitch(&device_ptr, &pitch, kWidth, kHeight));
  HIP_CHECK(hipMemcpy2D(device_ptr, pitch, src.data(), kWidth, kWidth, kHeight,
                        hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy2D(dst.data(), kWidth, device_ptr, pitch, kWidth, kHeight,
                        hipMemcpyDeviceToHost));

  REQUIRE(dst == src);

  HIP_CHECK(hipFree(device_ptr));
}

HIP_TEST_CASE(Contract_PitchedMemory_Memcpy2D_SingleRowRoundTripsBytes) {
  const auto src = MakePattern(0x63);
  std::array<uint8_t, kWidth * kHeight> dst{};
  void* device_ptr = nullptr;
  size_t pitch = 0;

  HIP_CHECK(hipMallocPitch(&device_ptr, &pitch, kWidth, 1));
  HIP_CHECK(hipMemcpy2D(device_ptr, pitch, src.data(), kWidth, kWidth, 1,
                        hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy2D(dst.data(), kWidth, device_ptr, pitch, kWidth, 1,
                        hipMemcpyDeviceToHost));

  for (size_t i = 0; i < kWidth; ++i) {
    REQUIRE(dst[i] == src[i]);
  }

  HIP_CHECK(hipFree(device_ptr));
}

HIP_TEST_CASE(Contract_PitchedMemory_FreePitchedAllocation_Succeeds) {
  void* device_ptr = nullptr;
  size_t pitch = 0;

  HIP_CHECK(hipMallocPitch(&device_ptr, &pitch, kWidth, kHeight));
  HIP_CHECK(hipFree(device_ptr));
}
