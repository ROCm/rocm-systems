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
constexpr size_t kWidth = 13;
constexpr size_t kHeight = 7;

std::array<uint8_t, kWidth * kHeight> MakePattern(uint8_t seed) {
  std::array<uint8_t, kWidth * kHeight> pattern{};
  for (size_t i = 0; i < pattern.size(); ++i) {
    pattern[i] = static_cast<uint8_t>(seed + i);
  }
  return pattern;
}

// Some runtime/device paths (for example certain WSL2 dxg-backed configurations)
// do not provide a pitched allocator and report hipErrorOutOfMemory for every
// hipMallocPitch request. That is an allocator-support condition, not a contract
// violation, so the helper distinguishes it from genuine failures: it returns
// false when the allocation was refused with hipErrorOutOfMemory (signalling the
// caller to skip), asserts on any other error, and returns true with a populated
// pointer/pitch on success.
bool TryMallocPitch(void** device_ptr, size_t* pitch, size_t width, size_t height) {
  const hipError_t status = hipMallocPitch(device_ptr, pitch, width, height);
  if (status == hipErrorOutOfMemory) {
    return false;
  }
  HIP_CHECK(status);
  return true;
}
}  // namespace

// @asserts: hipMallocPitch - returns a non-null pointer and a pitch at least as large as the requested row width
HIP_TEST_CASE(Contract_PitchedMemory_MallocPitch_ReturnsPitchAtLeastWidth) {
  hip::contract::ContractCleanup cleanup;
  void* device_ptr = nullptr;
  size_t pitch = 0;

  if (!TryMallocPitch(&device_ptr, &pitch, kWidth, kHeight)) {
    HIP_SKIP_TEST("hipMallocPitch is not supported by this device/runtime path.");
  }
  cleanup.Add([device_ptr] { (void)hipFree(device_ptr); });

  REQUIRE(device_ptr != nullptr);
  REQUIRE(pitch >= kWidth);
}

// @asserts: hipMemcpy2D - a pitched host->device->host 2D round trip preserves every row's bytes
HIP_TEST_CASE(Contract_PitchedMemory_Memcpy2D_HostDeviceRoundTripsRows) {
  hip::contract::ContractCleanup cleanup;
  const auto src = MakePattern(0x24);
  std::array<uint8_t, kWidth * kHeight> dst{};
  void* device_ptr = nullptr;
  size_t pitch = 0;

  if (!TryMallocPitch(&device_ptr, &pitch, kWidth, kHeight)) {
    HIP_SKIP_TEST("hipMallocPitch is not supported by this device/runtime path.");
  }
  cleanup.Add([device_ptr] { (void)hipFree(device_ptr); });

  HIP_CHECK(hipMemcpy2D(device_ptr, pitch, src.data(), kWidth, kWidth, kHeight,
                        hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy2D(dst.data(), kWidth, device_ptr, pitch, kWidth, kHeight,
                        hipMemcpyDeviceToHost));

  REQUIRE(dst == src);
}

// @asserts: hipMemcpy2D - a single-row pitched 2D round trip preserves all bytes in that row
HIP_TEST_CASE(Contract_PitchedMemory_Memcpy2D_SingleRowRoundTripsBytes) {
  hip::contract::ContractCleanup cleanup;
  const auto src = MakePattern(0x63);
  std::array<uint8_t, kWidth * kHeight> dst{};
  void* device_ptr = nullptr;
  size_t pitch = 0;

  if (!TryMallocPitch(&device_ptr, &pitch, kWidth, 1)) {
    HIP_SKIP_TEST("hipMallocPitch is not supported by this device/runtime path.");
  }
  cleanup.Add([device_ptr] { (void)hipFree(device_ptr); });

  HIP_CHECK(hipMemcpy2D(device_ptr, pitch, src.data(), kWidth, kWidth, 1,
                        hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy2D(dst.data(), kWidth, device_ptr, pitch, kWidth, 1,
                        hipMemcpyDeviceToHost));

  for (size_t i = 0; i < kWidth; ++i) {
    REQUIRE(dst[i] == src[i]);
  }
}

// @asserts: hipFree - frees a hipMallocPitch allocation successfully
HIP_TEST_CASE(Contract_PitchedMemory_FreePitchedAllocation_Succeeds) {
  void* device_ptr = nullptr;
  size_t pitch = 0;

  if (!TryMallocPitch(&device_ptr, &pitch, kWidth, kHeight)) {
    HIP_SKIP_TEST("hipMallocPitch is not supported by this device/runtime path.");
  }

  HIP_CHECK(hipFree(device_ptr));
}
