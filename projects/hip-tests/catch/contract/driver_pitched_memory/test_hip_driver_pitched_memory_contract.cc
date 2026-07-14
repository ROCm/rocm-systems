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

constexpr size_t kUnalignedWidthBytes = 64;
constexpr size_t kUnalignedHeight = 4;

hip_Memcpy2D HostToDeviceUnaligned(hipDeviceptr_t dst, const void* src, size_t width_bytes,
                                   size_t height) {
  hip_Memcpy2D copy{};
  copy.srcMemoryType = hipMemoryTypeHost;
  copy.srcHost = src;
  copy.srcPitch = width_bytes;
  copy.dstMemoryType = hipMemoryTypeDevice;
  copy.dstDevice = dst;
  copy.dstPitch = width_bytes;
  copy.WidthInBytes = width_bytes;
  copy.Height = height;
  return copy;
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

HIP_TEST_CASE(Contract_DriverPitchedMemory_Memcpy2DUnaligned_HostToDevice_RoundTripsBytes) {
  std::array<uint8_t, kUnalignedWidthBytes * kUnalignedHeight> src{};
  std::array<uint8_t, kUnalignedWidthBytes * kUnalignedHeight> dst{};
  for (size_t i = 0; i < src.size(); ++i) {
    src[i] = static_cast<uint8_t>(0x30u + i);
  }

  void* device_ptr = nullptr;
  HIP_CHECK(hipMalloc(&device_ptr, src.size()));

  auto h2d = HostToDeviceUnaligned(reinterpret_cast<hipDeviceptr_t>(device_ptr), src.data(),
                                   kUnalignedWidthBytes, kUnalignedHeight);
  HIP_CHECK(hipDrvMemcpy2DUnaligned(&h2d));
  HIP_CHECK(hipMemcpy(dst.data(), device_ptr, dst.size(), hipMemcpyDeviceToHost));

  REQUIRE(dst == src);

  HIP_CHECK(hipFree(device_ptr));
}

HIP_TEST_CASE(Contract_DriverPitchedMemory_Memcpy2DUnaligned_NullInner_IsRejected) {
  std::array<uint8_t, kUnalignedWidthBytes * kUnalignedHeight> host{};
  void* device_ptr = nullptr;
  HIP_CHECK(hipMalloc(&device_ptr, host.size()));

  HIP_CHECK(hipGetLastError());
  auto null_src = HostToDeviceUnaligned(reinterpret_cast<hipDeviceptr_t>(device_ptr), nullptr,
                                        kUnalignedWidthBytes, kUnalignedHeight);
  const hipError_t null_src_status = hipDrvMemcpy2DUnaligned(&null_src);
  HIP_CHECK_ERROR(hipGetLastError(), null_src_status);
  HIP_CHECK(hipGetLastError());

  auto valid_copy = HostToDeviceUnaligned(reinterpret_cast<hipDeviceptr_t>(device_ptr),
                                          host.data(), kUnalignedWidthBytes, kUnalignedHeight);
  HIP_CHECK(hipDrvMemcpy2DUnaligned(&valid_copy));

  HIP_CHECK(hipFree(device_ptr));

  REQUIRE(null_src_status != hipSuccess);
}

HIP_TEST_CASE(Contract_DriverPitchedMemory_RejectsInvalidArgs) {
  hipDeviceptr_t device_ptr = 0;
  size_t pitch = 0;
  const size_t width_bytes = kWidth * sizeof(uint32_t);

  REQUIRE(hipMemAllocPitch(nullptr, &pitch, width_bytes, kHeight, kElementBytes) != hipSuccess);
  REQUIRE(hipMemAllocPitch(&device_ptr, nullptr, width_bytes, kHeight, kElementBytes) != hipSuccess);
}
