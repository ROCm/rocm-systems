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

std::array<uint8_t, kWidth * kHeight> MakePattern(uint8_t seed) {
  std::array<uint8_t, kWidth * kHeight> pattern{};
  for (size_t i = 0; i < pattern.size(); ++i) {
    pattern[i] = static_cast<uint8_t>(seed + i);
  }
  return pattern;
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

hip_Memcpy2D HostToDeviceCopy(void* dst, size_t dst_pitch, const void* src, size_t src_pitch,
                              size_t width, size_t height) {
  hip_Memcpy2D copy{};
  copy.srcMemoryType = hipMemoryTypeHost;
  copy.srcHost = src;
  copy.srcPitch = src_pitch;
  copy.dstMemoryType = hipMemoryTypeDevice;
  copy.dstDevice = reinterpret_cast<hipDeviceptr_t>(dst);
  copy.dstPitch = dst_pitch;
  copy.WidthInBytes = width;
  copy.Height = height;
  return copy;
}

hip_Memcpy2D DeviceToHostCopy(void* dst, size_t dst_pitch, const void* src, size_t src_pitch,
                              size_t width, size_t height) {
  hip_Memcpy2D copy{};
  copy.srcMemoryType = hipMemoryTypeDevice;
  copy.srcDevice = reinterpret_cast<hipDeviceptr_t>(const_cast<void*>(src));
  copy.srcPitch = src_pitch;
  copy.dstMemoryType = hipMemoryTypeHost;
  copy.dstHost = dst;
  copy.dstPitch = dst_pitch;
  copy.WidthInBytes = width;
  copy.Height = height;
  return copy;
}

hip_Memcpy2D DeviceToDeviceCopy(void* dst, size_t dst_pitch, const void* src, size_t src_pitch,
                                size_t width, size_t height) {
  hip_Memcpy2D copy{};
  copy.srcMemoryType = hipMemoryTypeDevice;
  copy.srcDevice = reinterpret_cast<hipDeviceptr_t>(const_cast<void*>(src));
  copy.srcPitch = src_pitch;
  copy.dstMemoryType = hipMemoryTypeDevice;
  copy.dstDevice = reinterpret_cast<hipDeviceptr_t>(dst);
  copy.dstPitch = dst_pitch;
  copy.WidthInBytes = width;
  copy.Height = height;
  return copy;
}
}  // namespace

HIP_TEST_CASE(Contract_DriverMemcpy2D_HtoDtoH_RoundTripsRows) {
  const auto src = MakePattern(0x12);
  std::array<uint8_t, kWidth * kHeight> dst{};
  void* device_ptr = nullptr;
  size_t pitch = 0;

  if (!TryMallocPitch(&device_ptr, &pitch, kWidth, kHeight)) {
    SkipPitchedAllocationUnsupported();
  }

  auto h2d = HostToDeviceCopy(device_ptr, pitch, src.data(), kWidth, kWidth, kHeight);
  HIP_CHECK(hipMemcpyParam2D(&h2d));

  auto d2h = DeviceToHostCopy(dst.data(), kWidth, device_ptr, pitch, kWidth, kHeight);
  HIP_CHECK(hipMemcpyParam2D(&d2h));

  REQUIRE(dst == src);

  HIP_CHECK(hipFree(device_ptr));
}

HIP_TEST_CASE(Contract_DriverMemcpy2D_DtoD_SingleDevice_CopiesRows) {
  const auto src = MakePattern(0x34);
  std::array<uint8_t, kWidth * kHeight> dst{};
  void* src_device_ptr = nullptr;
  void* dst_device_ptr = nullptr;
  size_t src_pitch = 0;
  size_t dst_pitch = 0;

  if (!TryMallocPitch(&src_device_ptr, &src_pitch, kWidth, kHeight)) {
    SkipPitchedAllocationUnsupported();
  }
  if (!TryMallocPitch(&dst_device_ptr, &dst_pitch, kWidth, kHeight)) {
    HIP_CHECK(hipFree(src_device_ptr));
    SkipPitchedAllocationUnsupported();
  }

  auto h2d = HostToDeviceCopy(src_device_ptr, src_pitch, src.data(), kWidth, kWidth, kHeight);
  HIP_CHECK(hipMemcpyParam2D(&h2d));

  auto d2d = DeviceToDeviceCopy(dst_device_ptr, dst_pitch, src_device_ptr, src_pitch, kWidth, kHeight);
  HIP_CHECK(hipMemcpyParam2D(&d2d));

  auto d2h = DeviceToHostCopy(dst.data(), kWidth, dst_device_ptr, dst_pitch, kWidth, kHeight);
  HIP_CHECK(hipMemcpyParam2D(&d2h));

  REQUIRE(dst == src);

  HIP_CHECK(hipFree(dst_device_ptr));
  HIP_CHECK(hipFree(src_device_ptr));
}

HIP_TEST_CASE(Contract_DriverMemcpy2D_ZeroExtent_Succeeds) {
  const auto src = MakePattern(0x56);
  std::array<uint8_t, kWidth * kHeight> dst{};
  void* device_ptr = nullptr;
  size_t pitch = 0;

  if (!TryMallocPitch(&device_ptr, &pitch, kWidth, kHeight)) {
    SkipPitchedAllocationUnsupported();
  }

  auto initialize = HostToDeviceCopy(device_ptr, pitch, src.data(), kWidth, kWidth, kHeight);
  HIP_CHECK(hipMemcpyParam2D(&initialize));

  auto zero_copy = HostToDeviceCopy(device_ptr, pitch, dst.data(), kWidth, 0, 0);
  HIP_CHECK(hipMemcpyParam2D(&zero_copy));

  auto read_back = DeviceToHostCopy(dst.data(), kWidth, device_ptr, pitch, kWidth, kHeight);
  HIP_CHECK(hipMemcpyParam2D(&read_back));

  REQUIRE(dst == src);

  HIP_CHECK(hipFree(device_ptr));
}

HIP_TEST_CASE(Contract_DriverMemcpy2D_Async_OnStream_HostToDeviceVisibleAfterSync) {
  const auto src = MakePattern(0x78);
  std::array<uint8_t, kWidth * kHeight> dst{};
  void* device_ptr = nullptr;
  size_t pitch = 0;
  hipStream_t stream = nullptr;

  if (!TryMallocPitch(&device_ptr, &pitch, kWidth, kHeight)) {
    SkipPitchedAllocationUnsupported();
  }
  HIP_CHECK(hipStreamCreate(&stream));

  auto h2d = HostToDeviceCopy(device_ptr, pitch, src.data(), kWidth, kWidth, kHeight);
  HIP_CHECK(hipMemcpyParam2DAsync(&h2d, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  auto d2h = DeviceToHostCopy(dst.data(), kWidth, device_ptr, pitch, kWidth, kHeight);
  HIP_CHECK(hipMemcpyParam2D(&d2h));

  REQUIRE(dst == src);

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(device_ptr));
}

HIP_TEST_CASE(Contract_DriverMemcpy2D_NullPointersInStruct_AreRejected) {
  std::array<uint8_t, kWidth * kHeight> host{};
  void* device_ptr = nullptr;
  size_t pitch = 0;
  hipStream_t stream = nullptr;

  if (!TryMallocPitch(&device_ptr, &pitch, kWidth, kHeight)) {
    SkipPitchedAllocationUnsupported();
  }
  HIP_CHECK(hipStreamCreate(&stream));

  auto null_src = HostToDeviceCopy(device_ptr, pitch, nullptr, kWidth, kWidth, kHeight);
  const hipError_t sync_null_src_status = hipMemcpyParam2D(&null_src);
  const hipError_t async_null_src_status = hipMemcpyParam2DAsync(&null_src, stream);

  auto null_dst = DeviceToHostCopy(nullptr, kWidth, device_ptr, pitch, kWidth, kHeight);
  const hipError_t sync_null_dst_status = hipMemcpyParam2D(&null_dst);
  const hipError_t async_null_dst_status = hipMemcpyParam2DAsync(&null_dst, stream);

  auto valid_copy = HostToDeviceCopy(device_ptr, pitch, host.data(), kWidth, kWidth, kHeight);
  HIP_CHECK(hipMemcpyParam2D(&valid_copy));

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(device_ptr));

  REQUIRE(sync_null_src_status != hipSuccess);
  REQUIRE(async_null_src_status != hipSuccess);
  REQUIRE(sync_null_dst_status != hipSuccess);
  REQUIRE(async_null_dst_status != hipSuccess);
}
