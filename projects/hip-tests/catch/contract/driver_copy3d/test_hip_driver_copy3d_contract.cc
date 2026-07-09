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
constexpr size_t kWidth = 7;
constexpr size_t kHeight = 5;
constexpr size_t kDepth = 3;

std::array<uint8_t, kWidth * kHeight * kDepth> MakePattern(uint8_t seed) {
  std::array<uint8_t, kWidth * kHeight * kDepth> pattern{};
  for (size_t i = 0; i < pattern.size(); ++i) {
    pattern[i] = static_cast<uint8_t>(seed + i);
  }
  return pattern;
}

hipExtent ByteExtent(size_t width, size_t height, size_t depth) {
  return make_hipExtent(width, height, depth);
}

hipPitchedPtr HostPitchedPtr(void* ptr, size_t width, size_t height) {
  return make_hipPitchedPtr(ptr, width, width, height);
}

bool TryMalloc3D(hipPitchedPtr* device_ptr, hipExtent extent) {
  const hipError_t status = hipMalloc3D(device_ptr, extent);
  if (status == hipSuccess) {
    return true;
  }
  if (status == hipErrorOutOfMemory || status == hipErrorNotSupported) {
    return false;
  }
  HIP_CHECK(status);
  return false;
}

HIP_MEMCPY3D HostToDeviceCopy(hipPitchedPtr dst, void* src, hipExtent extent) {
  HIP_MEMCPY3D copy{};
  copy.srcMemoryType = hipMemoryTypeHost;
  copy.srcHost = src;
  copy.srcPitch = extent.width;
  copy.srcHeight = extent.height;
  copy.dstMemoryType = hipMemoryTypeDevice;
  copy.dstDevice = reinterpret_cast<hipDeviceptr_t>(dst.ptr);
  copy.dstPitch = dst.pitch;
  copy.dstHeight = dst.ysize;
  copy.WidthInBytes = extent.width;
  copy.Height = extent.height;
  copy.Depth = extent.depth;
  return copy;
}

HIP_MEMCPY3D DeviceToDeviceCopy(hipPitchedPtr dst, hipPitchedPtr src, hipExtent extent) {
  HIP_MEMCPY3D copy{};
  copy.srcMemoryType = hipMemoryTypeDevice;
  copy.srcDevice = reinterpret_cast<hipDeviceptr_t>(src.ptr);
  copy.srcPitch = src.pitch;
  copy.srcHeight = src.ysize;
  copy.dstMemoryType = hipMemoryTypeDevice;
  copy.dstDevice = reinterpret_cast<hipDeviceptr_t>(dst.ptr);
  copy.dstPitch = dst.pitch;
  copy.dstHeight = dst.ysize;
  copy.WidthInBytes = extent.width;
  copy.Height = extent.height;
  copy.Depth = extent.depth;
  return copy;
}

void CopyDeviceToHost(std::array<uint8_t, kWidth * kHeight * kDepth>* dst, hipPitchedPtr src,
                      hipExtent extent) {
  hipMemcpy3DParms d2h{};
  d2h.srcPtr = src;
  d2h.dstPtr = HostPitchedPtr(dst->data(), kWidth, kHeight);
  d2h.extent = extent;
  d2h.kind = hipMemcpyDeviceToHost;
  HIP_CHECK(hipMemcpy3D(&d2h));
}
}  // namespace

HIP_TEST_CASE(Contract_DriverCopy3D_HostToDevice_RoundTripsExtent) {
  const auto src = MakePattern(0x23);
  std::array<uint8_t, kWidth * kHeight * kDepth> dst{};
  hipPitchedPtr device{};
  const auto extent = ByteExtent(kWidth, kHeight, kDepth);

  if (!TryMalloc3D(&device, extent)) {
    HIP_SKIP_TEST("hipMalloc3D is not supported by this device/runtime path.");
  }

  auto h2d = HostToDeviceCopy(device, const_cast<uint8_t*>(src.data()), extent);
  HIP_CHECK(hipDrvMemcpy3D(&h2d));
  CopyDeviceToHost(&dst, device, extent);

  REQUIRE(dst == src);

  HIP_CHECK(hipFree(device.ptr));
}

HIP_TEST_CASE(Contract_DriverCopy3D_DeviceToDevice_PreservesBytes) {
  const auto src = MakePattern(0x45);
  std::array<uint8_t, kWidth * kHeight * kDepth> dst{};
  hipPitchedPtr src_device{};
  hipPitchedPtr dst_device{};
  const auto extent = ByteExtent(kWidth, kHeight, kDepth);

  if (!TryMalloc3D(&src_device, extent)) {
    HIP_SKIP_TEST("hipMalloc3D is not supported by this device/runtime path.");
  }
  if (!TryMalloc3D(&dst_device, extent)) {
    HIP_CHECK(hipFree(src_device.ptr));
    HIP_SKIP_TEST("hipMalloc3D is not supported by this device/runtime path.");
  }

  auto h2d = HostToDeviceCopy(src_device, const_cast<uint8_t*>(src.data()), extent);
  HIP_CHECK(hipDrvMemcpy3D(&h2d));
  auto d2d = DeviceToDeviceCopy(dst_device, src_device, extent);
  HIP_CHECK(hipDrvMemcpy3D(&d2d));
  CopyDeviceToHost(&dst, dst_device, extent);

  REQUIRE(dst == src);

  HIP_CHECK(hipFree(dst_device.ptr));
  HIP_CHECK(hipFree(src_device.ptr));
}

HIP_TEST_CASE(Contract_DriverCopy3D_ZeroExtent_IsNoOp) {
  const auto src = MakePattern(0x67);
  std::array<uint8_t, kWidth * kHeight * kDepth> dst{};
  hipPitchedPtr device{};
  const auto extent = ByteExtent(kWidth, kHeight, kDepth);

  if (!TryMalloc3D(&device, extent)) {
    HIP_SKIP_TEST("hipMalloc3D is not supported by this device/runtime path.");
  }

  auto initialize = HostToDeviceCopy(device, const_cast<uint8_t*>(src.data()), extent);
  HIP_CHECK(hipDrvMemcpy3D(&initialize));
  auto zero_copy = HostToDeviceCopy(device, dst.data(), ByteExtent(0, 0, 0));
  HIP_CHECK(hipDrvMemcpy3D(&zero_copy));
  CopyDeviceToHost(&dst, device, extent);

  REQUIRE(dst == src);

  HIP_CHECK(hipFree(device.ptr));
}

HIP_TEST_CASE(Contract_DriverCopy3DAsync_HostToDevice_VisibleAfterSync) {
  const auto src = MakePattern(0x89);
  std::array<uint8_t, kWidth * kHeight * kDepth> dst{};
  hipPitchedPtr device{};
  hipStream_t stream = nullptr;
  const auto extent = ByteExtent(kWidth, kHeight, kDepth);

  if (!TryMalloc3D(&device, extent)) {
    HIP_SKIP_TEST("hipMalloc3D is not supported by this device/runtime path.");
  }
  HIP_CHECK(hipStreamCreate(&stream));

  auto h2d = HostToDeviceCopy(device, const_cast<uint8_t*>(src.data()), extent);
  HIP_CHECK(hipDrvMemcpy3DAsync(&h2d, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  CopyDeviceToHost(&dst, device, extent);

  REQUIRE(dst == src);

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(device.ptr));
}

HIP_TEST_CASE(Contract_DriverCopy3D_NullInnerPointer_IsRejected) {
  std::array<uint8_t, kWidth * kHeight * kDepth> host{};
  hipPitchedPtr device{};
  hipStream_t stream = nullptr;
  const auto extent = ByteExtent(kWidth, kHeight, kDepth);

  if (!TryMalloc3D(&device, extent)) {
    HIP_SKIP_TEST("hipMalloc3D is not supported by this device/runtime path.");
  }
  HIP_CHECK(hipStreamCreate(&stream));

  HIP_CHECK(hipGetLastError());
  auto null_src = HostToDeviceCopy(device, nullptr, extent);
  const hipError_t sync_null_src_status = hipDrvMemcpy3D(&null_src);
  HIP_CHECK_ERROR(hipGetLastError(), sync_null_src_status);
  HIP_CHECK(hipGetLastError());
  const hipError_t async_null_src_status = hipDrvMemcpy3DAsync(&null_src, stream);
  HIP_CHECK_ERROR(hipGetLastError(), async_null_src_status);
  HIP_CHECK(hipGetLastError());

  auto null_dst = DeviceToDeviceCopy(device, device, extent);
  null_dst.dstDevice = 0;
  const hipError_t sync_null_dst_status = hipDrvMemcpy3D(&null_dst);
  HIP_CHECK_ERROR(hipGetLastError(), sync_null_dst_status);
  HIP_CHECK(hipGetLastError());
  const hipError_t async_null_dst_status = hipDrvMemcpy3DAsync(&null_dst, stream);
  HIP_CHECK_ERROR(hipGetLastError(), async_null_dst_status);
  HIP_CHECK(hipGetLastError());

  auto valid_copy = HostToDeviceCopy(device, host.data(), extent);
  HIP_CHECK(hipDrvMemcpy3D(&valid_copy));

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(device.ptr));

  REQUIRE(sync_null_src_status != hipSuccess);
  REQUIRE(async_null_src_status != hipSuccess);
  REQUIRE(sync_null_dst_status != hipSuccess);
  REQUIRE(async_null_dst_status != hipSuccess);
}
