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

hipPitchedPtr HostPitchedPtr(void* ptr, size_t width, size_t height) {
  return make_hipPitchedPtr(ptr, width, width, height);
}

hipExtent ByteExtent(size_t width, size_t height, size_t depth) {
  return make_hipExtent(width, height, depth);
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
}  // namespace

HIP_TEST_CASE(Contract_Copy3D_Malloc3D_ReturnsPitchedPtr) {
  hip::contract::ContractCleanup cleanup;
  hipPitchedPtr device{};
  const auto extent = ByteExtent(kWidth, kHeight, kDepth);

  if (!TryMalloc3D(&device, extent)) {
    HIP_SKIP_TEST("hipMalloc3D is not supported by this device/runtime path.");
  }
  cleanup.Add([&] { (void)hipFree(device.ptr); });

  REQUIRE(device.ptr != nullptr);
  REQUIRE(device.pitch >= kWidth);
  REQUIRE(device.xsize == kWidth);
  REQUIRE(device.ysize == kHeight);
}

HIP_TEST_CASE(Contract_Copy3D_Memcpy3D_HostDeviceRoundTripsExtent) {
  hip::contract::ContractCleanup cleanup;
  const auto src = MakePattern(0x12);
  std::array<uint8_t, kWidth * kHeight * kDepth> dst{};
  hipPitchedPtr device{};
  const auto extent = ByteExtent(kWidth, kHeight, kDepth);

  if (!TryMalloc3D(&device, extent)) {
    HIP_SKIP_TEST("hipMalloc3D is not supported by this device/runtime path.");
  }
  cleanup.Add([&] { (void)hipFree(device.ptr); });

  hipMemcpy3DParms h2d{};
  h2d.srcPtr = HostPitchedPtr(const_cast<uint8_t*>(src.data()), kWidth, kHeight);
  h2d.dstPtr = device;
  h2d.extent = extent;
  h2d.kind = hipMemcpyHostToDevice;
  HIP_CHECK(hipMemcpy3D(&h2d));

  hipMemcpy3DParms d2h{};
  d2h.srcPtr = device;
  d2h.dstPtr = HostPitchedPtr(dst.data(), kWidth, kHeight);
  d2h.extent = extent;
  d2h.kind = hipMemcpyDeviceToHost;
  HIP_CHECK(hipMemcpy3D(&d2h));

  REQUIRE(dst == src);
}

HIP_TEST_CASE(Contract_Copy3D_Memcpy3D_SingleSliceRoundTripsBytes) {
  hip::contract::ContractCleanup cleanup;
  const auto src = MakePattern(0x56);
  std::array<uint8_t, kWidth * kHeight * kDepth> dst{};
  hipPitchedPtr device{};
  const auto extent = ByteExtent(kWidth, kHeight, 1);

  if (!TryMalloc3D(&device, extent)) {
    HIP_SKIP_TEST("hipMalloc3D is not supported by this device/runtime path.");
  }
  cleanup.Add([&] { (void)hipFree(device.ptr); });

  hipMemcpy3DParms h2d{};
  h2d.srcPtr = HostPitchedPtr(const_cast<uint8_t*>(src.data()), kWidth, kHeight);
  h2d.dstPtr = device;
  h2d.extent = extent;
  h2d.kind = hipMemcpyHostToDevice;
  HIP_CHECK(hipMemcpy3D(&h2d));

  hipMemcpy3DParms d2h{};
  d2h.srcPtr = device;
  d2h.dstPtr = HostPitchedPtr(dst.data(), kWidth, kHeight);
  d2h.extent = extent;
  d2h.kind = hipMemcpyDeviceToHost;
  HIP_CHECK(hipMemcpy3D(&d2h));

  for (size_t i = 0; i < kWidth * kHeight; ++i) {
    REQUIRE(dst[i] == src[i]);
  }
}

HIP_TEST_CASE(Contract_Copy3D_Free3DAllocation_Succeeds) {
  hipPitchedPtr device{};
  const auto extent = ByteExtent(kWidth, kHeight, kDepth);

  if (!TryMalloc3D(&device, extent)) {
    HIP_SKIP_TEST("hipMalloc3D is not supported by this device/runtime path.");
  }

  HIP_CHECK(hipFree(device.ptr));
}
