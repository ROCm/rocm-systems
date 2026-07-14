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
constexpr size_t kWidth = 8;
constexpr size_t kHeight = 4;
constexpr size_t kDepth = 2;
constexpr size_t kByteCount = kWidth * kHeight * kDepth;

std::array<uint8_t, kByteCount> MakePattern(uint8_t seed) {
  std::array<uint8_t, kByteCount> pattern{};
  for (size_t i = 0; i < pattern.size(); ++i) {
    pattern[i] = static_cast<uint8_t>(seed + i);
  }
  return pattern;
}

int CurrentDevice() {
  int device = 0;
  HIP_CHECK(hipGetDevice(&device));
  return device;
}

int DeviceCount() {
  int count = 0;
  HIP_CHECK(hipGetDeviceCount(&count));
  return count;
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

HIP_TEST_CASE(Contract_PeerCopy_SelfDevice1D_CopiesBytes) {
  // A peer copy where the source and destination device are the same device is
  // a valid degenerate case: it must behave like an ordinary device-to-device
  // copy. This keeps the positive contract runnable on a single-GPU host.
  hip::contract::ContractCleanup cleanup;
  const int device = CurrentDevice();
  const auto src = MakePattern(0x11);
  std::array<uint8_t, kByteCount> dst{};

  int* device_src = nullptr;
  int* device_dst = nullptr;
  HIP_CHECK(hipMalloc(&device_src, kByteCount));
  cleanup.Add([&] { (void)hipFree(device_src); });
  HIP_CHECK(hipMalloc(&device_dst, kByteCount));
  cleanup.Add([&] { (void)hipFree(device_dst); });
  HIP_CHECK(hipMemcpy(device_src, src.data(), kByteCount, hipMemcpyHostToDevice));

  HIP_CHECK(hipMemcpyPeer(device_dst, device, device_src, device, kByteCount));

  HIP_CHECK(hipMemcpy(dst.data(), device_dst, kByteCount, hipMemcpyDeviceToHost));
  REQUIRE(dst == src);
}

HIP_TEST_CASE(Contract_PeerCopy_SelfDevice1DAsync_CopiesBytesAfterSync) {
  hip::contract::ContractCleanup cleanup;
  const int device = CurrentDevice();
  const auto src = MakePattern(0x22);
  std::array<uint8_t, kByteCount> dst{};

  int* device_src = nullptr;
  int* device_dst = nullptr;
  hipStream_t stream = nullptr;
  HIP_CHECK(hipMalloc(&device_src, kByteCount));
  cleanup.Add([&] { (void)hipFree(device_src); });
  HIP_CHECK(hipMalloc(&device_dst, kByteCount));
  cleanup.Add([&] { (void)hipFree(device_dst); });
  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([&] { (void)hipStreamDestroy(stream); });
  HIP_CHECK(hipMemcpy(device_src, src.data(), kByteCount, hipMemcpyHostToDevice));

  // The async self-peer copy must complete and be visible after the stream is
  // synchronized.
  HIP_CHECK(hipMemcpyPeerAsync(device_dst, device, device_src, device, kByteCount, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  HIP_CHECK(hipMemcpy(dst.data(), device_dst, kByteCount, hipMemcpyDeviceToHost));
  REQUIRE(dst == src);
}

HIP_TEST_CASE(Contract_PeerCopy_SelfDevice3D_CopiesExtent) {
  // The 3D peer copy with matching source and destination device must round-trip
  // a pitched extent like an ordinary same-device 3D copy.
  hip::contract::ContractCleanup cleanup;
  const int device = CurrentDevice();
  const auto src = MakePattern(0x33);
  std::array<uint8_t, kByteCount> dst{};

  hipPitchedPtr device_src{};
  hipPitchedPtr device_dst{};
  const hipExtent extent = make_hipExtent(kWidth, kHeight, kDepth);
  if (!TryMalloc3D(&device_src, extent)) {
    HIP_SKIP_TEST("hipMalloc3D is not supported by this device/runtime path.");
  }
  cleanup.Add([&] { (void)hipFree(device_src.ptr); });
  if (!TryMalloc3D(&device_dst, extent)) {
    HIP_SKIP_TEST("hipMalloc3D is not supported by this device/runtime path.");
  }
  cleanup.Add([&] { (void)hipFree(device_dst.ptr); });

  // Seed the source pitched allocation via a host->device 3D copy.
  hipMemcpy3DParms up{};
  up.srcPtr = make_hipPitchedPtr(const_cast<uint8_t*>(src.data()), kWidth, kWidth, kHeight);
  up.dstPtr = device_src;
  up.extent = extent;
  up.kind = hipMemcpyHostToDevice;
  HIP_CHECK(hipMemcpy3D(&up));

  hipMemcpy3DPeerParms peer{};
  peer.srcPtr = device_src;
  peer.srcDevice = device;
  peer.dstPtr = device_dst;
  peer.dstDevice = device;
  peer.extent = extent;
  HIP_CHECK(hipMemcpy3DPeer(&peer));

  // Read the destination back and confirm the bytes survived the peer copy.
  hipMemcpy3DParms down{};
  down.srcPtr = device_dst;
  down.dstPtr = make_hipPitchedPtr(dst.data(), kWidth, kWidth, kHeight);
  down.extent = extent;
  down.kind = hipMemcpyDeviceToHost;
  HIP_CHECK(hipMemcpy3D(&down));

  REQUIRE(dst == src);
}

HIP_TEST_CASE(Contract_PeerCopy_SelfDevice3DAsync_CopiesExtentAfterSync) {
  hip::contract::ContractCleanup cleanup;
  const int device = CurrentDevice();
  const auto src = MakePattern(0x44);
  std::array<uint8_t, kByteCount> dst{};

  hipPitchedPtr device_src{};
  hipPitchedPtr device_dst{};
  hipStream_t stream = nullptr;
  const hipExtent extent = make_hipExtent(kWidth, kHeight, kDepth);
  if (!TryMalloc3D(&device_src, extent)) {
    HIP_SKIP_TEST("hipMalloc3D is not supported by this device/runtime path.");
  }
  cleanup.Add([&] { (void)hipFree(device_src.ptr); });
  if (!TryMalloc3D(&device_dst, extent)) {
    HIP_SKIP_TEST("hipMalloc3D is not supported by this device/runtime path.");
  }
  cleanup.Add([&] { (void)hipFree(device_dst.ptr); });
  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([&] { (void)hipStreamDestroy(stream); });

  hipMemcpy3DParms up{};
  up.srcPtr = make_hipPitchedPtr(const_cast<uint8_t*>(src.data()), kWidth, kWidth, kHeight);
  up.dstPtr = device_src;
  up.extent = extent;
  up.kind = hipMemcpyHostToDevice;
  HIP_CHECK(hipMemcpy3D(&up));

  hipMemcpy3DPeerParms peer{};
  peer.srcPtr = device_src;
  peer.srcDevice = device;
  peer.dstPtr = device_dst;
  peer.dstDevice = device;
  peer.extent = extent;
  HIP_CHECK(hipMemcpy3DPeerAsync(&peer, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  hipMemcpy3DParms down{};
  down.srcPtr = device_dst;
  down.dstPtr = make_hipPitchedPtr(dst.data(), kWidth, kWidth, kHeight);
  down.extent = extent;
  down.kind = hipMemcpyDeviceToHost;
  HIP_CHECK(hipMemcpy3D(&down));

  REQUIRE(dst == src);
}

HIP_TEST_CASE(Contract_PeerCopy_InvalidDevice_IsRejected) {
  // A peer copy naming an out-of-range device ordinal must not silently succeed.
  // The exact error code is backend-specific, so only a non-success status is
  // required. Buffers are valid so the rejection is attributable to the device
  // ordinal rather than a null pointer.
  hip::contract::ContractCleanup cleanup;
  const int device = CurrentDevice();
  const int invalid_device = DeviceCount();  // one past the last valid ordinal

  int* device_src = nullptr;
  int* device_dst = nullptr;
  HIP_CHECK(hipMalloc(&device_src, kByteCount));
  cleanup.Add([&] { (void)hipFree(device_src); });
  HIP_CHECK(hipMalloc(&device_dst, kByteCount));
  cleanup.Add([&] { (void)hipFree(device_dst); });

  HIP_CHECK(hipGetLastError());
  const hipError_t status =
      hipMemcpyPeer(device_dst, device, device_src, invalid_device, kByteCount);
  REQUIRE(status != hipSuccess);
  // The rejected copy sets a sticky last error matching the returned status;
  // consume it, then confirm the error state is clear.
  HIP_CHECK_ERROR(hipGetLastError(), status);
  HIP_CHECK(hipGetLastError());
}
