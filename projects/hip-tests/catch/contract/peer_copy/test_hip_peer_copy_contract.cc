/*
 * Copyright Advanced Micro Devices, Inc.
 *
 * SPDX-License-Identifier: MIT
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <numeric>

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

// @asserts: hipMemcpyPeer - a same-device (degenerate) 1D peer copy transfers bytes like an ordinary device-to-device copy
HIP_TEST_CASE(Contract_PeerCopy_HipMemcpyPeer_SelfDevice1D_CopiesBytes) {
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
  cleanup.Add([device_src] { (void)hipFree(device_src); });
  HIP_CHECK(hipMalloc(&device_dst, kByteCount));
  cleanup.Add([device_dst] { (void)hipFree(device_dst); });
  HIP_CHECK(hipMemcpy(device_src, src.data(), kByteCount, hipMemcpyHostToDevice));

  HIP_CHECK(hipMemcpyPeer(device_dst, device, device_src, device, kByteCount));

  HIP_CHECK(hipMemcpy(dst.data(), device_dst, kByteCount, hipMemcpyDeviceToHost));
  REQUIRE(dst == src);
}

// @asserts: hipMemcpyPeerAsync - a same-device async 1D peer copy is complete and visible after the stream is synchronized
HIP_TEST_CASE(Contract_PeerCopy_HipMemcpyPeerAsync_SelfDevice1DAsync_CopiesBytesAfterSync) {
  hip::contract::ContractCleanup cleanup;
  const int device = CurrentDevice();
  const auto src = MakePattern(0x22);
  std::array<uint8_t, kByteCount> dst{};

  int* device_src = nullptr;
  int* device_dst = nullptr;
  hipStream_t stream = nullptr;
  HIP_CHECK(hipMalloc(&device_src, kByteCount));
  cleanup.Add([device_src] { (void)hipFree(device_src); });
  HIP_CHECK(hipMalloc(&device_dst, kByteCount));
  cleanup.Add([device_dst] { (void)hipFree(device_dst); });
  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([stream] { (void)hipStreamDestroy(stream); });
  HIP_CHECK(hipMemcpy(device_src, src.data(), kByteCount, hipMemcpyHostToDevice));

  // The async self-peer copy must complete and be visible after the stream is
  // synchronized.
  HIP_CHECK(hipMemcpyPeerAsync(device_dst, device, device_src, device, kByteCount, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  HIP_CHECK(hipMemcpy(dst.data(), device_dst, kByteCount, hipMemcpyDeviceToHost));
  REQUIRE(dst == src);
}

// @asserts: hipMemcpy3DPeer - a same-device 3D peer copy round-trips a pitched extent like an ordinary same-device 3D copy
HIP_TEST_CASE(Contract_PeerCopy_HipMemcpy3DPeer_SelfDevice3D_CopiesExtent) {
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
  cleanup.Add([p0 = device_src.ptr] { (void)hipFree(p0); });
  if (!TryMalloc3D(&device_dst, extent)) {
    HIP_SKIP_TEST("hipMalloc3D is not supported by this device/runtime path.");
  }
  cleanup.Add([p0 = device_dst.ptr] { (void)hipFree(p0); });

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

// @asserts: hipMemcpy3DPeerAsync - a same-device async 3D peer copy round-trips the extent once the stream is synchronized
HIP_TEST_CASE(Contract_PeerCopy_HipMemcpy3DPeerAsync_SelfDevice3DAsync_CopiesExtentAfterSync) {
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
  cleanup.Add([p0 = device_src.ptr] { (void)hipFree(p0); });
  if (!TryMalloc3D(&device_dst, extent)) {
    HIP_SKIP_TEST("hipMalloc3D is not supported by this device/runtime path.");
  }
  cleanup.Add([p0 = device_dst.ptr] { (void)hipFree(p0); });
  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([stream] { (void)hipStreamDestroy(stream); });

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

// @asserts: hipMemcpyPeer - a VMM peer copy without destination write access returns hipErrorInvalidValue without changing access or data
HIP_TEST_CASE(Contract_PeerCopy_HipMemcpyPeer_VmmDeniedDestination_IsRejectedWithoutGrant) {
  if (DeviceCount() < 2) HIP_SKIP_TEST("Two GPUs are required for a VMM peer copy.");
  for (int device = 0; device < 2; ++device) {
    int supported = 0;
    HIP_CHECK(hipDeviceGetAttribute(&supported, hipDeviceAttributeVirtualMemoryManagementSupported,
                                    device));
    if (!supported) HIP_SKIP_TEST("VMM is unsupported on a selected GPU.");
    HIP_CHECK(hipDeviceCanAccessPeer(&supported, device, 1 - device));
    if (!supported) HIP_SKIP_TEST("Bidirectional peer access is unavailable.");
  }

  hip::contract::ContractCleanup cleanup;
  const int original_device = CurrentDevice();
  cleanup.Add([original_device] { (void)hipSetDevice(original_device); });
  hipMemAllocationProp prop{};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  size_t granularity[2]{};
  for (int device = 0; device < 2; ++device) {
    prop.location.id = device;
    HIP_CHECK(hipMemGetAllocationGranularity(&granularity[device], &prop,
                                             hipMemAllocationGranularityMinimum));
  }
  const size_t size = std::lcm(granularity[0], granularity[1]);
  void* mapped[2]{};
  for (int device = 0; device < 2; ++device) {
    HIP_CHECK(hipSetDevice(device));
    prop.location.id = device;
    HIP_CHECK(hipMemAddressReserve(&mapped[device], size, 0, nullptr, 0));
    cleanup.Add([address = mapped[device], size, device] {
      (void)hipSetDevice(device);
      (void)hipMemAddressFree(address, size);
    });
    hipMemGenericAllocationHandle_t handle{};
    HIP_CHECK(hipMemCreate(&handle, size, &prop, 0));
    cleanup.Add([handle, device] {
      (void)hipSetDevice(device);
      (void)hipMemRelease(handle);
    });
    HIP_CHECK(hipMemMap(mapped[device], size, 0, handle, 0));
    cleanup.Add([address = mapped[device], size, device] {
      (void)hipSetDevice(device);
      (void)hipMemUnmap(address, size);
    });
    hipMemAccessDesc access{};
    access.location.type = hipMemLocationTypeDevice;
    access.location.id = device;
    access.flags = hipMemAccessFlagsProtReadWrite;
    HIP_CHECK(hipMemSetAccess(mapped[device], size, &access, 1));
  }

  const auto source = MakePattern(0x51);
  const auto poison = MakePattern(0xA3);
  std::array<uint8_t, kByteCount> actual{};
  HIP_CHECK(hipSetDevice(0));
  HIP_CHECK(hipMemcpy(mapped[0], source.data(), source.size(), hipMemcpyHostToDevice));
  HIP_CHECK(hipSetDevice(1));
  HIP_CHECK(hipMemcpy(mapped[1], poison.data(), poison.size(), hipMemcpyHostToDevice));
  HIP_CHECK(hipSetDevice(0));
  const hipError_t enabled = hipDeviceEnablePeerAccess(1, 0);
  REQUIRE((enabled == hipSuccess || enabled == hipErrorPeerAccessAlreadyEnabled));
  if (enabled == hipSuccess) {
    cleanup.Add([] {
      (void)hipSetDevice(0);
      (void)hipDeviceDisablePeerAccess(1);
    });
  }
  (void)hipGetLastError();

  hipMemAccessDesc deny{};
  deny.location.type = hipMemLocationTypeDevice;
  deny.location.id = 0;
  deny.flags = hipMemAccessFlagsProtNone;
  HIP_CHECK(hipMemSetAccess(mapped[1], size, &deny, 1));
  HIP_CHECK_ERROR(hipMemcpyPeer(mapped[1], 1, mapped[0], 0, kByteCount), hipErrorInvalidValue);
  (void)hipGetLastError();

  hipMemLocation copy_location{hipMemLocationTypeDevice, 0};
  unsigned long long access_after = ~0ULL;
  HIP_CHECK(hipMemGetAccess(&access_after, &copy_location, mapped[1]));
  REQUIRE(access_after == hipMemAccessFlagsProtNone);
  HIP_CHECK(hipSetDevice(1));
  HIP_CHECK(hipMemcpy(actual.data(), mapped[1], actual.size(), hipMemcpyDeviceToHost));
  REQUIRE(actual == poison);
}

// @asserts: hipMemcpyPeer - rejects an out-of-range device ordinal and leaves a matching sticky last error that clears once consumed
HIP_TEST_CASE(Contract_PeerCopy_HipMemcpyPeer_InvalidDevice_IsRejected) {
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
  cleanup.Add([device_src] { (void)hipFree(device_src); });
  HIP_CHECK(hipMalloc(&device_dst, kByteCount));
  cleanup.Add([device_dst] { (void)hipFree(device_dst); });

  HIP_CHECK(hipGetLastError());
  const hipError_t status =
      hipMemcpyPeer(device_dst, device, device_src, invalid_device, kByteCount);
  REQUIRE(status != hipSuccess);
  // The rejected copy sets a sticky last error matching the returned status;
  // consume it, then confirm the error state is clear.
  HIP_CHECK_ERROR(hipGetLastError(), status);
  HIP_CHECK(hipGetLastError());
}
