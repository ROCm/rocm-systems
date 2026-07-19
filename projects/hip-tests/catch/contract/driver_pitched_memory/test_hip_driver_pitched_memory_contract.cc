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
constexpr size_t kWidth = 16;
constexpr size_t kHeight = 8;
constexpr unsigned int kElementBytes = sizeof(uint32_t);

// Establishes a device context before the driver-style pitched-memory entry
// points below. On the NVIDIA backend these map to the driver API, which
// requires a bound primary context; a test that calls one before any allocation
// would otherwise fail with an initialization / "invalid device context" error.
// hipFree(0) is the canonical no-op that forces primary-context initialization,
// and is a harmless success on AMD where the runtime already auto-initializes.
void EnsureContext() { HIP_CHECK(hipFree(0)); }

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

// @asserts: hipMemAllocPitch - allocates a valid device pointer with pitch no smaller than the requested row width
HIP_TEST_CASE(Contract_DriverPitchedMemory_AllocPitch_ReturnsPitchAtLeastWidth) {
  hip::contract::ContractCleanup cleanup;
  EnsureContext();
  hipDeviceptr_t device_ptr = 0;
  size_t pitch = 0;
  const size_t width_bytes = kWidth * sizeof(uint32_t);

  if (!TryMemAllocPitch(&device_ptr, &pitch, width_bytes, kHeight, kElementBytes)) {
    SkipPitchedAllocationUnsupported();
  }
  cleanup.Add([device_ptr] { (void)hipFree(reinterpret_cast<void*>(device_ptr)); });

  REQUIRE(device_ptr != 0);
  REQUIRE(pitch >= width_bytes);
}

// @asserts: hipMemcpy2D - host->device->host copy over a pitched allocation round-trips all rows unchanged
HIP_TEST_CASE(Contract_DriverPitchedMemory_Memcpy2D_HostDeviceRoundTripsRows) {
  hip::contract::ContractCleanup cleanup;
  EnsureContext();
  const auto src = MakePattern(0x10u);
  std::array<uint32_t, kWidth * kHeight> dst{};
  hipDeviceptr_t device_ptr = 0;
  size_t pitch = 0;
  const size_t width_bytes = kWidth * sizeof(uint32_t);

  if (!TryMemAllocPitch(&device_ptr, &pitch, width_bytes, kHeight, kElementBytes)) {
    SkipPitchedAllocationUnsupported();
  }
  cleanup.Add([device_ptr] { (void)hipFree(reinterpret_cast<void*>(device_ptr)); });

  HIP_CHECK(hipMemcpy2D(reinterpret_cast<void*>(device_ptr), pitch, src.data(), width_bytes,
                        width_bytes, kHeight, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy2D(dst.data(), width_bytes, reinterpret_cast<void*>(device_ptr), pitch,
                        width_bytes, kHeight, hipMemcpyDeviceToHost));

  REQUIRE(dst == src);
}

// @asserts: hipMemsetD2D32 - fills a pitched allocation so every 32-bit word reads back the written pattern
HIP_TEST_CASE(Contract_DriverPitchedMemory_MemsetD2D32_RoundTripsWords) {
  hip::contract::ContractCleanup cleanup;
  EnsureContext();
  constexpr uint32_t pattern = 0x12345678u;
  std::array<uint32_t, kWidth * kHeight> dst{};
  hipDeviceptr_t device_ptr = 0;
  size_t pitch = 0;
  const size_t width_bytes = kWidth * sizeof(uint32_t);

  if (!TryMemAllocPitch(&device_ptr, &pitch, width_bytes, kHeight, kElementBytes)) {
    SkipPitchedAllocationUnsupported();
  }
  cleanup.Add([device_ptr] { (void)hipFree(reinterpret_cast<void*>(device_ptr)); });

  HIP_CHECK(hipMemsetD2D32(device_ptr, pitch, static_cast<int>(pattern), kWidth, kHeight));
  HIP_CHECK(hipMemcpy2D(dst.data(), width_bytes, reinterpret_cast<void*>(device_ptr), pitch,
                        width_bytes, kHeight, hipMemcpyDeviceToHost));

  for (const auto value : dst) {
    REQUIRE(value == pattern);
  }
}

// @asserts: hipFree - freeing a pitched allocation obtained from hipMemAllocPitch succeeds
HIP_TEST_CASE(Contract_DriverPitchedMemory_FreePitchedAllocation_Succeeds) {
  EnsureContext();
  hipDeviceptr_t device_ptr = 0;
  size_t pitch = 0;
  const size_t width_bytes = kWidth * sizeof(uint32_t);

  if (!TryMemAllocPitch(&device_ptr, &pitch, width_bytes, kHeight, kElementBytes)) {
    SkipPitchedAllocationUnsupported();
  }

  HIP_CHECK(hipFree(reinterpret_cast<void*>(device_ptr)));
}

// @asserts: hipDrvMemcpy2DUnaligned - an unaligned host->device 2D copy transfers every byte intact
HIP_TEST_CASE(Contract_DriverPitchedMemory_Memcpy2DUnaligned_HostToDevice_RoundTripsBytes) {
  hip::contract::ContractCleanup cleanup;
  EnsureContext();
  std::array<uint8_t, kUnalignedWidthBytes * kUnalignedHeight> src{};
  std::array<uint8_t, kUnalignedWidthBytes * kUnalignedHeight> dst{};
  for (size_t i = 0; i < src.size(); ++i) {
    src[i] = static_cast<uint8_t>(0x30u + i);
  }

  void* device_ptr = nullptr;
  HIP_CHECK(hipMalloc(&device_ptr, src.size()));
  cleanup.Add([device_ptr] { (void)hipFree(device_ptr); });

  auto h2d = HostToDeviceUnaligned(reinterpret_cast<hipDeviceptr_t>(device_ptr), src.data(),
                                   kUnalignedWidthBytes, kUnalignedHeight);
  HIP_CHECK(hipDrvMemcpy2DUnaligned(&h2d));
  HIP_CHECK(hipMemcpy(dst.data(), device_ptr, dst.size(), hipMemcpyDeviceToHost));

  REQUIRE(dst == src);
}

// @asserts: hipDrvMemcpy2DUnaligned - a null inner source pointer is rejected via a non-success return status
HIP_TEST_CASE(Contract_DriverPitchedMemory_Memcpy2DUnaligned_NullInner_IsRejected) {
  hip::contract::ContractCleanup cleanup;
  EnsureContext();
  std::array<uint8_t, kUnalignedWidthBytes * kUnalignedHeight> host{};
  void* device_ptr = nullptr;
  HIP_CHECK(hipMalloc(&device_ptr, host.size()));
  cleanup.Add([device_ptr] { (void)hipFree(device_ptr); });

  HIP_CHECK(hipGetLastError());
  auto null_src = HostToDeviceUnaligned(reinterpret_cast<hipDeviceptr_t>(device_ptr), nullptr,
                                        kUnalignedWidthBytes, kUnalignedHeight);
  // The contract is that the driver-style copy rejects a null inner source
  // pointer through its returned status. Whether the rejection also latches into
  // the thread-global last-error is backend-specific: the AMD runtime sets it,
  // but the NVIDIA driver-API path (cuMemcpy2DUnaligned) reports the error only
  // through the return value and leaves hipGetLastError() clear. Assert the
  // returned status, not the sticky error, so the contract holds on both backends.
  const hipError_t null_src_status = hipDrvMemcpy2DUnaligned(&null_src);
  (void)hipGetLastError();  // clear any latched error so it does not leak forward

  auto valid_copy = HostToDeviceUnaligned(reinterpret_cast<hipDeviceptr_t>(device_ptr),
                                          host.data(), kUnalignedWidthBytes, kUnalignedHeight);
  HIP_CHECK(hipDrvMemcpy2DUnaligned(&valid_copy));

  REQUIRE(null_src_status != hipSuccess);
}

// @asserts: hipMemAllocPitch - rejects a null pointer or null pitch out-parameter with a non-success error
HIP_TEST_CASE(Contract_DriverPitchedMemory_RejectsInvalidArgs) {
  EnsureContext();
  hipDeviceptr_t device_ptr = 0;
  size_t pitch = 0;
  const size_t width_bytes = kWidth * sizeof(uint32_t);

  REQUIRE(hipMemAllocPitch(nullptr, &pitch, width_bytes, kHeight, kElementBytes) != hipSuccess);
  REQUIRE(hipMemAllocPitch(&device_ptr, nullptr, width_bytes, kHeight, kElementBytes) != hipSuccess);
}
