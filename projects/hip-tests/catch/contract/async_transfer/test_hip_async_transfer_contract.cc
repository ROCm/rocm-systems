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
constexpr size_t kElementCount = 256;

std::array<uint8_t, kElementCount> MakePattern(uint8_t seed) {
  std::array<uint8_t, kElementCount> pattern{};
  for (size_t i = 0; i < pattern.size(); ++i) {
    pattern[i] = static_cast<uint8_t>(seed + i);
  }
  return pattern;
}
}

// @asserts: hipMemcpyAsync - host->device->host async copies round-trip byte-for-byte once the stream is synchronized
HIP_TEST_CASE(Contract_AsyncTransfer_HostToDeviceToHost_RoundTripsAfterStreamSynchronize) {
  hip::contract::ContractCleanup cleanup;
  const auto src = MakePattern(0x21);
  std::array<uint8_t, kElementCount> dst{};
  void* device_ptr = nullptr;
  hipStream_t stream = nullptr;

  HIP_CHECK(hipMalloc(&device_ptr, src.size()));
  cleanup.Add([device_ptr] { (void)hipFree(device_ptr); });
  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([stream] { (void)hipStreamDestroy(stream); });

  HIP_CHECK(hipMemcpyAsync(device_ptr, src.data(), src.size(), hipMemcpyHostToDevice, stream));
  HIP_CHECK(hipMemcpyAsync(dst.data(), device_ptr, dst.size(), hipMemcpyDeviceToHost, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  REQUIRE(dst == src);
}

// @asserts: hipMemcpyAsync - a device-to-device async copy reproduces the source data once the stream is synchronized
HIP_TEST_CASE(Contract_AsyncTransfer_DeviceToDevice_CopiesAfterStreamSynchronize) {
  hip::contract::ContractCleanup cleanup;
  const auto src = MakePattern(0x58);
  std::array<uint8_t, kElementCount> dst{};
  void* src_device_ptr = nullptr;
  void* dst_device_ptr = nullptr;
  hipStream_t stream = nullptr;

  HIP_CHECK(hipMalloc(&src_device_ptr, src.size()));
  cleanup.Add([src_device_ptr] { (void)hipFree(src_device_ptr); });
  HIP_CHECK(hipMalloc(&dst_device_ptr, dst.size()));
  cleanup.Add([dst_device_ptr] { (void)hipFree(dst_device_ptr); });
  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([stream] { (void)hipStreamDestroy(stream); });

  HIP_CHECK(hipMemcpyAsync(src_device_ptr, src.data(), src.size(), hipMemcpyHostToDevice, stream));
  HIP_CHECK(hipMemcpyAsync(dst_device_ptr, src_device_ptr, src.size(), hipMemcpyDeviceToDevice,
                           stream));
  HIP_CHECK(hipMemcpyAsync(dst.data(), dst_device_ptr, dst.size(), hipMemcpyDeviceToHost, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  REQUIRE(dst == src);
}

// @asserts: hipMemcpyAsync - a zero-byte async copy succeeds and leaves the destination unchanged
HIP_TEST_CASE(Contract_AsyncTransfer_ZeroBytes_Succeeds) {
  hip::contract::ContractCleanup cleanup;
  uint8_t src = 0x1;
  uint8_t dst = 0x2;
  hipStream_t stream = nullptr;

  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([stream] { (void)hipStreamDestroy(stream); });
  HIP_CHECK(hipMemcpyAsync(&dst, &src, 0, hipMemcpyHostToHost, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  REQUIRE(dst == 0x2);
}

// @asserts: hipMemcpyAsync - rejects an invalid hipMemcpyKind with a non-success status that matches the recorded last error
HIP_TEST_CASE(Contract_AsyncTransfer_InvalidDirection_ReturnsConsistentError) {
  hip::contract::ContractCleanup cleanup;
  const auto src = MakePattern(0x83);
  void* device_ptr = nullptr;
  hipStream_t stream = nullptr;

  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMalloc(&device_ptr, src.size()));
  cleanup.Add([device_ptr] { (void)hipFree(device_ptr); });
  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([stream] { (void)hipStreamDestroy(stream); });

  const hipError_t error =
      hipMemcpyAsync(device_ptr, src.data(), src.size(), static_cast<hipMemcpyKind>(-1), stream);

  REQUIRE(error != hipSuccess);
  HIP_CHECK_ERROR(hipGetLastError(), error);
  HIP_CHECK(hipGetLastError());
}
