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
constexpr size_t kElementCount = 256;

std::array<uint8_t, kElementCount> MakePattern(uint8_t seed) {
  std::array<uint8_t, kElementCount> pattern{};
  for (size_t i = 0; i < pattern.size(); ++i) {
    pattern[i] = static_cast<uint8_t>(seed + i);
  }
  return pattern;
}
}

HIP_TEST_CASE(Contract_AsyncTransfer_HostToDeviceToHost_RoundTripsAfterStreamSynchronize) {
  const auto src = MakePattern(0x21);
  std::array<uint8_t, kElementCount> dst{};
  void* device_ptr = nullptr;
  hipStream_t stream = nullptr;

  HIP_CHECK(hipMalloc(&device_ptr, src.size()));
  HIP_CHECK(hipStreamCreate(&stream));

  HIP_CHECK(hipMemcpyAsync(device_ptr, src.data(), src.size(), hipMemcpyHostToDevice, stream));
  HIP_CHECK(hipMemcpyAsync(dst.data(), device_ptr, dst.size(), hipMemcpyDeviceToHost, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  REQUIRE(dst == src);

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(device_ptr));
}

HIP_TEST_CASE(Contract_AsyncTransfer_DeviceToDevice_CopiesAfterStreamSynchronize) {
  const auto src = MakePattern(0x58);
  std::array<uint8_t, kElementCount> dst{};
  void* src_device_ptr = nullptr;
  void* dst_device_ptr = nullptr;
  hipStream_t stream = nullptr;

  HIP_CHECK(hipMalloc(&src_device_ptr, src.size()));
  HIP_CHECK(hipMalloc(&dst_device_ptr, dst.size()));
  HIP_CHECK(hipStreamCreate(&stream));

  HIP_CHECK(hipMemcpyAsync(src_device_ptr, src.data(), src.size(), hipMemcpyHostToDevice, stream));
  HIP_CHECK(hipMemcpyAsync(dst_device_ptr, src_device_ptr, src.size(), hipMemcpyDeviceToDevice,
                           stream));
  HIP_CHECK(hipMemcpyAsync(dst.data(), dst_device_ptr, dst.size(), hipMemcpyDeviceToHost, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  REQUIRE(dst == src);

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(dst_device_ptr));
  HIP_CHECK(hipFree(src_device_ptr));
}

HIP_TEST_CASE(Contract_AsyncTransfer_ZeroBytes_Succeeds) {
  uint8_t src = 0x1;
  uint8_t dst = 0x2;
  hipStream_t stream = nullptr;

  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipMemcpyAsync(&dst, &src, 0, hipMemcpyHostToHost, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  REQUIRE(dst == 0x2);

  HIP_CHECK(hipStreamDestroy(stream));
}

HIP_TEST_CASE(Contract_AsyncTransfer_InvalidDirection_ReturnsConsistentError) {
  const auto src = MakePattern(0x83);
  void* device_ptr = nullptr;
  hipStream_t stream = nullptr;

  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMalloc(&device_ptr, src.size()));
  HIP_CHECK(hipStreamCreate(&stream));

  const hipError_t error =
      hipMemcpyAsync(device_ptr, src.data(), src.size(), static_cast<hipMemcpyKind>(-1), stream);

  REQUIRE(error != hipSuccess);
  HIP_CHECK_ERROR(hipGetLastError(), error);
  HIP_CHECK(hipGetLastError());

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(device_ptr));
}
