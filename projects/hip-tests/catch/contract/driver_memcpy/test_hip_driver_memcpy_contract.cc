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
}  // namespace

HIP_TEST_CASE(Contract_DriverMemcpy_HtoDtoH_RoundTripsBytes) {
  const auto src = MakePattern(0x11);
  std::array<uint8_t, kElementCount> dst{};
  void* device_ptr = nullptr;

  HIP_CHECK(hipMalloc(&device_ptr, src.size()));

  HIP_CHECK(hipMemcpyHtoD(reinterpret_cast<hipDeviceptr_t>(device_ptr), src.data(), src.size()));
  HIP_CHECK(hipMemcpyDtoH(dst.data(), reinterpret_cast<hipDeviceptr_t>(device_ptr), dst.size()));

  REQUIRE(dst == src);

  HIP_CHECK(hipFree(device_ptr));
}

HIP_TEST_CASE(Contract_DriverMemcpy_DtoD_SingleDevice_CopiesBytes) {
  const auto src = MakePattern(0x29);
  std::array<uint8_t, kElementCount> dst{};
  void* src_device_ptr = nullptr;
  void* dst_device_ptr = nullptr;

  HIP_CHECK(hipMalloc(&src_device_ptr, src.size()));
  HIP_CHECK(hipMalloc(&dst_device_ptr, dst.size()));

  HIP_CHECK(hipMemcpyHtoD(reinterpret_cast<hipDeviceptr_t>(src_device_ptr), src.data(), src.size()));
  HIP_CHECK(hipMemcpyDtoD(reinterpret_cast<hipDeviceptr_t>(dst_device_ptr),
                          reinterpret_cast<hipDeviceptr_t>(src_device_ptr), src.size()));
  HIP_CHECK(hipMemcpyDtoH(dst.data(), reinterpret_cast<hipDeviceptr_t>(dst_device_ptr),
                          dst.size()));

  REQUIRE(dst == src);

  HIP_CHECK(hipFree(dst_device_ptr));
  HIP_CHECK(hipFree(src_device_ptr));
}

HIP_TEST_CASE(Contract_DriverMemcpy_ZeroBytes_Succeeds) {
  uint8_t host_src = 0x1;
  uint8_t host_dst = 0x2;
  void* src_device_ptr = nullptr;
  void* dst_device_ptr = nullptr;

  HIP_CHECK(hipMalloc(&src_device_ptr, sizeof(host_src)));
  HIP_CHECK(hipMalloc(&dst_device_ptr, sizeof(host_dst)));

  HIP_CHECK(hipMemcpyHtoD(reinterpret_cast<hipDeviceptr_t>(src_device_ptr), &host_src, 0));
  HIP_CHECK(hipMemcpyDtoH(&host_dst, reinterpret_cast<hipDeviceptr_t>(src_device_ptr), 0));
  HIP_CHECK(hipMemcpyDtoD(reinterpret_cast<hipDeviceptr_t>(dst_device_ptr),
                          reinterpret_cast<hipDeviceptr_t>(src_device_ptr), 0));

  REQUIRE(host_dst == 0x2);

  HIP_CHECK(hipFree(dst_device_ptr));
  HIP_CHECK(hipFree(src_device_ptr));
}

HIP_TEST_CASE(Contract_DriverMemcpy_Async_OnStream_RoundTripsBytes) {
  const auto src = MakePattern(0x43);
  std::array<uint8_t, kElementCount> dst{};
  void* src_device_ptr = nullptr;
  void* dst_device_ptr = nullptr;
  hipStream_t stream = nullptr;

  HIP_CHECK(hipMalloc(&src_device_ptr, src.size()));
  HIP_CHECK(hipMalloc(&dst_device_ptr, dst.size()));
  HIP_CHECK(hipStreamCreate(&stream));

  HIP_CHECK(hipMemcpyHtoDAsync(reinterpret_cast<hipDeviceptr_t>(src_device_ptr), src.data(),
                               src.size(), stream));
  HIP_CHECK(hipMemcpyDtoDAsync(reinterpret_cast<hipDeviceptr_t>(dst_device_ptr),
                               reinterpret_cast<hipDeviceptr_t>(src_device_ptr), src.size(), stream));
  HIP_CHECK(hipMemcpyDtoHAsync(dst.data(), reinterpret_cast<hipDeviceptr_t>(dst_device_ptr),
                               dst.size(), stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  REQUIRE(dst == src);

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(dst_device_ptr));
  HIP_CHECK(hipFree(src_device_ptr));
}

HIP_TEST_CASE(Contract_DriverMemcpy_NullPointers_AreRejected) {
  uint8_t host[kElementCount]{};
  void* src_device_ptr = nullptr;
  void* dst_device_ptr = nullptr;

  HIP_CHECK(hipMalloc(&src_device_ptr, kElementCount));
  HIP_CHECK(hipMalloc(&dst_device_ptr, kElementCount));

  REQUIRE(hipMemcpyHtoD(hipDeviceptr_t(nullptr), host, kElementCount) != hipSuccess);
  REQUIRE(hipMemcpyHtoD(reinterpret_cast<hipDeviceptr_t>(dst_device_ptr), nullptr,
                        kElementCount) != hipSuccess);
  REQUIRE(hipMemcpyDtoH(nullptr, reinterpret_cast<hipDeviceptr_t>(src_device_ptr),
                        kElementCount) != hipSuccess);
  REQUIRE(hipMemcpyDtoH(host, hipDeviceptr_t(nullptr), kElementCount) != hipSuccess);
  REQUIRE(hipMemcpyDtoD(hipDeviceptr_t(nullptr), reinterpret_cast<hipDeviceptr_t>(src_device_ptr),
                        kElementCount) != hipSuccess);
  REQUIRE(hipMemcpyDtoD(reinterpret_cast<hipDeviceptr_t>(dst_device_ptr), hipDeviceptr_t(nullptr),
                        kElementCount) != hipSuccess);

  HIP_CHECK(hipFree(dst_device_ptr));
  HIP_CHECK(hipFree(src_device_ptr));
}

HIP_TEST_CASE(Contract_DriverMemcpy_AsyncNullPointers_AreRejected) {
  uint8_t host[kElementCount]{};
  void* src_device_ptr = nullptr;
  void* dst_device_ptr = nullptr;
  hipStream_t stream = nullptr;

  HIP_CHECK(hipMalloc(&src_device_ptr, kElementCount));
  HIP_CHECK(hipMalloc(&dst_device_ptr, kElementCount));
  HIP_CHECK(hipStreamCreate(&stream));

  REQUIRE(hipMemcpyHtoDAsync(hipDeviceptr_t(nullptr), host, kElementCount, stream) != hipSuccess);
  REQUIRE(hipMemcpyHtoDAsync(reinterpret_cast<hipDeviceptr_t>(dst_device_ptr), nullptr,
                             kElementCount, stream) != hipSuccess);
  REQUIRE(hipMemcpyDtoHAsync(nullptr, reinterpret_cast<hipDeviceptr_t>(src_device_ptr),
                             kElementCount, stream) != hipSuccess);
  REQUIRE(hipMemcpyDtoHAsync(host, hipDeviceptr_t(nullptr), kElementCount, stream) != hipSuccess);
  REQUIRE(hipMemcpyDtoDAsync(hipDeviceptr_t(nullptr),
                             reinterpret_cast<hipDeviceptr_t>(src_device_ptr), kElementCount,
                             stream) != hipSuccess);
  REQUIRE(hipMemcpyDtoDAsync(reinterpret_cast<hipDeviceptr_t>(dst_device_ptr),
                             hipDeviceptr_t(nullptr), kElementCount, stream) != hipSuccess);

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(dst_device_ptr));
  HIP_CHECK(hipFree(src_device_ptr));
}
