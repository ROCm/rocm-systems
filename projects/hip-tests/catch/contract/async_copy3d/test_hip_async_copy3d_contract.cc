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
constexpr size_t kLinearBytes = 256;

std::array<uint8_t, kWidth * kHeight * kDepth> Make3DPattern(uint8_t seed) {
  std::array<uint8_t, kWidth * kHeight * kDepth> pattern{};
  for (size_t i = 0; i < pattern.size(); ++i) {
    pattern[i] = static_cast<uint8_t>(seed + i);
  }
  return pattern;
}

std::array<uint8_t, kLinearBytes> MakeLinearPattern(uint8_t seed) {
  std::array<uint8_t, kLinearBytes> pattern{};
  for (size_t i = 0; i < pattern.size(); ++i) {
    pattern[i] = static_cast<uint8_t>(seed + i);
  }
  return pattern;
}

template <typename T, size_t N>
void RequireAllEqual(const std::array<T, N>& values, T expected) {
  for (const auto value : values) {
    REQUIRE(value == expected);
  }
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

// @asserts: hipMemcpy3DAsync - a 3D extent round-trips host->device->host intact once the stream is synchronized
HIP_TEST_CASE(Contract_AsyncCopy3D_Memcpy3DAsync_HostDeviceRoundTripsExtent_VisibleAfterSync) {
  hip::contract::ContractCleanup cleanup;
  const auto src = Make3DPattern(0x23);
  std::array<uint8_t, kWidth * kHeight * kDepth> dst{};
  hipPitchedPtr device{};
  hipStream_t stream = nullptr;
  const auto extent = ByteExtent(kWidth, kHeight, kDepth);

  if (!TryMalloc3D(&device, extent)) {
    HIP_SKIP_TEST("hipMalloc3D is not supported by this device/runtime path.");
  }
  cleanup.Add([p0 = device.ptr] { (void)hipFree(p0); });
  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([stream] { (void)hipStreamDestroy(stream); });

  hipMemcpy3DParms h2d{};
  h2d.srcPtr = HostPitchedPtr(const_cast<uint8_t*>(src.data()), kWidth, kHeight);
  h2d.dstPtr = device;
  h2d.extent = extent;
  h2d.kind = hipMemcpyHostToDevice;
  HIP_CHECK(hipMemcpy3DAsync(&h2d, stream));

  hipMemcpy3DParms d2h{};
  d2h.srcPtr = device;
  d2h.dstPtr = HostPitchedPtr(dst.data(), kWidth, kHeight);
  d2h.extent = extent;
  d2h.kind = hipMemcpyDeviceToHost;
  HIP_CHECK(hipMemcpy3DAsync(&d2h, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  REQUIRE(dst == src);
}

// @asserts: hipMemset3D - fills every byte of a 3D extent with the given value, visible after copy-back
HIP_TEST_CASE(Contract_AsyncCopy3D_Memset3D_FillsExtent_VisibleAfterCopyBack) {
  hip::contract::ContractCleanup cleanup;
  constexpr uint8_t pattern = 0x6d;
  std::array<uint8_t, kWidth * kHeight * kDepth> dst{};
  hipPitchedPtr device{};
  const auto extent = ByteExtent(kWidth, kHeight, kDepth);

  if (!TryMalloc3D(&device, extent)) {
    HIP_SKIP_TEST("hipMalloc3D is not supported by this device/runtime path.");
  }
  cleanup.Add([p0 = device.ptr] { (void)hipFree(p0); });

  HIP_CHECK(hipMemset3D(device, pattern, extent));

  hipMemcpy3DParms d2h{};
  d2h.srcPtr = device;
  d2h.dstPtr = HostPitchedPtr(dst.data(), kWidth, kHeight);
  d2h.extent = extent;
  d2h.kind = hipMemcpyDeviceToHost;
  HIP_CHECK(hipMemcpy3D(&d2h));

  RequireAllEqual(dst, pattern);
}

// @asserts: hipMemcpy3DAsync - rejects null copy params and surfaces the failure via hipGetLastError, then clears it
HIP_TEST_CASE(Contract_AsyncCopy3D_Memcpy3DAsync_NullParams_IsRejected) {
  hip::contract::ContractCleanup cleanup;
  hipStream_t stream = nullptr;

  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([stream] { (void)hipStreamDestroy(stream); });

  const hipError_t status = hipMemcpy3DAsync(nullptr, stream);

  REQUIRE(status != hipSuccess);
  HIP_CHECK_ERROR(hipGetLastError(), status);
  HIP_CHECK(hipGetLastError());
}

// @asserts: hipMemcpyWithStream - a linear buffer round-trips host->device->host intact once the stream is synchronized
HIP_TEST_CASE(Contract_AsyncCopy3D_MemcpyWithStream_RoundTripsBytes_VisibleAfterSync) {
  hip::contract::ContractCleanup cleanup;
  const auto src = MakeLinearPattern(0x45);
  std::array<uint8_t, kLinearBytes> dst{};
  void* device_ptr = nullptr;
  hipStream_t stream = nullptr;

  HIP_CHECK(hipMalloc(&device_ptr, src.size()));
  cleanup.Add([device_ptr] { (void)hipFree(device_ptr); });
  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([stream] { (void)hipStreamDestroy(stream); });

  HIP_CHECK(hipMemcpyWithStream(device_ptr, src.data(), src.size(), hipMemcpyHostToDevice, stream));
  HIP_CHECK(hipMemcpyWithStream(dst.data(), device_ptr, dst.size(), hipMemcpyDeviceToHost, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  REQUIRE(dst == src);
}

// @asserts: hipMemcpyWithStream - rejects an invalid hipMemcpyKind and surfaces the failure via hipGetLastError, then clears it
HIP_TEST_CASE(Contract_AsyncCopy3D_MemcpyWithStream_InvalidKind_IsRejected) {
  hip::contract::ContractCleanup cleanup;
  const auto src = MakeLinearPattern(0x67);
  void* device_ptr = nullptr;
  hipStream_t stream = nullptr;

  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMalloc(&device_ptr, src.size()));
  cleanup.Add([device_ptr] { (void)hipFree(device_ptr); });
  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([stream] { (void)hipStreamDestroy(stream); });

  const hipError_t status = hipMemcpyWithStream(device_ptr, src.data(), src.size(),
                                                static_cast<hipMemcpyKind>(-1), stream);

  REQUIRE(status != hipSuccess);
  HIP_CHECK_ERROR(hipGetLastError(), status);
  HIP_CHECK(hipGetLastError());
}

// @asserts: hipMemset3D - rejects a null (zero-initialized) pitched pointer and surfaces the failure via hipGetLastError
HIP_TEST_CASE(Contract_AsyncCopy3D_Memset3D_NullPitchedPtr_IsRejected) {
  hipPitchedPtr null_ptr{};
  const auto extent = ByteExtent(kWidth, kHeight, kDepth);

  HIP_CHECK(hipGetLastError());
  const hipError_t status = hipMemset3D(null_ptr, 0x5a, extent);

  REQUIRE(status != hipSuccess);
  HIP_CHECK_ERROR(hipGetLastError(), status);
  HIP_CHECK(hipGetLastError());
}
