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
constexpr size_t kWidth = 17;
constexpr size_t kHeight = 9;
constexpr size_t kDepth = 3;

template <typename T, size_t N>
void RequireAllEqual(const std::array<T, N>& values, T expected) {
  for (const auto value : values) {
    REQUIRE(value == expected);
  }
}

bool TryMallocPitch(void** device_ptr, size_t* pitch, size_t width, size_t height) {
  const hipError_t status = hipMallocPitch(device_ptr, pitch, width, height);
  if (status == hipErrorOutOfMemory) {
    return false;
  }
  HIP_CHECK(status);
  return true;
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

hipPitchedPtr HostPitchedPtr(void* ptr, size_t width, size_t height) {
  return make_hipPitchedPtr(ptr, width, width, height);
}
}  // namespace

HIP_TEST_CASE(Contract_DriverMemsetAsync2D3D_D2D16Async_FillsWordRows_VisibleAfterSync) {
  hip::contract::ContractCleanup cleanup;
  constexpr uint16_t pattern = 0x1357;
  std::array<uint16_t, kWidth * kHeight> dst{};
  void* device_ptr = nullptr;
  size_t pitch = 0;
  hipStream_t stream = nullptr;

  if (!TryMallocPitch(&device_ptr, &pitch, kWidth * sizeof(uint16_t), kHeight)) {
    HIP_SKIP_TEST("hipMallocPitch is not supported by this device/runtime path.");
  }
  cleanup.Add([&] { (void)hipFree(device_ptr); });
  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([&] { (void)hipStreamDestroy(stream); });

  HIP_CHECK(hipMemsetD2D16Async(reinterpret_cast<hipDeviceptr_t>(device_ptr), pitch, pattern,
                                kWidth * sizeof(uint16_t), kHeight, stream));
  HIP_CHECK(hipMemcpy2DAsync(dst.data(), kWidth * sizeof(uint16_t), device_ptr, pitch,
                             kWidth * sizeof(uint16_t), kHeight, hipMemcpyDeviceToHost, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  RequireAllEqual(dst, pattern);
}

HIP_TEST_CASE(Contract_DriverMemsetAsync2D3D_D2D32Async_FillsDwordRows_VisibleAfterSync) {
  hip::contract::ContractCleanup cleanup;
  constexpr int pattern = 0x12345678;
  std::array<int, kWidth * kHeight> dst{};
  void* device_ptr = nullptr;
  size_t pitch = 0;
  hipStream_t stream = nullptr;

  if (!TryMallocPitch(&device_ptr, &pitch, kWidth * sizeof(int), kHeight)) {
    HIP_SKIP_TEST("hipMallocPitch is not supported by this device/runtime path.");
  }
  cleanup.Add([&] { (void)hipFree(device_ptr); });
  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([&] { (void)hipStreamDestroy(stream); });

  HIP_CHECK(hipMemsetD2D32Async(reinterpret_cast<hipDeviceptr_t>(device_ptr), pitch, pattern,
                                kWidth * sizeof(int), kHeight, stream));
  HIP_CHECK(hipMemcpy2DAsync(dst.data(), kWidth * sizeof(int), device_ptr, pitch,
                             kWidth * sizeof(int), kHeight, hipMemcpyDeviceToHost, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  RequireAllEqual(dst, pattern);
}

HIP_TEST_CASE(Contract_DriverMemsetAsync2D3D_Memset2DAsync_FillsRegion_VisibleAfterSync) {
  hip::contract::ContractCleanup cleanup;
  constexpr uint8_t pattern = 0x7b;
  std::array<uint8_t, kWidth * kHeight> dst{};
  void* device_ptr = nullptr;
  size_t pitch = 0;
  hipStream_t stream = nullptr;

  if (!TryMallocPitch(&device_ptr, &pitch, kWidth, kHeight)) {
    HIP_SKIP_TEST("hipMallocPitch is not supported by this device/runtime path.");
  }
  cleanup.Add([&] { (void)hipFree(device_ptr); });
  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([&] { (void)hipStreamDestroy(stream); });

  HIP_CHECK(hipMemset2DAsync(device_ptr, pitch, pattern, kWidth, kHeight, stream));
  HIP_CHECK(hipMemcpy2DAsync(dst.data(), kWidth, device_ptr, pitch, kWidth, kHeight,
                             hipMemcpyDeviceToHost, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  RequireAllEqual(dst, pattern);
}

HIP_TEST_CASE(Contract_DriverMemsetAsync2D3D_Memset3DAsync_FillsExtent_VisibleAfterSync) {
  hip::contract::ContractCleanup cleanup;
  constexpr uint8_t pattern = 0x5c;
  std::array<uint8_t, kWidth * kHeight * kDepth> dst{};
  hipPitchedPtr device{};
  hipStream_t stream = nullptr;
  const auto extent = make_hipExtent(kWidth, kHeight, kDepth);

  if (!TryMalloc3D(&device, extent)) {
    HIP_SKIP_TEST("hipMalloc3D is not supported by this device/runtime path.");
  }
  cleanup.Add([&] { (void)hipFree(device.ptr); });
  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([&] { (void)hipStreamDestroy(stream); });

  HIP_CHECK(hipMemset3DAsync(device, pattern, extent, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  hipMemcpy3DParms d2h{};
  d2h.srcPtr = device;
  d2h.dstPtr = HostPitchedPtr(dst.data(), kWidth, kHeight);
  d2h.extent = extent;
  d2h.kind = hipMemcpyDeviceToHost;
  HIP_CHECK(hipMemcpy3D(&d2h));

  RequireAllEqual(dst, pattern);
}

HIP_TEST_CASE(Contract_DriverMemsetAsync2D3D_NullDestination_IsRejected) {
  hip::contract::ContractCleanup cleanup;
  hipStream_t stream = nullptr;
  hipPitchedPtr null_3d{};
  const auto extent = make_hipExtent(kWidth, kHeight, kDepth);
  constexpr size_t pitch = kWidth * sizeof(uint32_t);

  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([&] { (void)hipStreamDestroy(stream); });

  const hipError_t d2d16_status = hipMemsetD2D16Async(
      hipDeviceptr_t(nullptr), pitch, 0x1357, kWidth * sizeof(uint16_t), kHeight, stream);
  const hipError_t d2d32_status = hipMemsetD2D32Async(
      hipDeviceptr_t(nullptr), pitch, 0x12345678, kWidth * sizeof(int), kHeight, stream);
  const hipError_t memset2d_status = hipMemset2DAsync(nullptr, pitch, 0x7b, kWidth, kHeight, stream);
  const hipError_t memset3d_status = hipMemset3DAsync(null_3d, 0x5c, extent, stream);

  REQUIRE(d2d16_status != hipSuccess);
  REQUIRE(d2d32_status != hipSuccess);
  REQUIRE(memset2d_status != hipSuccess);
  REQUIRE(memset3d_status != hipSuccess);
}
