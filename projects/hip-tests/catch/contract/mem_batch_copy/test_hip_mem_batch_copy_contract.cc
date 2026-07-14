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
constexpr size_t kBytes = 256;

int CurrentDevice() {
  int device = 0;
  HIP_CHECK(hipGetDevice(&device));
  return device;
}

std::array<uint8_t, kBytes> MakePattern(uint8_t seed) {
  std::array<uint8_t, kBytes> pattern{};
  for (size_t i = 0; i < pattern.size(); ++i) {
    pattern[i] = static_cast<uint8_t>(seed + i);
  }
  return pattern;
}

hipMemLocation DeviceLocation() {
  hipMemLocation location{};
  location.type = hipMemLocationTypeDevice;
  location.id = CurrentDevice();
  return location;
}
}  // namespace

HIP_TEST_CASE(Contract_MemBatchCopy_TwoOps_RoundTripBytes) {
  hip::contract::ContractCleanup cleanup;
  const auto src_a = MakePattern(0x10);
  const auto src_b = MakePattern(0x90);
  std::array<uint8_t, kBytes> out_a{};
  std::array<uint8_t, kBytes> out_b{};

  void* dev_src_a = nullptr;
  void* dev_src_b = nullptr;
  void* dev_dst_a = nullptr;
  void* dev_dst_b = nullptr;
  hipStream_t stream = nullptr;

  HIP_CHECK(hipMalloc(&dev_src_a, kBytes));
  cleanup.Add([&] { (void)hipFree(dev_src_a); });
  HIP_CHECK(hipMalloc(&dev_src_b, kBytes));
  cleanup.Add([&] { (void)hipFree(dev_src_b); });
  HIP_CHECK(hipMalloc(&dev_dst_a, kBytes));
  cleanup.Add([&] { (void)hipFree(dev_dst_a); });
  HIP_CHECK(hipMalloc(&dev_dst_b, kBytes));
  cleanup.Add([&] { (void)hipFree(dev_dst_b); });
  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([&] { (void)hipStreamDestroy(stream); });

  HIP_CHECK(hipMemcpy(dev_src_a, src_a.data(), kBytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dev_src_b, src_b.data(), kBytes, hipMemcpyHostToDevice));

  // A batch of two independent device-to-device copies must deliver each source
  // to its matching destination after the stream is synchronized, behaving like
  // two ordinary device-to-device copies issued together. With no per-copy
  // attributes the default access ordering applies.
  void* dsts[2] = {dev_dst_a, dev_dst_b};
  void* srcs[2] = {dev_src_a, dev_src_b};
  size_t sizes[2] = {kBytes, kBytes};
  size_t fail_index = 0;
  const hipError_t status =
      hipMemcpyBatchAsync(dsts, srcs, sizes, 2, nullptr, nullptr, 0, &fail_index, stream);
  if (status == hipErrorNotSupported) {
    HIP_SKIP_TEST("Batch memcpy is not supported by this device/runtime path.");
  }
  HIP_CHECK(status);
  HIP_CHECK(hipStreamSynchronize(stream));

  HIP_CHECK(hipMemcpy(out_a.data(), dev_dst_a, kBytes, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(out_b.data(), dev_dst_b, kBytes, hipMemcpyDeviceToHost));
  REQUIRE(out_a == src_a);
  REQUIRE(out_b == src_b);
}

HIP_TEST_CASE(Contract_MemBatchCopy_WithAttributes_RoundTripBytes) {
  hip::contract::ContractCleanup cleanup;
  const auto src = MakePattern(0x24);
  std::array<uint8_t, kBytes> out{};

  void* dev_src = nullptr;
  void* dev_dst = nullptr;
  hipStream_t stream = nullptr;

  HIP_CHECK(hipMalloc(&dev_src, kBytes));
  cleanup.Add([&] { (void)hipFree(dev_src); });
  HIP_CHECK(hipMalloc(&dev_dst, kBytes));
  cleanup.Add([&] { (void)hipFree(dev_dst); });
  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([&] { (void)hipStreamDestroy(stream); });
  HIP_CHECK(hipMemcpy(dev_src, src.data(), kBytes, hipMemcpyHostToDevice));

  // A per-copy attribute selecting stream access ordering and device location
  // hints must not change the copy result: the destination still receives the
  // source bytes. The attribute index maps the single attribute onto the single
  // copy operation.
  hipMemcpyAttributes attribute{};
  attribute.srcAccessOrder = hipMemcpySrcAccessOrderStream;
  attribute.srcLocHint = DeviceLocation();
  attribute.dstLocHint = DeviceLocation();

  void* dsts[1] = {dev_dst};
  void* srcs[1] = {dev_src};
  size_t sizes[1] = {kBytes};
  hipMemcpyAttributes attrs[1] = {attribute};
  size_t attr_indices[1] = {0};
  size_t fail_index = 0;
  const hipError_t status =
      hipMemcpyBatchAsync(dsts, srcs, sizes, 1, attrs, attr_indices, 1, &fail_index, stream);
  if (status == hipErrorNotSupported) {
    HIP_SKIP_TEST("Batch memcpy is not supported by this device/runtime path.");
  }
  HIP_CHECK(status);
  HIP_CHECK(hipStreamSynchronize(stream));

  HIP_CHECK(hipMemcpy(out.data(), dev_dst, kBytes, hipMemcpyDeviceToHost));
  REQUIRE(out == src);
}

HIP_TEST_CASE(Contract_MemBatchCopy_NullDestination_IsRejected) {
  hip::contract::ContractCleanup cleanup;
  const auto src = MakePattern(0x33);
  void* dev_src = nullptr;
  hipStream_t stream = nullptr;

  HIP_CHECK(hipMalloc(&dev_src, kBytes));
  cleanup.Add([&] { (void)hipFree(dev_src); });
  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([&] { (void)hipStreamDestroy(stream); });
  HIP_CHECK(hipMemcpy(dev_src, src.data(), kBytes, hipMemcpyHostToDevice));

  // A batch containing a null destination must not silently succeed. The exact
  // error code is backend-specific, so only a non-success status is required.
  // The sticky error left by the rejected call is cleared afterward so it does
  // not leak into later tests.
  HIP_CHECK(hipGetLastError());
  void* dsts[1] = {nullptr};
  void* srcs[1] = {dev_src};
  size_t sizes[1] = {kBytes};
  size_t fail_index = 0;
  const hipError_t status =
      hipMemcpyBatchAsync(dsts, srcs, sizes, 1, nullptr, nullptr, 0, &fail_index, stream);
  if (status == hipErrorNotSupported) {
    HIP_SKIP_TEST("Batch memcpy is not supported by this device/runtime path.");
  }
  REQUIRE(status != hipSuccess);
  (void)hipGetLastError();
}
