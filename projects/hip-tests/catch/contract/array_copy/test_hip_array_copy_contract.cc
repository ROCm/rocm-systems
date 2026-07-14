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
constexpr size_t kByteCount = kWidth * kHeight;

std::array<uint8_t, kByteCount> MakePattern(uint8_t seed) {
  std::array<uint8_t, kByteCount> pattern{};
  for (size_t i = 0; i < pattern.size(); ++i) {
    pattern[i] = static_cast<uint8_t>(seed + i);
  }
  return pattern;
}

hipChannelFormatDesc ByteChannelDesc() { return hipCreateChannelDesc<uint8_t>(); }
}  // namespace

HIP_TEST_CASE(Contract_ArrayCopy_MemcpyToArrayAndFromArray_RoundTripsBytes) {
  CHECK_IMAGE_SUPPORT;

  hip::contract::ContractCleanup cleanup;
  const auto src = MakePattern(0x12);
  std::array<uint8_t, kByteCount> dst{};
  hipArray_t array = nullptr;
  const auto desc = ByteChannelDesc();

  HIP_CHECK(hipMallocArray(&array, &desc, kWidth, kHeight));
  cleanup.Add([&] { (void)hipFreeArray(array); });
  HIP_CHECK(hipMemcpyToArray(array, 0, 0, src.data(), src.size(), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpyFromArray(dst.data(), array, 0, 0, dst.size(), hipMemcpyDeviceToHost));

  REQUIRE(dst == src);
}

HIP_TEST_CASE(Contract_ArrayCopy_MemcpyHtoAAndAtoH_RoundTripsBytes) {
  CHECK_IMAGE_SUPPORT;

  hip::contract::ContractCleanup cleanup;
  const auto src = MakePattern(0x34);
  std::array<uint8_t, kByteCount> dst{};
  hipArray_t array = nullptr;
  const auto desc = ByteChannelDesc();

  HIP_CHECK(hipMallocArray(&array, &desc, kByteCount, 1));
  cleanup.Add([&] { (void)hipFreeArray(array); });
  HIP_CHECK(hipMemcpyHtoA(array, 0, src.data(), src.size()));
  HIP_CHECK(hipMemcpyAtoH(dst.data(), array, 0, dst.size()));

  REQUIRE(dst == src);
}

HIP_TEST_CASE(Contract_ArrayCopy_Memcpy2DToFromArrayAsync_VisibleAfterSync) {
  CHECK_IMAGE_SUPPORT;

  hip::contract::ContractCleanup cleanup;
  const auto src = MakePattern(0x56);
  std::array<uint8_t, kByteCount> dst{};
  hipArray_t array = nullptr;
  hipStream_t stream = nullptr;
  const auto desc = ByteChannelDesc();

  HIP_CHECK(hipMallocArray(&array, &desc, kWidth, kHeight));
  cleanup.Add([&] { (void)hipFreeArray(array); });
  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([&] { (void)hipStreamDestroy(stream); });

  HIP_CHECK(hipMemcpy2DToArrayAsync(array, 0, 0, src.data(), kWidth, kWidth, kHeight,
                                    hipMemcpyHostToDevice, stream));
  HIP_CHECK(hipMemcpy2DFromArrayAsync(dst.data(), kWidth, array, 0, 0, kWidth, kHeight,
                                      hipMemcpyDeviceToHost, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  REQUIRE(dst == src);
}

HIP_TEST_CASE(Contract_ArrayCopy_MemcpyToArray_NullArray_IsRejected) {
  CHECK_IMAGE_SUPPORT;

  const auto src = MakePattern(0x9a);

  HIP_CHECK(hipGetLastError());
  const hipError_t status =
      hipMemcpyToArray(nullptr, 0, 0, src.data(), src.size(), hipMemcpyHostToDevice);

  REQUIRE(status != hipSuccess);
  HIP_CHECK_ERROR(hipGetLastError(), status);
  HIP_CHECK(hipGetLastError());
}

HIP_TEST_CASE(Contract_ArrayCopy_Memcpy2DFromArrayAsync_InvalidKind_IsRejected) {
  CHECK_IMAGE_SUPPORT;

  hip::contract::ContractCleanup cleanup;
  std::array<uint8_t, kByteCount> dst{};
  hipArray_t array = nullptr;
  hipStream_t stream = nullptr;
  const auto desc = ByteChannelDesc();

  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMallocArray(&array, &desc, kWidth, kHeight));
  cleanup.Add([&] { (void)hipFreeArray(array); });
  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([&] { (void)hipStreamDestroy(stream); });

  const hipError_t status = hipMemcpy2DFromArrayAsync(
      dst.data(), kWidth, array, 0, 0, kWidth, kHeight, static_cast<hipMemcpyKind>(-1), stream);

  REQUIRE(status != hipSuccess);
  HIP_CHECK_ERROR(hipGetLastError(), status);
  HIP_CHECK(hipGetLastError());
}
