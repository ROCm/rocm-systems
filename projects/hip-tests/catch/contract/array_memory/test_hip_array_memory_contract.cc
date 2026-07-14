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

std::array<uint8_t, kWidth * kHeight> MakePattern(uint8_t seed) {
  std::array<uint8_t, kWidth * kHeight> pattern{};
  for (size_t i = 0; i < pattern.size(); ++i) {
    pattern[i] = static_cast<uint8_t>(seed + i);
  }
  return pattern;
}

hipChannelFormatDesc ByteChannelDesc() { return hipCreateChannelDesc<uint8_t>(); }
}  // namespace

HIP_TEST_CASE(Contract_ArrayMemory_MallocArray_ReturnsUsableArray) {
  CHECK_IMAGE_SUPPORT;

  hip::contract::ContractCleanup cleanup;
  hipArray_t array = nullptr;
  const auto desc = ByteChannelDesc();

  HIP_CHECK(hipMallocArray(&array, &desc, kWidth, kHeight));
  cleanup.Add([&] { (void)hipFreeArray(array); });

  REQUIRE(array != nullptr);
}

HIP_TEST_CASE(Contract_ArrayMemory_Memcpy2DToArrayAndBack_RoundTripsBytes) {
  CHECK_IMAGE_SUPPORT;

  hip::contract::ContractCleanup cleanup;
  const auto src = MakePattern(0x39);
  std::array<uint8_t, kWidth * kHeight> dst{};
  hipArray_t array = nullptr;
  const auto desc = ByteChannelDesc();

  HIP_CHECK(hipMallocArray(&array, &desc, kWidth, kHeight));
  cleanup.Add([&] { (void)hipFreeArray(array); });
  HIP_CHECK(hipMemcpy2DToArray(array, 0, 0, src.data(), kWidth, kWidth, kHeight,
                               hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy2DFromArray(dst.data(), kWidth, array, 0, 0, kWidth, kHeight,
                                 hipMemcpyDeviceToHost));

  REQUIRE(dst == src);
}

HIP_TEST_CASE(Contract_ArrayMemory_FreeArray_Succeeds) {
  CHECK_IMAGE_SUPPORT;

  hipArray_t array = nullptr;
  const auto desc = ByteChannelDesc();

  HIP_CHECK(hipMallocArray(&array, &desc, kWidth, kHeight));
  HIP_CHECK(hipFreeArray(array));
}

HIP_TEST_CASE(Contract_ArrayMemory_ArrayGetInfo_ReturnsDescriptorIfAvailable) {
  CHECK_IMAGE_SUPPORT;

  hip::contract::ContractCleanup cleanup;
  hipArray_t array = nullptr;
  const auto desc = ByteChannelDesc();
  hipChannelFormatDesc returned_desc{};
  hipExtent returned_extent{};
  unsigned int returned_flags = 0;

  HIP_CHECK(hipMallocArray(&array, &desc, kWidth, kHeight));
  cleanup.Add([&] { (void)hipFreeArray(array); });
  HIP_CHECK(hipArrayGetInfo(&returned_desc, &returned_extent, &returned_flags, array));

  REQUIRE(returned_extent.width == kWidth);
  REQUIRE(returned_extent.height == kHeight);
  REQUIRE(returned_extent.depth == 0);
  REQUIRE(returned_flags == 0);
  REQUIRE(returned_desc.x == desc.x);
  REQUIRE(returned_desc.y == desc.y);
  REQUIRE(returned_desc.z == desc.z);
  REQUIRE(returned_desc.w == desc.w);
  REQUIRE(returned_desc.f == desc.f);
}
