/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <cstddef>
#include <cstdint>

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>

namespace {
bool QueryTextureWidthOrSkip(size_t* max_width, const hipChannelFormatDesc* desc, int device) {
  const hipError_t status = hipDeviceGetTexture1DLinearMaxWidth(max_width, desc, device);
  if (status == hipErrorNotSupported) {
    return false;
  }
  HIP_CHECK(status);
  return true;
}
}  // namespace

// @asserts: hipDeviceGetTexture1DLinearMaxWidth - a float-channel descriptor yields a positive max 1D linear texture width
HIP_TEST_CASE(Contract_DeviceTextureQuery_ReturnsPositiveWidth_ForFloatChannel) {
  CHECK_IMAGE_SUPPORT;

  int device = 0;
  HIP_CHECK(hipGetDevice(&device));

  size_t max_width = 0;
  const auto desc = hipCreateChannelDesc<float>();
  if (!QueryTextureWidthOrSkip(&max_width, &desc, device)) {
    HIP_SKIP_TEST(HipTest::SkipReason::kTextureImageUnsupported);
  }

  REQUIRE(max_width > 0);
}

// @asserts: hipDeviceGetTexture1DLinearMaxWidth - a byte-channel descriptor yields a positive max 1D linear texture width
HIP_TEST_CASE(Contract_DeviceTextureQuery_ReturnsPositiveWidth_ForByteChannel) {
  CHECK_IMAGE_SUPPORT;

  int device = 0;
  HIP_CHECK(hipGetDevice(&device));

  size_t max_width = 0;
  const auto desc = hipCreateChannelDesc<uint8_t>();
  if (!QueryTextureWidthOrSkip(&max_width, &desc, device)) {
    HIP_SKIP_TEST(HipTest::SkipReason::kTextureImageUnsupported);
  }

  REQUIRE(max_width > 0);
}

// @asserts: hipDeviceGetTexture1DLinearMaxWidth - rejects a null max-width output pointer with a non-success error
HIP_TEST_CASE(Contract_DeviceTextureQuery_NullMaxWidth_IsRejected) {
  CHECK_IMAGE_SUPPORT;

  int device = 0;
  HIP_CHECK(hipGetDevice(&device));

  const auto desc = hipCreateChannelDesc<float>();
  REQUIRE(hipDeviceGetTexture1DLinearMaxWidth(nullptr, &desc, device) != hipSuccess);
}

// @asserts: hipDeviceGetTexture1DLinearMaxWidth - rejects a null channel-format descriptor with a non-success error
HIP_TEST_CASE(Contract_DeviceTextureQuery_NullDescriptor_IsRejected) {
  CHECK_IMAGE_SUPPORT;

  int device = 0;
  HIP_CHECK(hipGetDevice(&device));

  size_t max_width = 0;
  REQUIRE(hipDeviceGetTexture1DLinearMaxWidth(&max_width, nullptr, device) != hipSuccess);
}

// @asserts: hipDeviceGetTexture1DLinearMaxWidth - rejects an all-zero (no channel bits) descriptor with a non-success error
HIP_TEST_CASE(Contract_DeviceTextureQuery_ZeroSizedDescriptor_IsRejected) {
  CHECK_IMAGE_SUPPORT;

  int device = 0;
  HIP_CHECK(hipGetDevice(&device));

  size_t max_width = 0;
  hipChannelFormatDesc desc{};
  REQUIRE(hipDeviceGetTexture1DLinearMaxWidth(&max_width, &desc, device) != hipSuccess);
}

// @asserts: hipDeviceGetTexture1DLinearMaxWidth - rejects an out-of-range device ordinal with a non-success error
HIP_TEST_CASE(Contract_DeviceTextureQuery_InvalidDevice_IsRejected) {
  CHECK_IMAGE_SUPPORT;

  int device_count = 0;
  HIP_CHECK(hipGetDeviceCount(&device_count));

  size_t max_width = 0;
  const auto desc = hipCreateChannelDesc<float>();
  REQUIRE(hipDeviceGetTexture1DLinearMaxWidth(&max_width, &desc, device_count + 100) != hipSuccess);
}
