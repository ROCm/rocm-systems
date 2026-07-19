/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>
#include <contract_cleanup.hh>

namespace {
constexpr size_t kWidth = 8;
constexpr size_t kHeight = 4;
constexpr size_t kDepth = 2;

// Establishes a device context before the driver-style array entry points below.
// On the NVIDIA backend hipArrayCreate maps to the driver API, which requires a
// bound primary context; a test that calls it before any allocation would
// otherwise fail with an initialization / "invalid device context" error.
// hipFree(0) is the canonical no-op that forces primary-context initialization,
// and is a harmless success on AMD where the runtime already auto-initializes.
void EnsureContext() { HIP_CHECK(hipFree(0)); }

HIP_ARRAY_DESCRIPTOR Array2DDesc() {
  HIP_ARRAY_DESCRIPTOR desc{};
  desc.Width = kWidth;
  desc.Height = kHeight;
  desc.Format = HIP_AD_FORMAT_UNSIGNED_INT8;
  desc.NumChannels = 1;
  return desc;
}

HIP_ARRAY3D_DESCRIPTOR Array3DDesc() {
  HIP_ARRAY3D_DESCRIPTOR desc{};
  desc.Width = kWidth;
  desc.Height = kHeight;
  desc.Depth = kDepth;
  desc.Format = HIP_AD_FORMAT_UNSIGNED_INT8;
  desc.NumChannels = 1;
  desc.Flags = 0;
  return desc;
}
}  // namespace

// @asserts: hipArrayCreate - creating a 2D array from a valid descriptor yields a non-null array handle
HIP_TEST_CASE(Contract_DriverArray_ArrayCreate_2D_ReturnsUsableArray) {
  CHECK_IMAGE_SUPPORT;
  EnsureContext();

  hip::contract::ContractCleanup cleanup;
  hipArray_t array = nullptr;
  auto desc = Array2DDesc();

  HIP_CHECK(hipArrayCreate(&array, &desc));
  cleanup.Add([array] { (void)hipArrayDestroy(array); });

  REQUIRE(array != nullptr);
}

// @asserts: hipArrayGetDescriptor - reads back the width, height, format, and channel count the array was created with
HIP_TEST_CASE(Contract_DriverArray_GetDescriptor_RoundTripsDimsAndFormat) {
  CHECK_IMAGE_SUPPORT;
  EnsureContext();

  hip::contract::ContractCleanup cleanup;
  hipArray_t array = nullptr;
  auto desc = Array2DDesc();
  HIP_ARRAY_DESCRIPTOR returned_desc{};

  HIP_CHECK(hipArrayCreate(&array, &desc));
  cleanup.Add([array] { (void)hipArrayDestroy(array); });
  HIP_CHECK(hipArrayGetDescriptor(&returned_desc, array));

  REQUIRE(returned_desc.Width == desc.Width);
  REQUIRE(returned_desc.Height == desc.Height);
  REQUIRE(returned_desc.Format == desc.Format);
  REQUIRE(returned_desc.NumChannels == desc.NumChannels);
}

// @asserts: hipArrayCreate - rejects a null array-out pointer or null descriptor with a non-success error
HIP_TEST_CASE(Contract_DriverArray_ArrayCreate_InvalidArgs_AreRejected) {
  CHECK_IMAGE_SUPPORT;
  EnsureContext();

  hipArray_t array = nullptr;
  auto desc = Array2DDesc();

  REQUIRE(hipArrayCreate(nullptr, &desc) != hipSuccess);
  REQUIRE(hipArrayCreate(&array, nullptr) != hipSuccess);
}

// @asserts: hipArrayGetDescriptor - rejects a null descriptor-out pointer or null array handle with a non-success error
HIP_TEST_CASE(Contract_DriverArray_GetDescriptor_InvalidArgs_AreRejected) {
  CHECK_IMAGE_SUPPORT;
  EnsureContext();

  hip::contract::ContractCleanup cleanup;
  hipArray_t array = nullptr;
  auto desc = Array2DDesc();
  HIP_ARRAY_DESCRIPTOR returned_desc{};

  HIP_CHECK(hipArrayCreate(&array, &desc));
  cleanup.Add([array] { (void)hipArrayDestroy(array); });

  REQUIRE(hipArrayGetDescriptor(nullptr, array) != hipSuccess);
  REQUIRE(hipArrayGetDescriptor(&returned_desc, nullptr) != hipSuccess);
}

// @asserts: hipArray3DCreate - creating a 3D array from a valid descriptor yields a non-null array handle
HIP_TEST_CASE(Contract_DriverArray_Array3DCreate_ReturnsUsableArray) {
  CHECK_IMAGE_SUPPORT;
  EnsureContext();

  hip::contract::ContractCleanup cleanup;
  hipArray_t array = nullptr;
  auto desc = Array3DDesc();

  HIP_CHECK(hipArray3DCreate(&array, &desc));
  cleanup.Add([array] { (void)hipArrayDestroy(array); });

  REQUIRE(array != nullptr);
}

// @asserts: hipArray3DGetDescriptor - reads back the dims, depth, format, channels, and flags the 3D array was created with
HIP_TEST_CASE(Contract_DriverArray_Array3DGetDescriptor_RoundTripsDepthAndFlags) {
  CHECK_IMAGE_SUPPORT;
  EnsureContext();

  hip::contract::ContractCleanup cleanup;
  hipArray_t array = nullptr;
  auto desc = Array3DDesc();
  HIP_ARRAY3D_DESCRIPTOR returned_desc{};

  HIP_CHECK(hipArray3DCreate(&array, &desc));
  cleanup.Add([array] { (void)hipArrayDestroy(array); });
  HIP_CHECK(hipArray3DGetDescriptor(&returned_desc, array));

  REQUIRE(returned_desc.Width == desc.Width);
  REQUIRE(returned_desc.Height == desc.Height);
  REQUIRE(returned_desc.Depth == desc.Depth);
  REQUIRE(returned_desc.Format == desc.Format);
  REQUIRE(returned_desc.NumChannels == desc.NumChannels);
  REQUIRE(returned_desc.Flags == desc.Flags);
}
