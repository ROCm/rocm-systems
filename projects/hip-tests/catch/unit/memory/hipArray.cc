/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
TEST_CASE(Unit_hipArray_Valid) {
  CHECK_IMAGE_SUPPORT

  hipArray_t array = nullptr;
  HIP_ARRAY_DESCRIPTOR desc;
  desc.Format = HIP_AD_FORMAT_FLOAT;
  desc.NumChannels = 1;
  desc.Width = 1024;
  desc.Height = 1024;
  HIP_CHECK(hipArrayCreate(&array, &desc));
  HIP_CHECK(hipFreeArray(array));
}

TEST_CASE(Unit_hipArray_Invalid) {
  CHECK_IMAGE_SUPPORT

  void* data = malloc(sizeof(char));
  hipArray_t arrayPtr = static_cast<hipArray_t>(data);
  REQUIRE(hipFreeArray(arrayPtr) == hipErrorContextIsDestroyed);
  free(data);
}
TEST_CASE(Unit_hipArray_Nullptr) {
  CHECK_IMAGE_SUPPORT

  hipArray_t array = nullptr;
  REQUIRE(hipFreeArray(array) == hipErrorInvalidValue);
}
TEST_CASE(Unit_hipArray_DoubleFree) {
  CHECK_IMAGE_SUPPORT

  hipArray_t array = nullptr;
  HIP_ARRAY_DESCRIPTOR desc;
  desc.Format = HIP_AD_FORMAT_FLOAT;
  desc.NumChannels = 1;
  desc.Width = 1024;
  desc.Height = 1024;
  HIP_CHECK(hipArrayCreate(&array, &desc));
  HIP_CHECK(hipFreeArray(array));
  REQUIRE(hipFreeArray(array) == hipErrorContextIsDestroyed);
}
TEST_CASE(Unit_hipArray_TrippleDestroy) {
  CHECK_IMAGE_SUPPORT

  hipArray_t array = nullptr;
  HIP_ARRAY_DESCRIPTOR desc;
  desc.Format = HIP_AD_FORMAT_FLOAT;
  desc.NumChannels = 1;
  desc.Width = 1024;
  desc.Height = 1024;
  HIP_CHECK(hipArrayCreate(&array, &desc));
  HIP_CHECK(hipArrayDestroy(array));
  REQUIRE(hipArrayDestroy(array) == hipErrorContextIsDestroyed);
  REQUIRE(hipArrayDestroy(array) == hipErrorContextIsDestroyed);
}
TEST_CASE(Unit_hipArray_DoubleNullptr) {
  CHECK_IMAGE_SUPPORT

  hipArray_t array = nullptr;
  REQUIRE(hipFreeArray(array) == hipErrorInvalidValue);
  REQUIRE(hipFreeArray(array) == hipErrorInvalidValue);
}
TEST_CASE(Unit_hipArray_DoubleInvalid) {
  CHECK_IMAGE_SUPPORT

  void* data = malloc(sizeof(char));
  hipArray_t arrayPtr = static_cast<hipArray_t>(data);
  REQUIRE(hipFreeArray(arrayPtr) == hipErrorContextIsDestroyed);
  REQUIRE(hipFreeArray(arrayPtr) == hipErrorContextIsDestroyed);
  free(data);
}
