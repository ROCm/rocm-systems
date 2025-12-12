/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "streamCommon.hh"

HIP_TEST_CASE(Unit_hipStreamCreate_default) {
  int id = GENERATE(range(0, HipTest::getDeviceCount()));
  HIP_CHECK(hipSetDevice(id));

  hipStream_t stream{nullptr};
  HIP_CHECK(hipStreamCreate(&stream));
  REQUIRE(stream != nullptr);         // Check if stream has a valid ptr
  REQUIRE(hip::checkStream(stream));  // check its flags and priority
  HIP_CHECK(hipStreamDestroy(stream));
}

HIP_TEST_CASE(Unit_hipStreamCreate_Negative) {
  REQUIRE(hipErrorInvalidValue == hipStreamCreate(nullptr));
}
