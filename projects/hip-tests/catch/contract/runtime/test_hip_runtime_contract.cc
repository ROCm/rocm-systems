/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <cstring>

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>

HIP_TEST_CASE(Contract_Runtime_InitZeroFlags_Succeeds) {
  HIP_CHECK(hipInit(0));
}

HIP_TEST_CASE(Contract_Runtime_GetDeviceCount_ReturnsPositiveCount) {
  int device_count = 0;

  HIP_CHECK(hipGetDeviceCount(&device_count));

  REQUIRE(device_count > 0);
}

HIP_TEST_CASE(Contract_Runtime_GetDevice_ReturnsValidOrdinal) {
  int device_count = 0;
  int device = -1;

  HIP_CHECK(hipGetDeviceCount(&device_count));
  HIP_CHECK(hipGetDevice(&device));

  REQUIRE(device_count > 0);
  REQUIRE(device >= 0);
  REQUIRE(device < device_count);
}

HIP_TEST_CASE(Contract_Runtime_SetCurrentDevice_RoundTripsThroughGetDevice) {
  int original_device = 0;
  int current_device = -1;

  HIP_CHECK(hipGetDevice(&original_device));
  HIP_CHECK(hipSetDevice(original_device));
  HIP_CHECK(hipGetDevice(&current_device));

  REQUIRE(current_device == original_device);
}

HIP_TEST_CASE(Contract_Runtime_RuntimeVersion_ReturnsPositiveVersion) {
  int runtime_version = 0;

  HIP_CHECK(hipRuntimeGetVersion(&runtime_version));

  REQUIRE(runtime_version > 0);
}

HIP_TEST_CASE(Contract_Runtime_DriverVersion_ReturnsNonNegativeVersion) {
  int driver_version = -1;

  HIP_CHECK(hipDriverGetVersion(&driver_version));

  REQUIRE(driver_version >= 0);
}

HIP_TEST_CASE(Contract_Runtime_GetErrorNameSuccess_ReturnsSuccessString) {
  const char* name = hipGetErrorName(hipSuccess);

  REQUIRE(name != nullptr);
  REQUIRE(std::strlen(name) > 0);
  REQUIRE(std::strstr(name, "Success") != nullptr);
}

HIP_TEST_CASE(Contract_Runtime_GetErrorStringSuccess_ReturnsSuccessString) {
  const char* message = hipGetErrorString(hipSuccess);

  REQUIRE(message != nullptr);
  REQUIRE(std::strlen(message) > 0);
}

HIP_TEST_CASE(Contract_Runtime_GetLastError_ClearsPreviousError) {
  HIP_CHECK(hipGetLastError());
  const hipError_t error = hipMalloc(nullptr, 1);
  REQUIRE(error != hipSuccess);
  HIP_CHECK_ERROR(hipGetLastError(), error);
  HIP_CHECK(hipGetLastError());
}

HIP_TEST_CASE(Contract_Runtime_PeekAtLastError_DoesNotClearPreviousError) {
  HIP_CHECK(hipGetLastError());
  const hipError_t error = hipMalloc(nullptr, 1);
  REQUIRE(error != hipSuccess);
  HIP_CHECK_ERROR(hipPeekAtLastError(), error);
  HIP_CHECK_ERROR(hipGetLastError(), error);
  HIP_CHECK(hipPeekAtLastError());
}
