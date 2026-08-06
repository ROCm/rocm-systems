/*
 * Copyright Advanced Micro Devices, Inc.
 *
 * SPDX-License-Identifier: MIT
 */

#include <cstring>

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>

namespace {
int CurrentDevice() {
  int device = -1;
  HIP_CHECK(hipGetDevice(&device));
  return device;
}

hipDeviceProp_t CurrentDeviceProperties() {
  hipDeviceProp_t properties{};
  HIP_CHECK(hipGetDeviceProperties(&properties, CurrentDevice()));
  return properties;
}
}

// @asserts: hipGetDeviceProperties - succeeds in populating properties for the current device
HIP_TEST_CASE(Contract_Device_HipGetDeviceProperties_GetProperties_SucceedsForCurrentDevice) {
  hipDeviceProp_t properties{};

  HIP_CHECK(hipGetDeviceProperties(&properties, CurrentDevice()));
}

// @asserts: hipGetDeviceProperties - the device name string is non-empty
HIP_TEST_CASE(Contract_Device_HipGetDeviceProperties_Name_IsNonEmpty) {
  const auto properties = CurrentDeviceProperties();

  REQUIRE(std::strlen(properties.name) > 0);
}

// @asserts: hipGetDeviceProperties - reported total global memory is positive
HIP_TEST_CASE(Contract_Device_HipGetDeviceProperties_TotalGlobalMem_IsPositive) {
  const auto properties = CurrentDeviceProperties();

  REQUIRE(properties.totalGlobalMem > 0);
}

// @asserts: hipGetDeviceProperties - reported multiprocessor count is positive
HIP_TEST_CASE(Contract_Device_HipGetDeviceProperties_MultiProcessorCount_IsPositive) {
  const auto properties = CurrentDeviceProperties();

  REQUIRE(properties.multiProcessorCount > 0);
}

// @asserts: hipGetDeviceProperties - reported warp size is positive
HIP_TEST_CASE(Contract_Device_HipGetDeviceProperties_WarpSize_IsPositive) {
  const auto properties = CurrentDeviceProperties();

  REQUIRE(properties.warpSize > 0);
}

// @asserts: hipDeviceGetAttribute - hipDeviceAttributeWarpSize matches the warp size from hipGetDeviceProperties
HIP_TEST_CASE(Contract_Device_HipDeviceGetAttribute_WarpSize_MatchesProperties) {
  const auto properties = CurrentDeviceProperties();
  int attribute_warp_size = 0;

  HIP_CHECK(hipDeviceGetAttribute(&attribute_warp_size, hipDeviceAttributeWarpSize, CurrentDevice()));

  REQUIRE(attribute_warp_size == properties.warpSize);
}

// @asserts: hipGetDevice - the current device ordinal is in range [0, device_count)
HIP_TEST_CASE(Contract_Device_HipGetDevice_CurrentOrdinal_IsWithinDeviceCount) {
  int device_count = 0;
  const int current_device = CurrentDevice();

  HIP_CHECK(hipGetDeviceCount(&device_count));

  REQUIRE(device_count > 0);
  REQUIRE(current_device >= 0);
  REQUIRE(current_device < device_count);
}
