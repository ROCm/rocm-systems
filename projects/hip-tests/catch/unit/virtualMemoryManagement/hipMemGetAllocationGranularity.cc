/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup hipMemGetAllocationGranularity hipMemGetAllocationGranularity
 * @{
 * @ingroup VirtualMemoryManagementTest
 * `hipError_t hipMemGetAllocationGranularity (size_t* granularity,
 *                                             const hipMemAllocationProp* prop,
 *                                             hipMemAllocationGranularity_flags option)` -
 * Calculates either the minimal or recommended granularity.
 */

#include <hip_test_checkers.hh>
#include <hip_test_kernels.hh>
#include <hip_test_common.hh>

#include "hip_vmm_common.hh"

/**
 local function to invoke hipMemGetAllocationGranularity.
 */
void getGranularity(size_t* granularity, hipMemAllocationGranularity_flags option, int device) {
  hipMemAllocationProp prop{};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = device;  // Current Devices
  HIP_CHECK(hipMemGetAllocationGranularity(granularity, &prop, option));
}

/**
 * Test Description
 * ------------------------
 *    - Functional Test to get granularity size for
 * hipMemAllocationGranularityMinimum option for all GPUs.
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemGetAllocationGranularity.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.1
 */
TEST_CASE(Unit_hipMemGetAllocationGranularity_AllGPUs) {
  HIP_CHECK(hipFree(0));
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  for (int dev = 0; dev < numDevices; dev++) {
    hipDevice_t device;
    HIP_CHECK(hipDeviceGet(&device, dev));
    checkVMMSupported(device);

    size_t min_granularity = 0;
    size_t recommended_granularity = 0;

    getGranularity(&min_granularity, hipMemAllocationGranularityMinimum, dev);
    REQUIRE(min_granularity >= 1024);

    getGranularity(&recommended_granularity, hipMemAllocationGranularityRecommended, dev);
    REQUIRE(recommended_granularity >= 1024);

    // Check the recommended_granularity is greater than or equal to the minimum
    REQUIRE(recommended_granularity >= min_granularity);
  }
}

/**
 * Test Description
 * ------------------------
 *    - Negative Tests
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemGetAllocationGranularity.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.1
 */
TEST_CASE(Unit_hipMemGetAllocationGranularity_NegativeTests) {
  HIP_CHECK(hipFree(0));
  size_t granularity = 0;
  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, 0));
  checkVMMSupported(device);
  hipMemAllocationProp prop{};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = 0;  // Current Devices

  SECTION("Granularity is nullptr") {
    REQUIRE(hipErrorInvalidValue ==
            hipMemGetAllocationGranularity(nullptr, &prop, hipMemAllocationGranularityMinimum));
  }
  SECTION("Prop is nullptr") {
    REQUIRE(hipErrorInvalidValue == hipMemGetAllocationGranularity(
                                        &granularity, nullptr, hipMemAllocationGranularityMinimum));
  }

  SECTION("flag is invalid") {
    REQUIRE(hipErrorInvalidValue ==
            hipMemGetAllocationGranularity(&granularity, &prop,
                                           (hipMemAllocationGranularity_flags)0xff));
  }

#if HT_AMD  // succeeds on NVIDIA
  SECTION("device id > highest device id") {
    int numDevices = 0;
    HIP_CHECK(hipGetDeviceCount(&numDevices));
    prop.location.id = numDevices;  // set to non existing device
    REQUIRE(hipErrorInvalidValue == hipMemGetAllocationGranularity(
                                        &granularity, &prop, hipMemAllocationGranularityMinimum));
  }
  SECTION("device id < lowest device id") {
    prop.location.id = -1;  // set to non existing device
    REQUIRE(hipErrorInvalidValue == hipMemGetAllocationGranularity(
                                        &granularity, &prop, hipMemAllocationGranularityMinimum));
  }
  SECTION("allocation type as invalid") {
    prop.type = hipMemAllocationTypeInvalid;
    REQUIRE(hipErrorInvalidValue == hipMemGetAllocationGranularity(
                                        &granularity, &prop, hipMemAllocationGranularityMinimum));
  }
  SECTION("location type as invalid") {
    prop.location.type = hipMemLocationTypeInvalid;
    REQUIRE(hipErrorInvalidValue == hipMemGetAllocationGranularity(
                                        &granularity, &prop, hipMemAllocationGranularityMinimum));
  }
#endif
}

TEST_CASE(Unit_hipMemGetAllocationGranularity_Capture) {
  CTX_CREATE();

  size_t granularity = 0;
  constexpr int kDeviceId = 0;
  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, kDeviceId));

  hipMemAllocationProp allocation_prop{};
  allocation_prop.type = hipMemAllocationTypePinned;
  allocation_prop.location.type = hipMemLocationTypeDevice;
  allocation_prop.location.id = device;  // Current Device

  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));

  GENERATE_CAPTURE();
  BEGIN_CAPTURE(stream);
  HIP_CHECK(hipMemGetAllocationGranularity(&granularity, &allocation_prop,
                                           hipMemAllocationGranularityMinimum));
  END_CAPTURE(stream);

  HIP_CHECK(hipStreamDestroy(stream));

  CTX_DESTROY();
}


/**
 * End doxygen group VirtualMemoryManagementTest.
 * @}
 */
