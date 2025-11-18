/*
Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
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
 * hipMemAllocationGranularityMinimum option.
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemGetAllocationGranularity.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.1
 */
TEST_CASE("Unit_hipMemGetAllocationGranularity_MinGranularity") {
  HIP_CHECK(hipFree(0));
  size_t granularity = 0;
  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, 0));
  checkVMMSupported(device);
  getGranularity(&granularity, hipMemAllocationGranularityMinimum, 0);
  REQUIRE(granularity > 0);
}

/**
 * Test Description
 * ------------------------
 *    - Functional Test to get granularity size for
 * hipMemAllocationGranularityRecommended option.
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemGetAllocationGranularity.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.1
 */
TEST_CASE("Unit_hipMemGetAllocationGranularity_RecommendedGranularity") {
  HIP_CHECK(hipFree(0));
  size_t granularity = 0;
  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, 0));
  checkVMMSupported(device);
  getGranularity(&granularity, hipMemAllocationGranularityRecommended, 0);
  REQUIRE(granularity > 0);
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
TEST_CASE("Unit_hipMemGetAllocationGranularity_AllGPUs") {
  HIP_CHECK(hipFree(0));
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  for (int dev = 0; dev < numDevices; dev++) {
    size_t granularity = 0;
    hipDevice_t device;
    HIP_CHECK(hipDeviceGet(&device, dev));
    checkVMMSupported(device);
    getGranularity(&granularity, hipMemAllocationGranularityRecommended, dev);
    REQUIRE(granularity > 0);
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
TEST_CASE("Unit_hipMemGetAllocationGranularity_NegativeTests") {
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

TEST_CASE("Unit_hipMemGetAllocationGranularity_Capture") {
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
