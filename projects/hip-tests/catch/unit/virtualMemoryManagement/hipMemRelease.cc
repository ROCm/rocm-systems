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
 * @addtogroup hipMemRelease hipMemRelease
 * @{
 * @ingroup VirtualMemoryManagementTest
 * `hipMemRelease(hipMemGenericAllocationHandle_t handle)` -
 * Release a memory handle representing a memory allocation which was previously
 * allocated through hipMemCreate.
 */

#include <hip_test_common.hh>

/**
 * Test Description
 * ------------------------
 *    - Negative Tests
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemRelease.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.1
 */
TEST_CASE("Unit_hipMemRelease_negative") {
  CTX_CREATE();
  SECTION("Nullptr to handle") {
    REQUIRE(hipMemRelease((hipMemGenericAllocationHandle_t) nullptr) == hipErrorInvalidValue);
  }
  CTX_DESTROY();
}

TEST_CASE("Unit_hipMemRelease_Capture") {
  CTX_CREATE();

  hipMemGenericAllocationHandle_t allocation_handle;
  size_t allocation_granularity = 0;
  int device_id = 0;
  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, device_id));

  hipMemAllocationProp allocation_prop{};
  allocation_prop.type = hipMemAllocationTypePinned;
  allocation_prop.location.type = hipMemLocationTypeDevice;
  allocation_prop.location.id = device;

  HIP_CHECK(hipMemGetAllocationGranularity(&allocation_granularity, &allocation_prop,
                                           hipMemAllocationGranularityMinimum));
  HIP_CHECK(hipMemCreate(&allocation_handle, allocation_granularity, &allocation_prop, 0));

  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));
  GENERATE_CAPTURE();
  BEGIN_CAPTURE(stream);
  HIP_CHECK(hipMemRelease(allocation_handle));
  END_CAPTURE(stream);
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * End doxygen group VirtualMemoryManagementTest.
 * @}
 */
