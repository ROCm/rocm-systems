/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip_test_defgroups.hh>

/**
 * @addtogroup hipArrayGetPlane hipArrayGetPlane
 * @{
 * @ingroup MemoryTest
 * `hipError_t hipArrayGetPlane(hipArray_t* pPlaneArray, hipArray_t hArray, unsigned int planeIdx)`
 * - Gets a single format plane of a multiplanar HIP array. AMD devices do not support multiplanar
 *   arrays, so this validates its arguments and returns hipErrorNotSupported.
 */

namespace {

hipArray_t CreateOrdinaryArray() {
  HIP_ARRAY3D_DESCRIPTOR desc{};
  desc.Width = 64;
  desc.Height = 64;
  desc.Depth = 0;
  desc.Format = HIP_AD_FORMAT_UNSIGNED_INT8;
  desc.NumChannels = 1;
  desc.Flags = hipArrayDefault;

  hipArray_t array = nullptr;
  HIP_CHECK(hipArray3DCreate(&array, &desc));
  return array;
}

}  // namespace

/**
 * Test Description
 * ------------------------
 *  - No HIP array format is multiplanar on AMD, so a well-formed call on a valid array reports
 *    that the device does not support the feature.
 * Test source
 * ------------------------
 *  - unit/memory/hipArrayGetPlane.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 */
TEST_CASE("Unit_hipArrayGetPlane_Negative_NotSupported") {
  hipArray_t array = CreateOrdinaryArray();
  hipArray_t plane = nullptr;

  REQUIRE(hipArrayGetPlane(&plane, array, 0) == hipErrorNotSupported);

  HIP_CHECK(hipArrayDestroy(array));
}

/**
 * Test Description
 * ------------------------
 *  - Verifies the argument contract: a null output pointer is an invalid value, while a null or
 *    unknown array handle is an invalid handle.
 * Test source
 * ------------------------
 *  - unit/memory/hipArrayGetPlane.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 */
TEST_CASE("Unit_hipArrayGetPlane_Negative_Parameters") {
  hipArray_t array = CreateOrdinaryArray();
  hipArray_t plane = nullptr;

  SECTION("pPlaneArray is nullptr") {
    REQUIRE(hipArrayGetPlane(nullptr, array, 0) == hipErrorInvalidValue);
  }

  SECTION("hArray is nullptr") {
    REQUIRE(hipArrayGetPlane(&plane, nullptr, 0) == hipErrorInvalidHandle);
  }

  SECTION("hArray is not a valid array") {
    hipArray_t destroyed = CreateOrdinaryArray();
    HIP_CHECK(hipArrayDestroy(destroyed));
    REQUIRE(hipArrayGetPlane(&plane, destroyed, 0) == hipErrorInvalidHandle);
  }

  HIP_CHECK(hipArrayDestroy(array));
}

/**
* End doxygen group MemoryTest.
* @}
*/
