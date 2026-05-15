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

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <hip_test_common.hh>

/**
 * @addtogroup hipDeviceSetGraphMemAttribute hipDeviceSetGraphMemAttribute
 * @{
 * @ingroup GraphTest
 * `hipDeviceSetGraphMemAttribute(int device, hipGraphMemAttributeType attr, void* value)` -
 * Set the mem attribute for graphs.
 */

static void GraphSetGetAttribute(int device, hipGraphMemAttributeType attr, size_t set_value) {
  size_t get_value = 100;
  HIP_CHECK(hipDeviceSetGraphMemAttribute(device, attr, &set_value));
  HIP_CHECK(hipDeviceGetGraphMemAttribute(device, attr, &get_value));
  REQUIRE(get_value == set_value);
}

/**
 * Test Description
 * ------------------------
 *  - Basic test to verify that valid attributes can be reset to zero.
 * Test source
 * ------------------------
 *  - /unit/graph/hipDeviceSetGraphMemAttribute.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
TEST_CASE("Unit_hipDeviceSetGraphMemAttribute_Positive_Default") {
  const auto device = GENERATE(range(0, HipTest::getDeviceCount()));
  const auto attr_type = GENERATE(hipGraphMemAttrUsedMemHigh, hipGraphMemAttrReservedMemHigh);

  // Check if attributes can be reset
  size_t set_value = 0;
  GraphSetGetAttribute(device, attr_type, set_value);
}


/**
 * Test Description
 * ------------------------
 *  - Test to verify hipDeviceSetGraphMemAttribute behavior with invalid arguments:
 *    -# Device is not valid
 *    -# Attribute value is not supported
 *    -# Attribute value is not valid
 *    -# Set hipGraphMemAttrUsedMemHigh to non-zero
 *    -# Set hipGraphMemAttrReservedMemHigh to non-zero
 * Test source
 * ------------------------
 *  - /unit/graph/hipDeviceSetGraphMemAttribute.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
TEST_CASE("Unit_hipDeviceSetGraphMemAttribute_Negative_Parameters") {
  int device_id = 0;
  HIP_CHECK(hipSetDevice(device_id));

  int num_dev = 0;
  HIP_CHECK(hipGetDeviceCount(&num_dev));

  hipGraphMemAttributeType attr = hipGraphMemAttrUsedMemHigh;
  size_t set_value = 0;

  SECTION("device is not valid") {
    HIP_CHECK_ERROR(
        hipDeviceSetGraphMemAttribute(num_dev, attr, reinterpret_cast<void*>(&set_value)),
        hipErrorInvalidDevice);
  }

  SECTION("Attribute value is not supported") {
    HIP_CHECK_ERROR(hipDeviceSetGraphMemAttribute(0, hipGraphMemAttrUsedMemCurrent,
                                                  reinterpret_cast<void*>(&set_value)),
                    hipErrorInvalidValue);
  }

  SECTION("Attribute value is not valid") {
    HIP_CHECK_ERROR(hipDeviceSetGraphMemAttribute(0, static_cast<hipGraphMemAttributeType>(0x7),
                                                  reinterpret_cast<void*>(&set_value)),
                    hipErrorInvalidValue);
  }

  SECTION("Set hipGraphMemAttrUsedMemHigh to non-zero") {
    size_t invalid_value = 1;
    HIP_CHECK_ERROR(hipDeviceSetGraphMemAttribute(0, attr, reinterpret_cast<void*>(&invalid_value)),
                    hipErrorInvalidValue);
  }

  SECTION("Set hipGraphMemAttrReservedMemHigh to non-zero") {
    attr = hipGraphMemAttrReservedMemHigh;
    size_t invalid_value = 1;
    HIP_CHECK_ERROR(hipDeviceSetGraphMemAttribute(0, attr, reinterpret_cast<void*>(&invalid_value)),
                    hipErrorInvalidValue);
  }
}

/**
 * End doxygen group GraphTest.
 * @}
 */
