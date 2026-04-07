/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
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
HIP_TEST_CASE(Unit_hipDeviceSetGraphMemAttribute_Positive_Default) {
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
HIP_TEST_CASE(Unit_hipDeviceSetGraphMemAttribute_Negative_Parameters) {
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
