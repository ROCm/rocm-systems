/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

#include "error_handling_common.hh"

/**
 * @addtogroup hipGetErrorString hipGetErrorString
 * @{
 * @ingroup ErrorTest
 * `hipGetErrorString(hipError_t hipError)` -
 * Return handy text string message to explain the error which occurred.
 */

/**
 * Test Description
 * ------------------------
 *  - Validate that a non-empty string is returned for each supported
 *    device error enumeration.
 * Test source
 * ------------------------
 *  - unit/errorHandling/hipGetErrorString.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_hipGetErrorString_Positive_Basic) {
  const char* error_string = nullptr;
  const auto enumerator =
      GENERATE(from_range(std::begin(kErrorEnumerators), std::end(kErrorEnumerators)));
  INFO("Error: " << enumerator);

  error_string = hipGetErrorString(enumerator);

  REQUIRE(error_string != nullptr);
  REQUIRE(strlen(error_string) > 0);
}

/**
 * Test Description
 * ------------------------
 *  - Validate handling of invalid arguments:
 *    -# When error enumerator is invalid (-1)
 *      - Expected output: do not return `nullptr`
 * Test source
 * ------------------------
 *  - unit/errorHandling/hipGetErrorString.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_hipGetErrorString_Negative_Parameters) {
  const char* error_string = hipGetErrorString(static_cast<hipError_t>(-1));
  REQUIRE(error_string != nullptr);
}

/**
 * End doxygen group ErrorTest.
 * @}
 */
