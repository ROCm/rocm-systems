/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

#include "error_handling_common.hh"

/**
 * @addtogroup hipDrvGetErrorString hipDrvGetErrorString
 * @{
 * @ingroup ErrorTest
 * `hipDrvGetErrorString(hipError_t hipError)` -
 * Return handy text string message to explain the error which occurred.
 */

/**
 * Test Description
 * ------------------------
 *  - Validate that the correct string is returned for each supported
 *    device error enumeration.
 * Test source
 * ------------------------
 *  - unit/errorHandling/hipDrvGetErrorString.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.4
 */
TEST_CASE(Unit_hipDrvGetErrorString_Positive_Basic) {
  const char* error_string = nullptr;
  const auto enumerator =
      GENERATE(from_range(std::begin(kErrorEnumerators), std::end(kErrorEnumerators)));
  INFO("Error: " << enumerator);

  HIP_CHECK(hipDrvGetErrorString(enumerator, &error_string));

  REQUIRE(error_string != nullptr);
  REQUIRE(strcmp(error_string, ErrorString(enumerator)) == 0);
}

/**
 * Test Description
 * ------------------------
 *  - Validate handling of invalid arguments:
 *    -# When error enumerator is invalid (-1)
 *      - Expected output: return "hipErrorInvalidValue"
 *    -# When nullptr is passed as store location
 *      - Expected output: return "hipErrorInvalidValue"
 * Test source
 * ------------------------
 *  - unit/errorHandling/hipDrvGetErrorString.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.4
 */
TEST_CASE(Unit_hipDrvGetErrorString_Negative_Parameters) {
  const char* error_string = nullptr;
  SECTION("pass unknown value to hipError") {
    HIP_CHECK_ERROR((hipDrvGetErrorString(static_cast<hipError_t>(-1), &error_string)),
                    hipErrorInvalidValue);
  }
#if HT_AMD  // segfaults on NVIDIA
  SECTION("pass nullptr to error string") {
    HIP_CHECK_ERROR((hipDrvGetErrorString(static_cast<hipError_t>(0), nullptr)),
                    hipErrorInvalidValue);
  }
#endif
}

/**
 * End doxygen group ErrorTest.
 * @}
 */
