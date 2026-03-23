/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

#include "error_handling_common.hh"

/**
 * @addtogroup hipDrvGetErrorName hipDrvGetErrorName
 * @{
 * @ingroup ErrorTest
 * `hipDrvGetErrorName(hipError_t hip_error)` -
 * Return hip error as text string form.
 */

/**
 * Test Description
 * ------------------------
 *  - Validate that the correct string is returned for each supported
 *    device error enumeration.
 * Test source
 * ------------------------
 *  - unit/errorHandling/hipDrvGetErrorName.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.4
 */
HIP_TEST_CASE(Unit_hipDrvGetErrorName_Positive_Basic) {
  const char* error_string = nullptr;
  const auto enumerator =
      GENERATE(from_range(std::begin(kErrorEnumerators), std::end(kErrorEnumerators)));
  INFO("Error: " << enumerator);

  HIP_CHECK(hipDrvGetErrorName(enumerator, &error_string));

  REQUIRE(error_string != nullptr);
  REQUIRE(strcmp(error_string, ErrorName(enumerator)) == 0);
}

/**
 * Test Description
 * ------------------------
 *  - Validate handling of invalid arguments:
 *    -# When error enumerator is invalid (-1)
 *      - AMD expected output: return "hipErrorUnknown"
 *      - NVIDIA expected output: return "cudaErrorUnknown"
 *    -# When nullptr is passed as store location
 *      - Expected output: return "hipErrorInvalidValue"
 * Test source
 * ------------------------
 *  - unit/errorHandling/hipDrvGetErrorName.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.4
 */
HIP_TEST_CASE(Unit_hipDrvGetErrorName_Negative_Parameters) {
  const char* error_string = nullptr;
  SECTION("pass unknown value to hipError") {
    HIP_CHECK_ERROR((hipDrvGetErrorName(static_cast<hipError_t>(-1), &error_string)),
                    hipErrorInvalidValue);
  }
#if HT_AMD  // segfaults on NVIDIA
  SECTION("pass nullptr to error string") {
    HIP_CHECK_ERROR((hipDrvGetErrorString(hipErrorInvalidValue, nullptr)), hipErrorInvalidValue);
  }
#endif
}

/**
 * End doxygen group ErrorTest.
 * @}
 */
