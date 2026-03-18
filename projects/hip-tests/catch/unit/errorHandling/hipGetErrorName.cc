/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

#include "error_handling_common.hh"

/**
 * @addtogroup hipGetErrorName hipGetErrorName
 * @{
 * @ingroup ErrorTest
 * `hipGetErrorName(hipError_t hip_error)` -
 * Return hip error as text string form.
 */

/**
 * Test Description
 * ------------------------
 *  - Validate that a non-empty string is returned for each supported
 *    device error enumeration.
 * Test source
 * ------------------------
 *  - unit/errorHandling/hipGetErrorName.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE(Unit_hipGetErrorName_Positive_Basic) {
  const char* error_string = nullptr;
  const auto enumerator =
      GENERATE(from_range(std::begin(kErrorEnumerators), std::end(kErrorEnumerators)));
  INFO("Error: " << enumerator);

  error_string = hipGetErrorName(enumerator);

  REQUIRE(error_string != nullptr);
  REQUIRE(strlen(error_string) > 0);
}

/**
 * Test Description
 * ------------------------
 *  - Validate handling of invalid arguments:
 *    -# When error enumerator is invalid (-1)
 *      - AMD expected output: return "hipErrorUnknown"
 *      - NVIDIA expected output: return "cudaErrorUnknown"
 * Test source
 * ------------------------
 *  - unit/errorHandling/hipGetErrorName.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE(Unit_hipGetErrorName_Negative_Parameters) {
  const char* error_string = hipGetErrorName(static_cast<hipError_t>(-1));
  REQUIRE(error_string != nullptr);
#if HT_NVIDIA
  REQUIRE_THAT(error_string, Catch::Matchers::Equals("cudaErrorUnknown"));
#elif HT_AMD
  REQUIRE_THAT(error_string, Catch::Matchers::Equals("hipErrorUnknown"));
#endif
}

/**
 * End doxygen group ErrorTest.
 * @}
 */
