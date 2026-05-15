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
TEST_CASE("Unit_hipDrvGetErrorName_Positive_Basic") {
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
TEST_CASE("Unit_hipDrvGetErrorName_Negative_Parameters") {
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
