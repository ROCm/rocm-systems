/*
Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.

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
TEST_CASE("Unit_hipGetErrorName_Positive_Basic") {
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
TEST_CASE("Unit_hipGetErrorName_Negative_Parameters") {
  const char* error_string = hipGetErrorName(static_cast<hipError_t>(-1));
  REQUIRE(error_string != nullptr);
#if HT_NVIDIA
  REQUIRE_THAT(error_string, Catch::Equals("cudaErrorUnknown"));
#elif HT_AMD
  REQUIRE_THAT(error_string, Catch::Equals("hipErrorUnknown"));
#endif
}

/**
 * End doxygen group ErrorTest.
 * @}
 */
