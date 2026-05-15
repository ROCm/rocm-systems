/*
Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INNCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANNY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/**
 * @addtogroup hipDeviceGetLimit hipDeviceGetLimit
 * @{
 * @ingroup DeviceTest
 * `hipDeviceGetLimit(size_t* pValue, enum hipLimit_t limit)` -
 * Get Resource limits of current device.
 */

#include <hip_test_common.hh>

/**
 * Test Description
 * ------------------------
 *  - Validates handling of invalid arguments:
 *    -# When output pointer to the limit value is `nullptr`
 *      - Expected output: do not return `hipSuccess`
 *    -# When limit enum is out of bounds
 *      - Expected output: do not return `hipSuccess`
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceGetLimit.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipDeviceGetLimit_NegTst") {
  size_t Value = 0;

  SECTION("NULL check") {
    REQUIRE_FALSE(hipDeviceGetLimit(nullptr, hipLimitMallocHeapSize) == hipSuccess);
  }

  SECTION("Invalid Input Flag") {
    REQUIRE_FALSE(hipDeviceGetLimit(&Value, static_cast<hipLimit_t>(0xff)) == hipSuccess);
  }
}

/**
 * Test Description
 * ------------------------
 *  - Validate that returned limit value for Malloc Heap size is valid.
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceGetLimit.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipDeviceGetLimit_CheckValidityOfOutputVal") {
  size_t Value = 0;

  REQUIRE(hipDeviceGetLimit(&Value, hipLimitMallocHeapSize) == hipSuccess);
  REQUIRE_FALSE(Value <= 0);
}

/**
 * End doxygen group DeviceTest.
 * @}
 */
