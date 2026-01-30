/*
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include <cassert>

/**
 * @addtogroup constexpr_assert constexpr_assert
 * @{
 * @ingroup DeviceLanguageTest
 * Tests for using assert() inside constexpr device functions.
 */

__device__ constexpr int safe_divide(int a, int b) {
  assert(b != 0);
  return a / b;
}

__device__ constexpr int result = safe_divide(10, 2);

/**
 * Test Description
 * ------------------------
 *  - Verifies that assert() can be used in constexpr __device__ functions.
 *  - Tests compile-time evaluation of constexpr device function with assert.
 *  - The assert condition is true, so it should not trigger.
 * Test source
 * ------------------------
 *  - unit/assertion/constexpr_assert.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.3
 */
TEST_CASE("Unit_ConstexprAssert_Positive_SafeDivide") {
  static_assert(result == 5, "safe_divide(10, 2) should equal 5");
  REQUIRE(result == 5);
}

/**
 * End doxygen group DeviceLanguageTest.
 * @}
 */

