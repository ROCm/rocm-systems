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

// RTC source for negative test - assert fails at compile time when b == 0
static constexpr auto kConstexprAssert_Negative{
    R"(
    #include <cassert>

    __device__ constexpr int safe_divide(int a, int b) {
      assert(b != 0);
      return a / b;
    }

    // This should fail to compile because assert(0 != 0) fails during
    // constant evaluation, causing __builtin_trap() to be called in constexpr context
    __device__ constexpr int bad_result = safe_divide(10, 0);
    )"};

/**
 * Test Description
 * ------------------------
 *  - Verifies that assert() in constexpr __device__ functions causes
 *    a compile-time error when the assertion fails.
 *  - Uses RTC to compile code where assert(b != 0) fails with b = 0.
 *  - Expects compilation to fail with an error.
 * Test source
 * ------------------------
 *  - unit/assertion/constexpr_assert.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.3
 */
TEST_CASE("Unit_ConstexprAssert_Negative_DivideByZero") {
  hiprtcProgram program{};

  HIPRTC_CHECK(hiprtcCreateProgram(&program, kConstexprAssert_Negative,
                                   "constexpr_assert_negative.cc", 0, nullptr, nullptr));
  hiprtcResult result = hiprtcCompileProgram(program, 0, nullptr);

  // Get the compile log
  size_t log_size{};
  HIPRTC_CHECK(hiprtcGetProgramLogSize(program, &log_size));
  std::string log(log_size, ' ');
  HIPRTC_CHECK(hiprtcGetProgramLog(program, log.data()));

  HIPRTC_CHECK(hiprtcDestroyProgram(&program));

  // Compilation should fail because assert fails during constant evaluation
  REQUIRE(result == HIPRTC_ERROR_COMPILATION);
}

// RTC source for testing a constexpr function that ALWAYS traps (like __assertfail)
// but returns a value instead of void
static constexpr auto kConstexprAlwaysTrap_Negative{
    R"(
    // Similar to __assertfail but returns int instead of void
    // This tests what happens when a constexpr function with __builtin_trap()
    // is called and assigned to a constexpr variable
    __device__ constexpr int always_fail() {
      __builtin_trap();
      return 0;  // Never reached, but needed for return type
    }

    // This should fail to compile because __builtin_trap() cannot be
    // evaluated at compile time
    __device__ constexpr int trap_result = always_fail();
    )"};

/**
 * Test Description
 * ------------------------
 *  - Verifies that a constexpr __device__ function that always calls
 *    __builtin_trap() causes a compile-time error when its result is
 *    assigned to a constexpr variable.
 *  - This directly tests the behavior of functions like __assertfail
 *    when marked constexpr and returning a value.
 *  - Uses RTC to verify compilation fails.
 * Test source
 * ------------------------
 *  - unit/assertion/constexpr_assert.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.3
 */
TEST_CASE("Unit_ConstexprAssert_Negative_AlwaysTrap") {
  hiprtcProgram program{};

  HIPRTC_CHECK(hiprtcCreateProgram(&program, kConstexprAlwaysTrap_Negative,
                                   "constexpr_always_trap.cc", 0, nullptr, nullptr));
  hiprtcResult result = hiprtcCompileProgram(program, 0, nullptr);

  // Get the compile log
  size_t log_size{};
  HIPRTC_CHECK(hiprtcGetProgramLogSize(program, &log_size));
  std::string log(log_size, ' ');
  HIPRTC_CHECK(hiprtcGetProgramLog(program, log.data()));

  HIPRTC_CHECK(hiprtcDestroyProgram(&program));

  // Compilation should fail because __builtin_trap() is not valid in constexpr context
  REQUIRE(result == HIPRTC_ERROR_COMPILATION);
}

// RTC source for testing __host__ __device__ constexpr void function with assert(false)
static constexpr auto kConstexprHostDeviceVoidAssert_Negative{
    R"(
    #include <cassert>

    // __host__ __device__ constexpr void function that always asserts false
    __host__ __device__ constexpr void my_assert() {
      assert(false);
    }

    // Wrapper function that calls my_assert and returns a value
    // This forces constexpr evaluation of my_assert
    __host__ __device__ constexpr int call_my_assert() {
      my_assert();
      return 42;
    }

    // This should fail to compile because assert(false) always fails
    // during constant evaluation
    constexpr int forced_result = call_my_assert();

    int main(int argc, char **argv) {
      return 0;
    }
    )"};

/**
 * Test Description
 * ------------------------
 *  - Verifies that a __host__ __device__ constexpr void function that
 *    always calls assert(false) causes a compile-time error when
 *    called in a constexpr context.
 *  - Tests the exact pattern: __host__ __device__ constexpr void my_assert() { assert(false); }
 *  - Uses a wrapper function to force constexpr evaluation of the void function.
 *  - Uses RTC to verify compilation fails.
 * Test source
 * ------------------------
 *  - unit/assertion/constexpr_assert.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.3
 */
TEST_CASE("Unit_ConstexprAssert_Negative_HostDeviceVoidAssert") {
  hiprtcProgram program{};

  HIPRTC_CHECK(hiprtcCreateProgram(&program, kConstexprHostDeviceVoidAssert_Negative,
                                   "constexpr_host_device_void_assert.cc", 0, nullptr, nullptr));
  hiprtcResult result = hiprtcCompileProgram(program, 0, nullptr);

  // Get the compile log
  size_t log_size{};
  HIPRTC_CHECK(hiprtcGetProgramLogSize(program, &log_size));
  std::string log(log_size, ' ');
  HIPRTC_CHECK(hiprtcGetProgramLog(program, log.data()));

  HIPRTC_CHECK(hiprtcDestroyProgram(&program));

  // Compilation should fail because assert(false) is not valid in constexpr context
  REQUIRE(result == HIPRTC_ERROR_COMPILATION);
}

/**
 * End doxygen group DeviceLanguageTest.
 * @}
 */

