/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include "static_assert_kernels_rtc.hh"

/**
 * @addtogroup static_assert static_assert
 * @{
 * @ingroup DeviceLanguageTest
 * `void static_assert(constexpr expression, const char* message)` -
 * Stops the compilation if expression is equal to zero, and displays the specified message.
 */

void StaticAssertWrapper(const char* program_source) {
  hiprtcProgram program{};

  HIPRTC_CHECK(
      hiprtcCreateProgram(&program, program_source, "static_assert_rtc.cc", 0, nullptr, nullptr));
  hiprtcResult result{hiprtcCompileProgram(program, 0, nullptr)};

  // Get the compile log and count compiler error messages
  size_t log_size{};
  HIPRTC_CHECK(hiprtcGetProgramLogSize(program, &log_size));
  std::string log(log_size, ' ');
  HIPRTC_CHECK(hiprtcGetProgramLog(program, log.data()));
  int error_count{0};

  int expected_error_count{2};
  std::string error_message{"error:"};

  size_t n_pos = log.find(error_message, 0);
  while (n_pos != std::string::npos) {
    ++error_count;
    n_pos = log.find(error_message, n_pos + 1);
  }

  HIPRTC_CHECK(hiprtcDestroyProgram(&program));
  HIPRTC_CHECK_ERROR(result, HIPRTC_ERROR_COMPILATION);
  REQUIRE(error_count == expected_error_count);
}

/**
 * Test Description
 * ------------------------
 *  - Compiles kernels with static_assert calls:
 *    -# Expected that static_assert passes and compilation is successful.
 *    -# Expected that static_assert fails and compilation has errors.
 *  - Uses RTC to perform compilation.
 * Test source
 * ------------------------
 *  - unit/assertion/static_assert.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_StaticAssert_Positive_Basic_RTC) { StaticAssertWrapper(kStaticAssert_Positive); }

/**
 * Test Description
 * ------------------------
 *  - Passes invalidly formed expressions to static_assert calls.
 *  - Uses expressions that are not constexpr and values that are not known during compilation.
 *  - Uses RTC to perform compilation.
 * Test source
 * ------------------------
 *  - unit/assertion/static_assert.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_StaticAssert_Negative_Basic_RTC) { StaticAssertWrapper(kStaticAssert_Negative); }

/**
 * End doxygen group DeviceLanguageTest.
 * @}
 */
