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

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "printf_negative_kernels_rtc.hh"

#include <hip_test_common.hh>
#include <hip_test_process.hh>

/**
 * @addtogroup printf printf
 * @{
 * @ingroup PrintfTest
 * `int printf()` -
 * Method to print the content on output device.
 */

/**
 * Test Description
 * ------------------------
 *    - Sanity test for `printf(format, ...)` to check all format specifier specifier characters and
 * precision/width sub-specifiers.
 *
 * Test source
 * ------------------------
 *    - unit/printf/printfSpecifier.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Printf_specifier_Sanity_Positive") {
  int pcieAtomic = 0;
  HIP_CHECK(hipDeviceGetAttribute(&pcieAtomic, hipDeviceAttributeHostNativeAtomicSupported, 0));
  if (!pcieAtomic) {
    HipTest::HIP_SKIP_TEST("Device doesn't support pcie atomic, Skipped");
    return;
  }
#if HT_NVIDIA
  std::string reference(R"here(xyzzy
%
hello % world
%s
%s0xf01dab1eca55e77e
%cxyzzy
sep
-42
-42
42
52
2a
2A
123.456000
-123.456000
-1.234560e+02
1.234560E+02
123.456
-123.456
0x1.edd2f1a9fbe77p+6
-0X1.EDD2F1A9FBE77P+6
x
(null)
(nil)
3.14159000    hello 0xf01dab1eca55e77e
)here");
#elif !defined(_WIN32)
  std::string reference(R"here(xyzzy
%
hello % world
%s
%s0xf01dab1eca55e77e
%cxyzzy
sep
-42
-42
42
52
2a
2A
123.456000
-123.456000
-1.234560e+02
1.234560E+02
123.456
-123.456
0x1.edd2f1a9fbe77p+6
-0X1.EDD2F1A9FBE77P+6
x

(nil)
3.14159000    hello 0xf01dab1eca55e77e
)here");
#else
  std::string reference(R"here(xyzzy
%
hello % world
%s
%sF01DAB1ECA55E77E
%cxyzzy
sep
-42
-42
42
52
2a
2A
123.456000
-123.456000
-1.234560e+02
1.234560E+02
123.456
-123.456
0x1.edd2f1a9fbe77p+6
-0X1.EDD2F1A9FBE77P+6
x

0000000000000000
3.14159000    hello F01DAB1ECA55E77E
)here");
#endif

  hip::SpawnProc proc("printfSpecifiers_exe", true);
  REQUIRE(0 == proc.run());
  REQUIRE(proc.getOutput() == reference);
}

/**
 * Test Description
 * ------------------------
 *    - RTCs kernels that pass combinations of arguments of invalid types for printf
 * Test source
 * ------------------------
 *    - unit/printf/printfSpecifiers.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Printf_Negative_Parameters_RTC") {
  int pcieAtomic = 0;
  HIP_CHECK(hipDeviceGetAttribute(&pcieAtomic, hipDeviceAttributeHostNativeAtomicSupported, 0));
  if (!pcieAtomic) {
    HipTest::HIP_SKIP_TEST("Device doesn't support pcie atomic, Skipped");
    return;
  }
  hiprtcProgram program{};

  const auto program_source = kPrintfParam;

  HIPRTC_CHECK(
      hiprtcCreateProgram(&program, program_source, "printf_negative.cc", 0, nullptr, nullptr));
  hiprtcResult result{hiprtcCompileProgram(program, 0, nullptr)};

  // Get the compile log and count compiler error messages
  size_t log_size{};
  HIPRTC_CHECK(hiprtcGetProgramLogSize(program, &log_size));
  std::string log(log_size, ' ');
  HIPRTC_CHECK(hiprtcGetProgramLog(program, log.data()));
  int error_count{0};

  int expected_error_count{11};
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
 * End doxygen group PrintfTest.
 * @}
 */
