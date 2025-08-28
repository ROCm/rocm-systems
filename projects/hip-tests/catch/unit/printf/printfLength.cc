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
 *    - Sanity test for `printf(format, ...)` to check all format specifier length sub-specifiers.
 *
 * Test source
 * ------------------------
 *    - unit/printf/printfLength.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Printf_length_Sanity_Positive") {
  int pcieAtomic = 0;
  HIP_CHECK(hipDeviceGetAttribute(&pcieAtomic, hipDeviceAttributeHostNativeAtomicSupported, 0));
  if (!pcieAtomic) {
    HipTest::HIP_SKIP_TEST("Device doesn't support pcie atomic, Skipped");
    return;
  }
#if HT_NVIDIA
  std::string reference(R"here(-42 -42
-42 -42
-42 -42
42 52
42 52
42 52
2a 2A
2a 2A
2a 2A
123.456000
x
)here");
#else
  std::string reference(R"here(-42 -42
-42 -42
-42 -42
42 52
42 52
42 52
2a 2A
2a 2A
2a 2A
123.456000
x
123.456000
-42 -42
-42 -42
-42 -42
0 0
42 52
42 52
42 52
0 0
)here");
#endif

  hip::SpawnProc proc("printfLength_exe", true);
  REQUIRE(0 == proc.run());
  REQUIRE(proc.getOutput() == reference);
}

/**
 * End doxygen group PrintfTest.
 * @}
 */
