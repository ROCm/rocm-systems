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
 * @addtogroup printf
 * @{
 * @ingroup PrintfTest
 * `int printf()` -
 * Method to print the content on output device.
 */
/**
 * Test Description
 * ------------------------
 * - Test case to verify the different format specifiers. Test case should compile with the compiler
 * option -mprintf-kind=buffered
 * - Fetch the printf content from a process which will verify format specifier. Compare it with
 * reference string. Test source
 * ------------------------
 * - catch/unit/printf/printfSpecifiersNonHost.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 5.7
 */

TEST_CASE("Unit_Buffered_Printf_Specifier") {
  int pcieAtomic = 0;
  HIP_CHECK(hipDeviceGetAttribute(&pcieAtomic, hipDeviceAttributeHostNativeAtomicSupported, 0));
  if (!pcieAtomic) {
    HipTest::HIP_SKIP_TEST("Device doesn't support pcie atomic, Skipped");
    return;
  }
#ifdef __HIP_PLATFORM_NVIDIA__
  std::string reference(R"here(xyzzy
%
hello % world
%s
%s0xf01dab1eca55e77e
%cxyzzy
sep
-42
42
123.456000
-123.456000
-1.234560e+02
1.234560E+02
123.456
-123.456
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
42
123.456000
-123.456000
-1.234560e+02
1.234560E+02
123.456
-123.456
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
42
123.456000
-123.456000
-1.234560e+02
1.234560E+02
123.456
-123.456
x

0000000000000000
3.14159000    hello F01DAB1ECA55E77E
)here");
#endif

  hip::SpawnProc proc("printfSpecifiersNonHost_exe", true);
  REQUIRE(0 == proc.run());
  REQUIRE(proc.getOutput() == reference);
}

/**
 * End doxygen group PrintfTest.
 * @}
 */
