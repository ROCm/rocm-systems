/*
Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include <hip_test_defgroups.hh>
#include "printf_common.h"  // NOLINT

__global__ void test_kernel_star() {
  printf("%*d\n", 16, 42);
  printf("%.*d\n", 8, 42);
  printf("%*.*d\n", -16, 8, 42);
  printf("%*.*f %s * %.*s\n", 16, 8, 123.456, "hello", 5, "worldxyz");
}
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
 * - Test case to verify the additional arguments (*) in the printf API
 * Test source
 * ------------------------
 * - catch/unit/printf/hipPrintfStar.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_Printf_PrintfStar") {
  int pcieAtomic = 0;
  HIP_CHECK(hipDeviceGetAttribute(&pcieAtomic, hipDeviceAttributeHostNativeAtomicSupported, 0));
  if (!pcieAtomic) {
    HipTest::HIP_SKIP_TEST("Device doesn't support pcie atomic, Skipped");
    return;
  }
  std::string reference(R"here(              42
00000042
00000042        
    123.45600000 hello * world
)here");
  CaptureStream captured(stdout);
  hipLaunchKernelGGL(test_kernel_star, dim3(1), dim3(1), 0, 0);
  HIP_CHECK(hipStreamSynchronize(0));
  auto CapturedData = captured.getCapturedData();
  std::string device_output = captured.gulp(CapturedData);
  REQUIRE(device_output == reference);
}

/**
 * End doxygen group PrintfTest.
 * @}
 */
