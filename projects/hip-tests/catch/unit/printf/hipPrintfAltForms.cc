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

__global__ void test_kernel() {
  printf("%#o\n", 042);
  printf("%#x\n", 0x42);
  printf("%#X\n", 0x42);
  printf("%#08x\n", 0x42);
  printf("%#f\n", -123.456);
  printf("%#F\n", 123.456);
  printf("%#e\n", 123.456);
  printf("%#E\n", -123.456);
  printf("%#g\n", -123.456);
  printf("%#G\n", 123.456);
  printf("%#a\n", 123.456);
  printf("%#A\n", -123.456);
  printf("%#.8x\n", 0x42);
  printf("%#16.8x\n", 0x42);
  printf("%-#16.8x\n", 0x42);
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
 * - Test case to verify alternate forms of printf API.
 * Test source
 * ------------------------
 * - catch/unit/printf/hipPrintfAltForms.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_Printf_PrintfAltFormsTsts") {
  int pcieAtomic = 0;
  HIP_CHECK(hipDeviceGetAttribute(&pcieAtomic, hipDeviceAttributeHostNativeAtomicSupported, 0));
  if (!pcieAtomic) {
    HipTest::HIP_SKIP_TEST("Device doesn't support pcie atomic, Skipped");
    return;
  }
  std::string reference(R"here(042
0x42
0X42
0x000042
-123.456000
123.456000
1.234560e+02
-1.234560E+02
-123.456
123.456
0x1.edd2f1a9fbe77p+6
-0X1.EDD2F1A9FBE77P+6
0x00000042
      0x00000042
0x00000042      
)here");
  CaptureStream captured(stdout);
  hipLaunchKernelGGL(test_kernel, dim3(1), dim3(1), 0, 0);
  HIP_CHECK(hipStreamSynchronize(0));
  auto CapturedData = captured.getCapturedData();
  std::string device_output = captured.gulp(CapturedData);
  REQUIRE(device_output == reference);
}
/**
 * End doxygen group PrintfTest.
 * @}
 */
