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


// Kernel Function
__global__ void run_printf(int* count) { *count = printf("Hello World"); }
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
 * - Test case to verify the printf return value(Number of Characters)for -mprintf-kind=hostcall
 * compiler option Test source
 * ------------------------
 * - catch/unit/printf/printfHost.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 5.7
 */
TEST_CASE("Unit_Host_Printf") {
  int pcieAtomic = 0;
  HIP_CHECK(hipDeviceGetAttribute(&pcieAtomic, hipDeviceAttributeHostNativeAtomicSupported, 0));
  if (!pcieAtomic) {
    HipTest::HIP_SKIP_TEST("Device doesn't support pcie atomic, Skipped");
    return;
  }
  int *count{nullptr}, *count_d{nullptr};
  count = reinterpret_cast<int*>(malloc(sizeof(int)));
  HIP_CHECK(hipMalloc(&count_d, sizeof(int)));

  hipLaunchKernelGGL(run_printf, dim3(1), dim3(1), 0, 0, count_d);

  HIP_CHECK(hipMemcpy(count, count_d, sizeof(int), hipMemcpyDeviceToHost));

  std::string str = "Hello World";
  int length = str.length();
#if HT_AMD
  REQUIRE(length == *count);
#else
  REQUIRE(*count == 0);
#endif
  free(count);
  HIP_CHECK(hipFree(count_d));
}

/**
 * End doxygen group PrintfTest.
 * @}
 */
