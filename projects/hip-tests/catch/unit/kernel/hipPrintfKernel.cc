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
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#include <hip_test_common.hh>

#include <cstring>
#include "../kernel/printf_common.h"

#define HIP_ENABLE_PRINTF

__global__ void run_printf() { printf("Hello World\n"); }
/**
* @addtogroup hipLaunchKernelGGL
* @{
* @ingroup KernelTest
* `void hipLaunchKernelGGL(F kernel, const dim3& numBlocks, const dim3& dimBlocks,
   std::uint32_t sharedMemBytes, hipStream_t stream, Args... args)` -
* Method to invocate kernel functions
*/

/**
 * Test Description
 * ------------------------
 * - Test case to check printf function via kernel call.

 * Test source
 * ------------------------
 * - catch/unit/kernel/hipPrintfKernel.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 5.6
 */
TEST_CASE("Unit_kernel_ChkPrintf") {
  int device_count = 0;
  CaptureStream capture(stdout);
  HIP_CHECK(hipGetDeviceCount(&device_count));
  std::string st = "Hello World";
  const char* check = st.c_str();
  for (int i = 0; i < device_count; ++i) {
    HIP_CHECK(hipSetDevice(i));
    if (!HipTest::isPcieAtomicSupported()) continue;
    hipLaunchKernelGGL(run_printf, dim3(1), dim3(1), 0, 0);
    HIP_CHECK(hipDeviceSynchronize());
    char* data = new char[st.size()];
    ;
    std::ifstream CapturedData = capture.getCapturedData();
    CapturedData.getline(data, st.size() + 1);
    int result = strcmp(data, check);
    REQUIRE(result == 0);
    delete[] data;
  }
}

/**
 * End doxygen group KernelTest.
 * @}
 */
