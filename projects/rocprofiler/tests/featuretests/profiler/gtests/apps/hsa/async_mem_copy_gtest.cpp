/*
Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.

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
#include <stdio.h>

#include <string>
#include <vector>

#include "gtests/apps/profiler_gtest.h"

constexpr auto kGOldenOutputAsyncCopy = "hsa_async_mem_copy_golden_traces.txt";

class HSATest : public ProfilerTest {
 protected:
  std::vector<KernelInfo> golden_kernel_info;
  void SetUp() {
    ProfilerTest::SetUp("hsa_async_mem_copy");
    GetKernelInfoForGoldenOutput("hsa_async_mem_copy", kGOldenOutputAsyncCopy,
                                 &golden_kernel_info);
  }
};

// Test:1 Given profiler don't intercept any hsa calls in this app
// we dont collect any counters by default. Expectation is, both vectors are
// empty
TEST_F(HSATest,
       WhenRunningProfilerWithAppThenKernelNumbersMatchWithGoldenOutput) {
  std::vector<KernelInfo> current_kernel_info;

  GetKernelInfoForRunningApplication(&current_kernel_info);

  EXPECT_EQ(current_kernel_info.size(), 0);
  EXPECT_EQ(golden_kernel_info.size(), 0);

  EXPECT_EQ(golden_kernel_info.size(), current_kernel_info.size());
}
