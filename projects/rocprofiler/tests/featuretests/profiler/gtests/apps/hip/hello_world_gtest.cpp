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
#include <vector>

#include "gtests/apps/profiler_gtest.h"

constexpr auto kGoldenOutputHelloworld = "hip_helloworld_golden_traces.txt";

class HelloWorldTest : public ProfilerTest {
 protected:
  std::vector<KernelInfo> golden_kernel_info;
  void SetUp() {
    ProfilerTest::SetUp("hip_helloworld");
    GetKernelInfoForGoldenOutput("hip_helloworld", kGoldenOutputHelloworld, &golden_kernel_info);
  }
};

// Test:1 Compares total num of kernel-names in golden output against current
// profiler output
TEST_F(HelloWorldTest, WhenRunningProfilerWithAppThenKernelNumbersMatchWithGoldenOutput) {
  // kernel info in current profiler run
  std::vector<KernelInfo> current_kernel_info;

  GetKernelInfoForRunningApplication(&current_kernel_info);
  ASSERT_TRUE(current_kernel_info.size());

  EXPECT_EQ(golden_kernel_info.size(), current_kernel_info.size());
}

// Test:2 Compares order of kernel-names in golden output against current
// profiler output
TEST_F(HelloWorldTest, WhenRunningProfilerWithAppThenKernelNamessMatchWithGoldenOutput) {
  // kernel info in current profiler run
  std::vector<KernelInfo> current_kernel_info;
  GetKernelInfoForRunningApplication(&current_kernel_info);

  ASSERT_TRUE(current_kernel_info.size());

  EXPECT_EQ(golden_kernel_info[0].kernel_name, current_kernel_info[0].kernel_name);
  EXPECT_EQ(golden_kernel_info[1].kernel_name, current_kernel_info[1].kernel_name);
}

// Test:3 Compares order of kernel-names in golden output against current
// profiler output
TEST_F(HelloWorldTest, WhenRunningProfilerWithAppThenKernelDurationShouldBePositive) {
  // kernel info in current profiler run
  std::vector<KernelInfo> current_kernel_info;

  GetKernelInfoForRunningApplication(&current_kernel_info);
  ASSERT_TRUE(current_kernel_info.size());

  EXPECT_GT(current_kernel_info.size(), 0);
}

// Test:4 Compares end-time is greater than start-time in current
// profiler output
TEST_F(HelloWorldTest, WhenRunningProfilerWithAppThenEndTimeIsGreaterThenStartTime) {
  // kernel info in current profiler run
  std::vector<KernelInfo> current_kernel_info;

  GetKernelInfoForRunningApplication(&current_kernel_info);
  ASSERT_TRUE(current_kernel_info.size());

  for (auto& itr : current_kernel_info) {
    if (!(itr.start_time).empty() && !(itr.end_time).empty()) {
      EXPECT_GT(itr.end_time, itr.start_time);
    }
  }
}
