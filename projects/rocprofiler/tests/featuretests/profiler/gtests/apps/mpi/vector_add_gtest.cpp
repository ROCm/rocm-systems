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
#include <string>
#include <vector>

#include "gtests/apps/profiler_gtest.h"
#include "utils/test_utils.h"

constexpr auto kGoldenOutputMpi = "mpi_vectoradd_golden_traces.txt";

class MPITest : public ProfilerTest {
 protected:
  void ProcessMPIApplication(const char *app_name);
  void ExecuteAndParseApplication(std::stringstream &ss);

  void SetUp() {
    /*To supress No protocol found prints*/
    setenv("HWLOC_COMPONENTS", "-gl", 1);

    // run as standalone test
    ProfilerTest::SetUp("mpi_vectoradd");

    // run mpirun script
    // ProcessMPIApplication("mpi_run.sh");
  }
};

void MPITest::ProcessMPIApplication(const char *app_name) {
  std::string app_path =
      GetRunningPath("tests/featuretests/profiler/runFeatureTests");
  std::string lib_path = app_path;

  std::stringstream hsa_tools_lib_path;

  hsa_tools_lib_path << app_path << "librocprofiler_tool.so";
  setenv("LD_PRELOAD", hsa_tools_lib_path.str().c_str(), true);

  std::stringstream os;
  os << app_path << "tests/featuretests/profiler/gtests/apps/" << app_name;
  ExecuteAndParseApplication(os);
}

void MPITest::ExecuteAndParseApplication(std::stringstream &ss) {
  FILE *handle = popen(ss.str().c_str(), "r");
  ASSERT_NE(handle, nullptr);
  char *ln{NULL};
  std::string temp{""};
  size_t len{0};

  while (getline(&ln, &len, handle) != -1) {
    temp = temp + std::string(ln);
  }

  free(ln);
  size_t pos{0};
  std::string delimiter{"\n"};
  while ((pos = temp.find(delimiter)) != std::string::npos) {
    output_lines.push_back(temp.substr(0, pos));
    temp.erase(0, pos + delimiter.length());
  }

  pclose(handle);
}

// Test:1 Compares total num of kernel-names in golden output against current
// profiler output
TEST_F(MPITest,
       WhenRunningProfilerWithAppThenKernelNumbersMatchWithGoldenOutput) {
  std::vector<KernelInfo> current_kernel_info;

  GetKernelInfoForRunningApplication(&current_kernel_info);
  ASSERT_TRUE(current_kernel_info.size());

  EXPECT_GT(current_kernel_info.size(), 0);
}
