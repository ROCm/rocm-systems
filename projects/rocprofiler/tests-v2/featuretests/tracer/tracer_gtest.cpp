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
#include <gtest/gtest.h>
#include <ostream>
#include <vector>
#include "tracer_gtest.h"
#include "src/utils/logger.h"

std::string running_path;
std::string lib_path;
std::string golden_trace_path;
std::string test_app_path;
std::string metrics_path;
std::string binary_path;
std::string profiler_api_lib_path = "";

static void init_test_path() {
  lib_path = "lib/rocprofiler/librocprofiler_tool.so";
  metrics_path = "libexec/rocprofiler/counters/derived_counters.xml";
  profiler_api_lib_path = "/lib";
  if (is_installed_path()) {
    INFO_LOGGING("operating from /opt/rocm");
    running_path = "share/rocprofiler/tests/runTracerFeatureTests";
    golden_trace_path = "share/rocprofiler/tests/featuretests/tracer/apps/goldentraces/";
    test_app_path = "share/rocprofiler/tests/featuretests/tracer/apps/";
    binary_path = "bin/rocprofv2";
  } else {
    INFO_LOGGING("operating from ./build");
    running_path = "tests-v2/featuretests/tracer/runTracerFeatureTests";
    golden_trace_path = "tests-v2/featuretests/tracer/apps/goldentraces/";
    test_app_path = "tests-v2/featuretests/tracer/apps/";
    binary_path = "rocprofv2";
  }
}


/**
 * Sets application enviornment by seting HSA_TOOLS_LIB.
 */
void ApplicationParser::SetApplicationEnv(const char* app_name, const char* trace_option) {
  init_test_path();
  std::string app_path = GetRunningPath(running_path);

  std::stringstream ld_library_path;
  ld_library_path << app_path << profiler_api_lib_path << []() {
    const char* path = getenv("LD_LIBRARY_PATH");
    if (path != nullptr) return ":" + std::string(path);
    return std::string();
  }();
  setenv("LD_LIBRARY_PATH", ld_library_path.str().c_str(), true);

  std::stringstream hsa_tools_lib_path;
  auto _existing_ld_preload = getenv("LD_PRELOAD");
  if (_existing_ld_preload && strnlen(_existing_ld_preload, 1) > 0)
    hsa_tools_lib_path << _existing_ld_preload << ":";

  hsa_tools_lib_path << app_path << lib_path;
  setenv("LD_PRELOAD", hsa_tools_lib_path.str().c_str(), true);

  std::string trace_type{trace_option};

  if (trace_type.find("hip") != std::string::npos) {
    // set --hip-api option
    unsetenv("ROCPROFILER_HSA_API_TRACE");
    unsetenv("ROCPROFILER_HSA_ACTIVITY_TRACE");
    unsetenv("ROCPROFILER_ROCTX_TRACE");
    setenv("ROCPROFILER_HIP_API_TRACE", "1", true);
  }

  if (trace_type.find("hsa") != std::string::npos) {
    // set --hsa-api and --hsa-activity
    unsetenv("ROCPROFILER_HIP_API_TRACE");
    unsetenv("ROCPROFILER_ROCTX_TRACE");
    setenv("ROCPROFILER_HSA_API_TRACE", "1", true);
    setenv("ROCPROFILER_HSA_ACTIVITY_TRACE", "1", true);
  }

  if (trace_type.find("roctx") != std::string::npos) {
    // set --roctx-trace option
    unsetenv("ROCPROFILER_HIP_API_TRACE");
    unsetenv("ROCPROFILER_HSA_API_TRACE");
    unsetenv("ROCPROFILER_HSA_ACTIVITY_TRACE");
    setenv("ROCPROFILER_ROCTX_TRACE", "1", true);
  }


  std::stringstream os;
  os << app_path << test_app_path << app_name;
  ProcessApplication(os);
}

/**
 * Parses kernel-info after running tracer against curent application
 * and saves them in a vector.
 */
void ApplicationParser::GetKernelInfoForRunningApplication(
    std::vector<tracer_kernel_info_t>* kernel_info_output) {
  tracer_kernel_info_t kinfo;
  for (std::string line : output_lines) {
    // Skip all the lines until  "_DOMAIN" is found
    if (line.empty() || line.find("_DOMAIN") == std::string::npos) {
      continue;  // Skip to the next line if "Dispatch_ID" is found
    }

    // Parse individual values and store them in the dispatch struct
    tokenize_tracer_output(line, kinfo);

    if (kinfo.domain != "") {
      kernel_info_output->push_back(kinfo);
    }
  }
}

/**
 * Parses kernel-names from a pre-saved golden out files
 * and saves them in a vector.
 */
void ApplicationParser::GetKernelInfoForGoldenOutput(
    const char* app_name, std::string file_name,
    std::vector<tracer_kernel_info_t>* kernel_info_output) {
  std::string entry;
  std::string path = GetRunningPath(running_path);
  entry = path.append(golden_trace_path) + file_name;
  // parse kernel info fields for golden output
  ParseKernelInfoFields(entry, kernel_info_output);
}

/**
 * Runs a given appllication and saves tracer output.
 * These output lines can be letter passed for kernel informations
 * i.e: kernel_names
 */
void ApplicationParser::ProcessApplication(std::stringstream& ss) {
  FILE* handle = popen(ss.str().c_str(), "r");
  ASSERT_NE(handle, nullptr);

  char* ln{NULL};
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

/**
 * Parses kernel-info for golden output file
 * and saves them in a vector.
 */
void ApplicationParser::ParseKernelInfoFields(
    const std::string& s, std::vector<tracer_kernel_info_t>* kernel_info_output) {
  std::string line;
  tracer_kernel_info_t kinfo;

  std::ifstream golden_file(s);
  while (!golden_file.eof()) {
    getline(golden_file, line);
    // Skip all the lines until  "_DOMAIN" is found
    if (line.empty() || line.find("_DOMAIN") == std::string::npos) {
      continue;  // Skip to the next line if "Dispatch_ID" is found
    }

    // Parse individual values and store them in the dispatch struct
    tokenize_tracer_output(line, kinfo);

    if (kinfo.domain != "") {
      kernel_info_output->push_back(kinfo);
    }
  }
  golden_file.close();
}
/*
 * ###################################################
 * ############ HelloWorld HIP Tests ################
 * ###################################################
 */

void __attribute__((constructor)) globalsetting() {
  init_test_path();
  std::string app_path = GetRunningPath(running_path);
  std::stringstream gfx_path;
  gfx_path << app_path << metrics_path;
  setenv("ROCPROFILER_METRICS_PATH", gfx_path.str().c_str(), true);
}

constexpr auto kGoldenOutputHelloworld = "hip_helloworld_golden_traces.txt";

class HelloWorldTest : public Tracertest {
 protected:
  std::vector<tracer_kernel_info_t> golden_kernel_info;
  void SetUp() {
    Tracertest::SetUp("tracer_hip_helloworld", "--hip-api ");
    GetKernelInfoForGoldenOutput("tracer_hip_helloworld", kGoldenOutputHelloworld,
                                 &golden_kernel_info);
  }
  void TearDown() { output_lines.clear(); }
};

// Test:1 Compares total num of kernel-names in golden output against current
// tracer output
TEST_F(HelloWorldTest, WhenRunningTracerWithAppThenKernelInfoMatchWithGoldenOutput) {
  // kernel info in current profler run
  std::vector<tracer_kernel_info_t> current_kernel_info;

  GetKernelInfoForRunningApplication(&current_kernel_info);
  ASSERT_TRUE(current_kernel_info.size());

  EXPECT_EQ(golden_kernel_info.size(), current_kernel_info.size());
}

// Test:2 Compares order of kernel-names in golden output against current
// tracer output
TEST_F(HelloWorldTest, WhenRunningTracerWithAppThenFunctionNamesMatchWithGoldenOutput) {
  // kernel info in current tracer run
  std::vector<tracer_kernel_info_t> current_kernel_info;
  GetKernelInfoForRunningApplication(&current_kernel_info);

  ASSERT_TRUE(current_kernel_info.size());

  int version_position = current_kernel_info[0].function.find('R');
  if(version_position != std::string::npos) {
    current_kernel_info[0].function = current_kernel_info[0].function.substr(0, version_position) + ')';
  }

  version_position = current_kernel_info[1].function.find('R');
  if(version_position != std::string::npos) {
    current_kernel_info[1].function = current_kernel_info[1].function.substr(0, version_position) + ')';
  }

  EXPECT_EQ(golden_kernel_info[0].function, current_kernel_info[0].function);
  EXPECT_EQ(golden_kernel_info[1].function, current_kernel_info[1].function);
}

// Test:3 Compares order of kernel-names in golden output against current
// tracer output
TEST_F(HelloWorldTest, WhenRunningTracerWithAppThenKernelDurationShouldBePositive) {
  // kernel info in current tracer run
  std::vector<tracer_kernel_info_t> current_kernel_info;

  GetKernelInfoForRunningApplication(&current_kernel_info);
  ASSERT_TRUE(current_kernel_info.size());

  EXPECT_GT(current_kernel_info.size(), 0);
}

// Test:4 Compares end-time is greater than start-time in current
// tracer output
TEST_F(HelloWorldTest, WhenRunningTracerWithAppThenEndTimeIsGreaterThenStartTime) {
  // kernel info in current profiler run
  std::vector<tracer_kernel_info_t> current_kernel_info;

  GetKernelInfoForRunningApplication(&current_kernel_info);
  ASSERT_TRUE(current_kernel_info.size());

  for (auto& itr : current_kernel_info) {
    if (!(itr.begin_time).empty() && !(itr.end_time).empty()) {
      EXPECT_GT(get_timestamp_value(itr.end_time), get_timestamp_value(itr.begin_time));
    }
  }
}


/*
 * ###################################################
 * ############ Async Copy HSA Tests ################
 * ###################################################
 */

class AsyncCopyTest : public Tracertest {
 protected:
  void SetUp() { Tracertest::SetUp("copy_on_engine", "--hsa-api --hsa-activity"); }
  void TearDown() { output_lines.clear(); }
};

// Test:1 Compares total num of kernel-names in golden output against current
// tracer output
// This test is disabled because it is not compatible with Navi architecture.
TEST_F(AsyncCopyTest, DISABLED_WhenRunningTracerWithAppThenAsyncCopyOutputIsGenerated) {
  // kernel info in current profler run
  std::vector<tracer_kernel_info_t> current_kernel_info;

  GetKernelInfoForRunningApplication(&current_kernel_info);
  ASSERT_TRUE(current_kernel_info.size());
}

// Test:2 Matches coelation Ids
TEST_F(AsyncCopyTest, WhenRunningTracerWithAppThenAsyncCorelationCountIsCorrect) {
  // kernel info in current profler run
  std::vector<tracer_kernel_info_t> current_kernel_info;

  GetKernelInfoForRunningApplication(&current_kernel_info);
  ASSERT_TRUE(current_kernel_info.size());

  std::vector<std::string> corelation_pair{};
  for (const auto& itr : current_kernel_info) {
    if (itr.domain.find("HSA_OPS_DOMAIN") != std::string::npos) {
      corelation_pair.push_back(itr.corelation_id);
      break;  // we just want first occurance to test
    }
  }
  ASSERT_TRUE(corelation_pair.size());

  uint32_t corelation_count = 0;
  // check if same corelation id appears again but as a different ops record
  for (size_t i = 0; i < corelation_pair.size(); i++) {
    for (const auto& itr : current_kernel_info) {
      if ((itr.corelation_id == corelation_pair[i])) {
        corelation_count++;
      }
    }
  }

  // To remove the current record that we are checking with
  corelation_count--;

  EXPECT_EQ(corelation_count, corelation_pair.size());
}

/*
 * ###################################################
 * ######### Matrix Transpose ROCTX Tests ############
 * ###################################################
 */

constexpr auto GoldenOutputMatrixTranspose = "matrix_transpose_golden_trace.txt";

class ROCTXTest : public Tracertest {
 protected:
  void SetUp() { Tracertest::SetUp("tracer_matrix_transpose", "roctx"); }
  void TearDown() {}
};

TEST_F(ROCTXTest, WhenRunningTracerWithAppThenROCTxOutputIsgenerated) {
  std::string golden_file_path =
      GetRunningPath(running_path).append(golden_trace_path).append(GoldenOutputMatrixTranspose);
  std::ifstream golden_file(golden_file_path);
  std::string golden_output_line;
  std::vector<std::string> roctx_output;
  for (auto& itr : output_lines) {
    if (itr.find("ROCTX") != std::string::npos) roctx_output.push_back(itr);
  }
  uint32_t i = 0;
  while (!golden_file.eof()) {
    ASSERT_TRUE(i < roctx_output.size())
        << "Current Output number of records is less than golden output number of records"
        << std::endl;
    std::string roctx_output_line = roctx_output[i];
    getline(golden_file, golden_output_line);
    std::vector<std::string> golden_output_split;
    std::vector<std::string> roctx_output_split;
    std::string temp = "";
    for (int j = 0; j < (int)roctx_output_line.size(); j++) {
      if (roctx_output_line[j] != ',') {
        if (roctx_output_line[j] == ' ') continue;
        temp += roctx_output_line[j];
      } else {
        roctx_output_split.push_back(temp);
        temp = "";
      }
    }
    roctx_output_split.push_back(temp);
    temp = "";
    for (int j = 0; j < (int)golden_output_line.size(); j++) {
      if (golden_output_line[j] != ',') {
        if (golden_output_line[j] == ' ') continue;
        temp += golden_output_line[j];
      } else {
        golden_output_split.push_back(temp);
        temp = "";
      }
    }
    golden_output_split.push_back(temp);
    ASSERT_TRUE(roctx_output_split.size() == golden_output_split.size())
        << "Current Output number of records(" << roctx_output_split.size()
        << ") is not equal to golden output number of records (" << golden_output_split.size()
        << ")" << std::endl;
    uint32_t j = 0;
    for (auto& roctx_record : roctx_output_split) {
      ASSERT_TRUE(j < golden_output_split.size())
          << "ROCTX record not found: " << roctx_record << std::endl;
      std::string golden_output_record = golden_output_split[j++];
      if (roctx_record.find("Timestamp") == std::string::npos)
        EXPECT_EQ(roctx_record, golden_output_record) << "ROCTX record mismatch" << std::endl;
      else
        continue;
    }
    i++;
  }
  EXPECT_EQ(roctx_output.size(), i)
      << "Current Output number of records is greater than golden output number of records"
      << std::endl;
}
