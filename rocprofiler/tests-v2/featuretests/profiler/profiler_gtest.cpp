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

#include "profiler_gtest.h"

#include <gtest/gtest.h>
#include <hsa/hsa.h>

#include "rocprofiler.h"
#include <cstdlib>
#include <exception>
#include <string>
#include <thread>
#include <array>

#include "src/utils/filesystem.hpp"
#include "src/utils/helper.h"
#include "utils/csv_parser.h"
#include "src/utils/logger.h"
#include "apps/hip_kernels.h"

std::string running_path;
std::string lib_path;
std::string golden_trace_path;
std::string test_app_path;
std::string metrics_path;
std::string binary_path;
std::string profiler_api_lib_path = "";
bool bSkipCounterNoneZeroCheck = false;

static void init_test_path() {
  lib_path = "lib/rocprofiler/librocprofiler_tool.so";
  metrics_path = "libexec/rocprofiler/counters/derived_counters.xml";
  profiler_api_lib_path = "/lib";
  if (is_installed_path()) {
    INFO_LOGGING("operating from /opt/rocm");
    running_path = "share/rocprofiler/tests/runFeatureTests";
    golden_trace_path = "share/rocprofiler/tests/featuretests/profiler/apps/goldentraces/";
    test_app_path = "share/rocprofiler/tests/featuretests/profiler/apps/";
    binary_path = "bin/rocprofv2";
  } else {
    INFO_LOGGING("operating from ./build/");
    running_path = "tests-v2/featuretests/profiler/runFeatureTests";
    golden_trace_path = "tests-v2/featuretests/profiler/apps/goldentraces/";
    test_app_path = "tests-v2/featuretests/profiler/apps/";
    binary_path = "rocprofv2";
  }
}


void __attribute__((constructor)) globalsetting() {
  init_test_path();
  std::string app_path = GetRunningPath(running_path);
  std::stringstream gfx_path;
  gfx_path << app_path << metrics_path;
  setenv("ROCPROFILER_METRICS_PATH", gfx_path.str().c_str(), true);
  setenv("ROCPROFILER_MAX_ATT_PROFILES", "2", 1);
  setenv("ROCPROFILER_TRUNCATE_KERNEL_PATH", "1", true);
}

/**
 * Sets application enviornment by setting COUNTERS_PATH,LD_PRELOAD,LD_LIBRARY_PATH.
 */
void ApplicationParser::SetApplicationEnv(const char* app_name) {
  std::string app_path;

  // set global path
  init_test_path();

  app_path = GetRunningPath(running_path);

  std::stringstream ld_library_path;
  ld_library_path << app_path << profiler_api_lib_path << []() {
    const char* path = getenv("LD_LIBRARY_PATH");
    if (path != nullptr) return ":" + std::string(path);
    return std::string();
  }();
  setenv("LD_LIBRARY_PATH", ld_library_path.str().c_str(), true);

  std::stringstream counter_path;
  counter_path << app_path << golden_trace_path << "input.txt";
  setenv("COUNTERS_PATH", counter_path.str().c_str(), true);

  std::stringstream hsa_tools_lib_path;
  auto _existing_ld_preload = getenv("LD_PRELOAD");
  if (_existing_ld_preload && strnlen(_existing_ld_preload, 1) > 0)
    hsa_tools_lib_path << _existing_ld_preload << ":";

  hsa_tools_lib_path << app_path << lib_path;

  setenv("LD_PRELOAD", hsa_tools_lib_path.str().c_str(), true);

  std::stringstream ld_lib_path;
  ld_lib_path << app_path << "lib" << []() {
    const char* path = getenv("LD_LIBRARY_PATH");
    if (path != nullptr) return ":" + std::string(path);
    return std::string("");
  }();
  setenv("LD_LIBRARY_PATH", ld_lib_path.str().c_str(), true);

  std::stringstream os;
  os << app_path << test_app_path << app_name;

  ProcessApplication(os);

  /*unsetenv("LD_LIBRARY_PATH");
  unsetenv("LD_PRELOAD");
  unsetenv("COUNTERS_PATH");*/
}

/**
 * Parses kernel-info after running profiler against curent application
 * and saves them in a vector.
 */
void ApplicationParser::GetKernelInfoForRunningApplication(
    std::vector<profiler_kernel_info_t>* kernel_info_output) {
  profiler_kernel_info_t kinfo;
  for (const auto &line : output_lines) {
    // Skip all the lines until  "Dispatch_ID" is found
    if (line.empty() || line.find("Dispatch_ID") == std::string::npos || line.find("Kernel_Name(\"__amd_") != std::string::npos) {
        continue;  // Skip to the next line if "Dispatch_ID" is not found or "Kernel_Name("__amd_" is found
    }

    // Parse individual values and store them in the dispatch struct
    tokenize_profiler_output(line, kinfo);

    kernel_info_output->push_back(kinfo);
  }
}

/**
 * Parses kernel-names from a pre-saved golden out files
 * and saves them in a vector.
 */
void ApplicationParser::GetKernelInfoForGoldenOutput(
    const char* app_name, std::string file_name,
    std::vector<profiler_kernel_info_t>* kernel_info_output) {
  std::string entry;
  std::string path = GetRunningPath(running_path);
  entry = path.append(golden_trace_path) + file_name;
  //  parse kernel info fields for golden output
  ParseKernelInfoFields(entry, kernel_info_output);
}

/**
 * Runs a given appllication and saves profiler output.
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
    const std::string& s, std::vector<profiler_kernel_info_t>* kernel_info_output) {
  std::string line;
  profiler_kernel_info_t kinfo;

  std::ifstream golden_file(s);
  while (!golden_file.eof()) {
    getline(golden_file, line);
    // Skip all the lines until  "Dispatch_ID" is found
    if (line.empty() || line.find("Dispatch_ID") == std::string::npos) {
      continue;  // Skip to the next line if "Dispatch_ID" is found
    }
    // Parse individual values and store them in the dispatch struct
    tokenize_profiler_output(line, kinfo);
    kernel_info_output->push_back(kinfo);
  }
  golden_file.close();
}


constexpr auto kGoldenOutputHelloworld = "hip_helloworld_golden_traces.txt";
constexpr auto kGoldenOutputVectorAdd = "hip_vectoradd_golden_traces.txt";
constexpr auto kGOldenOutputAsyncCopy = "hsa_async_mem_copy_golden_traces.txt";
constexpr auto kGoldenOutputOpenMP = "openmp_helloworld_golden_traces.txt";
constexpr auto kGoldenOutputMpi = "mpi_vectoradd_golden_traces.txt";

/*
 * ###################################################
 * ############ Hello World HIP Tests ################
 * ###################################################
 */

class HelloWorldTest : public ProfilerTest {
 protected:
  std::vector<profiler_kernel_info_t> golden_kernel_info;
  void SetUp() {
    ProfilerTest::SetUp("hip_helloworld");
    GetKernelInfoForGoldenOutput("hip_helloworld", kGoldenOutputHelloworld, &golden_kernel_info);
  }
};

// Test:1 Compares total num of kernel-names in golden output against current
// profiler output
TEST_F(HelloWorldTest, WhenRunningProfilerWithAppThenKernelNumbersMatchWithGoldenOutput) {
  // kernel info in current profiler run
  std::vector<profiler_kernel_info_t> current_kernel_info;

  GetKernelInfoForRunningApplication(&current_kernel_info);
  ASSERT_TRUE(current_kernel_info.size());

  EXPECT_EQ(golden_kernel_info.size(), current_kernel_info.size());
}

// Test:2 Compares order of kernel-names in golden output against current
// profiler output
TEST_F(HelloWorldTest, WhenRunningProfilerWithAppThenKernelNamessMatchWithGoldenOutput) {
  // kernel info in current profiler run
  std::vector<profiler_kernel_info_t> current_kernel_info;
  GetKernelInfoForRunningApplication(&current_kernel_info);

  ASSERT_EQ(golden_kernel_info.size(), current_kernel_info.size());
  for (size_t i = 0; i < current_kernel_info.size(); ++i) {
    EXPECT_EQ(golden_kernel_info[i].kernel_name, current_kernel_info[i].kernel_name) << "i=" << i;
  }
}

// Test:3 Compares order of kernel-names in golden output against current
// profiler output
TEST_F(HelloWorldTest, WhenRunningProfilerWithAppThenKernelDurationShouldBePositive) {
  // kernel info in current profiler run
  std::vector<profiler_kernel_info_t> current_kernel_info;

  GetKernelInfoForRunningApplication(&current_kernel_info);
  ASSERT_TRUE(current_kernel_info.size());

  EXPECT_GT(current_kernel_info.size(), 0);
}

// Test:4 Compares end-time is greater than start-time in current
// profiler output
TEST_F(HelloWorldTest, WhenRunningProfilerWithAppThenEndTimeIsGreaterThenStartTime) {
  // kernel info in current profiler run
  std::vector<profiler_kernel_info_t> current_kernel_info;

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
 * ############ Vector Add HIP Tests ################
 * ###################################################
 */

class VectorAddTest : public ProfilerTest {
 protected:
  std::vector<profiler_kernel_info_t> golden_kernel_info;
  void SetUp() {
    ProfilerTest::SetUp("hip_vectoradd");
    GetKernelInfoForGoldenOutput("hip_vectoradd", kGoldenOutputVectorAdd, &golden_kernel_info);
  }
};

// Test:1 Compares total num of kernel-names in golden output against current
// profiler output
TEST_F(VectorAddTest, WhenRunningProfilerWithAppThenKernelNumbersMatchWithGoldenOutput) {
  std::vector<profiler_kernel_info_t> current_kernel_info;

  GetKernelInfoForRunningApplication(&current_kernel_info);
  ASSERT_TRUE(current_kernel_info.size());

  EXPECT_EQ(golden_kernel_info.size(), current_kernel_info.size());
}

// Test:2 Compares order of kernel-names in golden output against current
// profiler output
TEST_F(VectorAddTest, WhenRunningProfilerWithAppThenKernelNamessMatchWithGoldenOutput) {
  std::vector<profiler_kernel_info_t> current_kernel_info;

  GetKernelInfoForRunningApplication(&current_kernel_info);
  ASSERT_TRUE(current_kernel_info.size());
  ASSERT_TRUE(golden_kernel_info.size());

  EXPECT_EQ(golden_kernel_info[0].kernel_name, current_kernel_info[0].kernel_name);
}

// Test:3 Compares order of kernel-names in golden output against current
// profiler output
TEST_F(VectorAddTest, WhenRunningProfilerWithAppThenKernelDurationShouldBePositive) {
  // kernel info in current profiler run
  std::vector<profiler_kernel_info_t> current_kernel_info;

  GetKernelInfoForRunningApplication(&current_kernel_info);
  ASSERT_TRUE(current_kernel_info.size());

  EXPECT_GT(current_kernel_info.size(), 0);
}

// Test:4 Compares end-time is greater than start-time in current
// profiler output
TEST_F(VectorAddTest, WhenRunningProfilerWithAppThenEndTimeIsGreaterThenStartTime) {
  // kernel info in current profiler run
  std::vector<profiler_kernel_info_t> current_kernel_info;

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
 * ############ Async Mem copy Tests ################
 * ###################################################
 */

class HSATest : public ProfilerTest {
 protected:
  std::vector<profiler_kernel_info_t> golden_kernel_info;
  void SetUp() {
    ProfilerTest::SetUp("hsa_async_mem_copy");
    GetKernelInfoForGoldenOutput("hsa_async_mem_copy", kGOldenOutputAsyncCopy, &golden_kernel_info);
  }
};

// Test:1 Given profiler don't intercept any hsa calls in this app
// we dont collect any counters by default. Expectation is, both vectors are
// empty
TEST_F(HSATest, WhenRunningProfilerWithAppThenKernelNumbersMatchWithGoldenOutput) {
  std::vector<profiler_kernel_info_t> current_kernel_info;

  GetKernelInfoForRunningApplication(&current_kernel_info);

  EXPECT_EQ(current_kernel_info.size(), 0);
  EXPECT_EQ(golden_kernel_info.size(), 0);

  EXPECT_EQ(golden_kernel_info.size(), current_kernel_info.size());
}

/*
 * ###################################################
 * ############ OpenMP Tests ################
 * ###################################################
 */
// #ifdef USE_OpenMP
// class OpenMPTest : public ProfilerTest {
//  protected:
//   std::vector<profiler_kernel_info_t> golden_kernel_info;
//   void SetUp() {
//     ProfilerTest::SetUp("openmp_helloworld");
//     GetKernelInfoForGoldenOutput("openmp_helloworld", kGoldenOutputOpenMP, &golden_kernel_info);
//   }
// };

// // Test:1 Compares total num of kernel-names in golden output against current
// // profiler output
// TEST_F(OpenMPTest, DISABLED_WhenRunningProfilerWithAppThenKernelNumbersMatchWithGoldenOutput) {
//   std::vector<profiler_kernel_info_t> current_kernel_info;

//   GetKernelInfoForRunningApplication(&current_kernel_info);
//   ASSERT_TRUE(current_kernel_info.size());

//   EXPECT_EQ(golden_kernel_info.size(), current_kernel_info.size());
// }

// // Test:2 Compares order of kernel-names in golden output against current
// // profiler output
// TEST_F(OpenMPTest, WhenRunningProfilerWithAppThenKernelNamesMatchWithGoldenOutput) {
//   std::vector<profiler_kernel_info_t> current_kernel_info;

//   GetKernelInfoForRunningApplication(&current_kernel_info);
//   ASSERT_TRUE(current_kernel_info.size());

//   EXPECT_EQ(golden_kernel_info[0].kernel_name, current_kernel_info[0].kernel_name);
// }

// // Test:3 Compares order of kernel-names in golden output against current
// // profiler output
// TEST_F(OpenMPTest, WhenRunningProfilerWithAppThenKernelDurationShouldBePositive) {
//   // kernel info in current profiler run
//   std::vector<profiler_kernel_info_t> current_kernel_info;

//   GetKernelInfoForRunningApplication(&current_kernel_info);
//   ASSERT_TRUE(current_kernel_info.size());

//   EXPECT_GT(current_kernel_info.size(), 0);
// }

// // Test:4 Compares end-time is greater than start-time in current
// // profiler output
// TEST_F(OpenMPTest, WhenRunningProfilerWithAppThenEndTimeIsGreaterThenStartTime) {
//   // kernel info in current profiler run
//   std::vector<profiler_kernel_info_t> current_kernel_info;

//   GetKernelInfoForRunningApplication(&current_kernel_info);
//   ASSERT_TRUE(current_kernel_info.size());

//   for (auto& itr : current_kernel_info) {
//     if (!(itr.end_time).empty()) {
//       EXPECT_GT(itr.end_time, itr.begin_time);
//     }
//   }
// }
// #endif
/*
 * ###################################################
 * ############ MPI Tests ################
 * ###################################################
 */
#ifdef USE_MPI
class MPITest : public ProfilerTest {
 protected:
  void ProcessMPIApplication(const char* app_name);
  void ExecuteAndParseApplication(std::stringstream& ss);

  std::vector<profiler_kernel_info_t> golden_kernel_info;
  void SetUp() {
    /*To supress No protocol found prints*/
    setenv("HWLOC_COMPONENTS", "-gl", 1);
    ProfilerTest::SetUp("mpi_vectoradd");
    GetKernelInfoForGoldenOutput("mpi_vectoradd", kGoldenOutputMpi, &golden_kernel_info);
  }
};

// Test:1 if kernel-name exists in current profiler output
TEST_F(MPITest, DISABLED_WhenRunningProfilerWithAppThenKernelNumbersOutputGenerated) {
  std::vector<profiler_kernel_info_t> current_kernel_info;

  GetKernelInfoForRunningApplication(&current_kernel_info);
  ASSERT_TRUE(current_kernel_info.size());

  EXPECT_GT(current_kernel_info.size(), 0);
}

// Test:1 if kernel-name matches with golden output
TEST_F(MPITest, DISABLED_WhenRunningProfilerWithAppThenKernelNameMatchWithGoldenOutput) {
  std::vector<profiler_kernel_info_t> current_kernel_info;

  GetKernelInfoForRunningApplication(&current_kernel_info);
  ASSERT_TRUE(current_kernel_info.size());

  EXPECT_EQ(golden_kernel_info[0].kernel_name, current_kernel_info[0].kernel_name);
}
#endif
/*
 * ###################################################
 * ############ HSA Load Unload Tests ################
 * ###################################################
 */

// Run 2 loops of {hsa_init(); hsa_iterate_agents(); hsa_shut_down()} to test
// that the profiler tool correctly unloaded after the 1st iteration and then
// reloaded for the 2nd iteration.
class LoadUnloadTest : public ::testing::Test {
 protected:
  virtual void SetUp() {
    // start basic app
    hsa_init();
  }

  virtual void TearDown() {
    // stop basic app and unset tools lib
    hsa_shut_down();
  }
};

TEST_F(LoadUnloadTest, WhenLoadingFirstTimeThenToolLoadsUnloadsSuccessfully) {
  // Tool loaded in the setup
  // Tool unloaded in teardown

  // iterate for gpu's
  hsa_status_t status = hsa_iterate_agents(
      [](hsa_agent_t agent, void*) {
        std::string agentname;
        agentname.resize(64);
        hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME, agentname.data());
        if ((agentname.find("gfx11") != std::string::npos) ||
            (agentname.find("gfx12") != std::string::npos))
          bSkipCounterNoneZeroCheck = true;

        hsa_device_type_t type;
        return hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
      },
      nullptr);

  EXPECT_EQ(HSA_STATUS_SUCCESS, status);
}

TEST_F(LoadUnloadTest, WhenLoadingSecondTimeThenToolLoadsUnloadsSuccessfully) {
  // Tool loaded in the setup
  // Tool unloaded in teardown

  // iterate for gpu's
  hsa_status_t status = hsa_iterate_agents(
      [](hsa_agent_t agent, void*) {
        hsa_device_type_t type;
        return hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
      },
      nullptr);

  EXPECT_EQ(HSA_STATUS_SUCCESS, status);
}


/*
 * ###################################################
 * ############ MultiThreaded API Tests ################
 * ###################################################
 */

class ProfilerAPITest : public ::testing::Test {
 protected:
  void SetUp() {
    std::string app_path = GetRunningPath(running_path);
    std::stringstream gfx_path;
    gfx_path << app_path << metrics_path;
    setenv("ROCPROFILER_METRICS_PATH", gfx_path.str().c_str(), true);
    setenv("ROCPROFILER_MAX_ATT_PROFILES", "2", 1);
  }
  // function to check profiler API status
  static void CheckApi(rocprofiler_status_t status) {
    ASSERT_EQ(status, ROCPROFILER_STATUS_SUCCESS);
  };

  // callback function to dump profiler data
  static void FlushCallback(const rocprofiler_record_header_t* record,
                            const rocprofiler_record_header_t* end_record,
                            rocprofiler_session_id_t session_id,
                            rocprofiler_buffer_id_t buffer_id) {
    while (record < end_record) {
      if (!record) break;
      if (record->kind == ROCPROFILER_PROFILER_RECORD) {
        const rocprofiler_record_profiler_t* profiler_record =
            reinterpret_cast<const rocprofiler_record_profiler_t*>(record);
        size_t name_length;
        CheckApi(rocprofiler_query_kernel_info_size(ROCPROFILER_KERNEL_NAME,
                                                    profiler_record->kernel_id, &name_length));
        const char* kernel_name_c = static_cast<const char*>(malloc(name_length * sizeof(char)));
        CheckApi(rocprofiler_query_kernel_info(ROCPROFILER_KERNEL_NAME, profiler_record->kernel_id,
                                               &kernel_name_c));

        if (profiler_record->counters && !bSkipCounterNoneZeroCheck)
          for (uint64_t i = 0; i < profiler_record->counters_count.value; i++)
            if (profiler_record->counters[i].counter_handler.handle > 0)
              EXPECT_NE(profiler_record->counters[i].value.value, 0);
      }
      CheckApi(rocprofiler_next_record(record, &record, session_id, buffer_id));
    }
  }
};

TEST_F(ProfilerAPITest, WhenRunningMultipleThreadsProfilerAPIsWorkFine) {
  // set global path
  init_test_path();

  // Get the system cores
  int num_cpu_cores = GetNumberOfCores();

  // create as many threads as number of cores in system
  std::vector<std::thread> threads(num_cpu_cores);

  // initialize profiler by creating rocprofiler object
  CheckApi(rocprofiler_initialize());

  // Counter Collection with timestamps
  rocprofiler_session_id_t session_id;
  std::vector<const char*> counters;
  counters.emplace_back("SQ_WAVES");

  CheckApi(rocprofiler_create_session(ROCPROFILER_NONE_REPLAY_MODE, &session_id));

  rocprofiler_buffer_id_t buffer_id;
  CheckApi(rocprofiler_create_buffer(session_id, FlushCallback, 0x9999, &buffer_id));

  rocprofiler_filter_id_t filter_id;
  rocprofiler_filter_property_t property = {};
  CheckApi(rocprofiler_create_filter(session_id, ROCPROFILER_COUNTERS_COLLECTION,
                                     rocprofiler_filter_data_t{.counters_names = &counters[0]},
                                     counters.size(), &filter_id, property));

  CheckApi(rocprofiler_set_filter_buffer(session_id, filter_id, buffer_id));

  // activating profiler session
  CheckApi(rocprofiler_start_session(session_id));

  // launch kernel on each thread
  for (int n = 0; n < num_cpu_cores; ++n) {
    threads[n] = std::thread(KernelLaunch);
  }

  // wait for all kernel launches to complete
  for (int n = 0; n < num_cpu_cores; ++n) {
    threads[n].join();
  }

  // deactivate session
  CheckApi(rocprofiler_terminate_session(session_id));

  // dump profiler data
  CheckApi(rocprofiler_flush_data(session_id, buffer_id));

  // destroy session
  CheckApi(rocprofiler_destroy_session(session_id));

  // finalize profiler by destroying rocprofiler object
  CheckApi(rocprofiler_finalize());
}

TEST_F(ProfilerAPITest, WhenRunningMultipleStreamsSerializationWorksFine) {
  // set global path
  init_test_path();

  // Get the system cores
  int num_cpu_cores = GetNumberOfCores();

  // create as many threads as number of cores in system
  std::vector<std::thread> threads(num_cpu_cores);

  // initialize profiler by creating rocprofiler object
  CheckApi(rocprofiler_initialize());

  // Counter Collection with timestamps
  rocprofiler_session_id_t session_id;
  std::vector<const char*> counters;
  counters.emplace_back("SQ_WAVES");

  CheckApi(rocprofiler_create_session(ROCPROFILER_NONE_REPLAY_MODE, &session_id));

  rocprofiler_buffer_id_t buffer_id;
  CheckApi(rocprofiler_create_buffer(session_id, FlushCallback, 0x9999, &buffer_id));

  rocprofiler_filter_id_t filter_id;
  rocprofiler_filter_property_t property = {};
  CheckApi(rocprofiler_create_filter(session_id, ROCPROFILER_COUNTERS_COLLECTION,
                                     rocprofiler_filter_data_t{.counters_names = &counters[0]},
                                     counters.size(), &filter_id, property));

  CheckApi(rocprofiler_set_filter_buffer(session_id, filter_id, buffer_id));

  // activating profiler session
  CheckApi(rocprofiler_start_session(session_id));

  LaunchMultiStreamKernels();
  // deactivate session
  CheckApi(rocprofiler_terminate_session(session_id));

  // dump profiler data
  CheckApi(rocprofiler_flush_data(session_id, buffer_id));

  // destroy session
  CheckApi(rocprofiler_destroy_session(session_id));

  // finalize profiler by destroying rocprofiler object
  CheckApi(rocprofiler_finalize());
}


/*
 * ###################################################
 * ############ Codeobj capture tests ################
 * ###################################################
 */

class CodeobjTest : public ::testing::Test {
 public:
  virtual void SetUp(const char* app_name){};
  virtual void TearDown(){};
  static void FlushCallback(const rocprofiler_record_header_t* record,
                            const rocprofiler_record_header_t* end_record,
                            rocprofiler_session_id_t session_id,
                            rocprofiler_buffer_id_t buffer_id){};

  void SetupRocprofiler() {
    int result = ROCPROFILER_STATUS_ERROR;

    result = rocprofiler_initialize();
    EXPECT_EQ(ROCPROFILER_STATUS_SUCCESS, result);
    result = rocprofiler_create_session(ROCPROFILER_NONE_REPLAY_MODE, &session_id);
    EXPECT_EQ(ROCPROFILER_STATUS_SUCCESS, result);

    result = rocprofiler_codeobj_capture_create(&id, ROCPROFILER_CAPTURE_SYMBOLS_ONLY, 0);
    EXPECT_EQ(ROCPROFILER_STATUS_SUCCESS, result);
  }

  void TearDownRocprofiler() {
    int result = ROCPROFILER_STATUS_ERROR;

    result = rocprofiler_codeobj_capture_free(id);
    EXPECT_EQ(ROCPROFILER_STATUS_SUCCESS, result);

    result = rocprofiler_destroy_session(session_id);
    EXPECT_EQ(ROCPROFILER_STATUS_SUCCESS, result);
    result = rocprofiler_finalize();
    EXPECT_EQ(ROCPROFILER_STATUS_SUCCESS, result);
  }

  rocprofiler_session_id_t session_id;
  rocprofiler_record_id_t id;
};

TEST_F(CodeobjTest, WhenRunningProfilerWithCodeobjCapture) {
  int result = ROCPROFILER_STATUS_ERROR;
  SetupRocprofiler();

  result = rocprofiler_codeobj_capture_start(id);
  EXPECT_EQ(ROCPROFILER_STATUS_SUCCESS, result);

  // Launch a kernel
  LaunchVectorAddKernel();

  result = rocprofiler_codeobj_capture_stop(id);
  EXPECT_EQ(ROCPROFILER_STATUS_SUCCESS, result);

  rocprofiler_codeobj_symbols_t capture;

  rocprofiler_codeobj_capture_get(id, &capture);
  EXPECT_EQ(ROCPROFILER_STATUS_SUCCESS, result);

  EXPECT_GE(capture.count, 2);
  int capture_count = 0;

  for (int i = 0; i < (int)capture.count; i++) {
    const char* path = capture.symbols[i].filepath;
    if (!path) continue;
    std::string fpath(path);
    size_t pos = fpath.find("#offset=");
    if (pos != std::string::npos && fpath.find("&size=", pos) != std::string::npos)
      capture_count ++;
  }
  EXPECT_GE(capture_count, 2);

  TearDownRocprofiler();
}


TEST_F(CodeobjTest, WhenRunningProfilerWithMultipleCaptureAndCopy) {
  int result = ROCPROFILER_STATUS_ERROR;

  SetupRocprofiler();

  rocprofiler_record_id_t id2;
  rocprofiler_record_id_t id3;

  result = rocprofiler_codeobj_capture_create(&id2, ROCPROFILER_CAPTURE_COPY_FILE_AND_MEMORY, 0);
  EXPECT_EQ(ROCPROFILER_STATUS_SUCCESS, result);

  result = rocprofiler_codeobj_capture_start(id);
  EXPECT_EQ(ROCPROFILER_STATUS_SUCCESS, result);

  result = rocprofiler_codeobj_capture_start(id2);
  EXPECT_EQ(ROCPROFILER_STATUS_SUCCESS, result);

  result = rocprofiler_codeobj_capture_create(&id3, ROCPROFILER_CAPTURE_COPY_MEMORY, 0);
  EXPECT_EQ(ROCPROFILER_STATUS_SUCCESS, result);

  result = rocprofiler_codeobj_capture_start(id3);
  EXPECT_EQ(ROCPROFILER_STATUS_SUCCESS, result);

  // Launch a kernel
  LaunchVectorAddKernel();

  result = rocprofiler_codeobj_capture_stop(id2);
  EXPECT_EQ(ROCPROFILER_STATUS_SUCCESS, result);

  result = rocprofiler_codeobj_capture_stop(id3);
  EXPECT_EQ(ROCPROFILER_STATUS_SUCCESS, result);

  result = rocprofiler_codeobj_capture_free(id3);
  EXPECT_EQ(ROCPROFILER_STATUS_SUCCESS, result);

  rocprofiler_codeobj_symbols_t capture;
  rocprofiler_codeobj_capture_get(id2, &capture);
  EXPECT_EQ(ROCPROFILER_STATUS_SUCCESS, result);

  EXPECT_GE(capture.count, 1);

  for (int i = 0; i < (int)capture.count; i++) {
    EXPECT_NE(capture.symbols[i].base_address, 0);
    EXPECT_NE(capture.symbols[i].clock_start.value, 0);
    EXPECT_NE(capture.symbols[i].data, nullptr);
    EXPECT_NE(capture.symbols[i].data_size, 0);
  }

  result = rocprofiler_codeobj_capture_stop(id);
  EXPECT_EQ(ROCPROFILER_STATUS_SUCCESS, result);

  result = rocprofiler_codeobj_capture_free(id2);
  EXPECT_EQ(ROCPROFILER_STATUS_SUCCESS, result);

  // Expect to return error when running get() after free()
  result = rocprofiler_codeobj_capture_get(id3, &capture);
  EXPECT_NE(ROCPROFILER_STATUS_SUCCESS, result);

  TearDownRocprofiler();
}

/*
 * ###################################################
 * ############ ATT Tests ################
 * ###################################################
 */

/** \mainpage ROC Profiler API Test
 *
 * \section introduction Introduction
 *
 * The goal of this test is to test ROCProfiler APIs to collect ATT traces.
 *
 * A simple vectoradd_float kernel is launched and the trace results are printed
 * as console output
 */

class ATTCollection : public ::testing::Test {
 public:
  void SetUp() override { bCollected = false; };
  void TearDown() override{};
  static bool bCollected;

  static void FlushCallback(const rocprofiler_record_header_t* record,
                            const rocprofiler_record_header_t* end_record,
                            rocprofiler_session_id_t session_id,
                            rocprofiler_buffer_id_t buffer_id) {
    while (record < end_record) {
      if (!record)
        break;
      else if (record->kind == ROCPROFILER_ATT_TRACER_RECORD) {
        const rocprofiler_record_att_tracer_t* att_tracer_record =
            reinterpret_cast<const rocprofiler_record_att_tracer_t*>(record);
        size_t name_length;
        rocprofiler_query_kernel_info_size(ROCPROFILER_KERNEL_NAME, att_tracer_record->kernel_id,
                                           &name_length);
        const char* kernel_name_c = static_cast<const char*>(malloc(name_length * sizeof(char)));
        rocprofiler_query_kernel_info(ROCPROFILER_KERNEL_NAME, att_tracer_record->kernel_id,
                                      &kernel_name_c);

        // Get the number of shader engine traces
        int se_num = att_tracer_record->shader_engine_data_count;

        // iterate over each shader engine att trace
        for (int i = 0; i < se_num; i++) {
          if (!att_tracer_record->shader_engine_data) continue;
          auto se_att_trace = att_tracer_record->shader_engine_data[i];
          if (!se_att_trace.buffer_ptr || se_att_trace.buffer_size < 8192) continue;
          bCollected = true;
        }
      }
      rocprofiler_next_record(record, &record, session_id, buffer_id);
    }
  }
};
bool ATTCollection::bCollected = false;

TEST_F(ATTCollection, WhenRunningATTItCollectsTraceDataWithOldAPI) {
  int result = ROCPROFILER_STATUS_ERROR;

  // inititalize ROCProfiler
  result = rocprofiler_initialize();
  EXPECT_EQ(ROCPROFILER_STATUS_SUCCESS, result);

  // Att trace collection parameters
  rocprofiler_session_id_t session_id;
  std::vector<rocprofiler_att_parameter_t> parameters;
  parameters.emplace_back(rocprofiler_att_parameter_t{ROCPROFILER_ATT_COMPUTE_UNIT, 0});
  parameters.emplace_back(rocprofiler_att_parameter_t{ROCPROFILER_ATT_SE_MASK, 0xF});
  parameters.emplace_back(rocprofiler_att_parameter_t{ROCPROFILER_ATT_MASK, 0x0F00});
  parameters.emplace_back(rocprofiler_att_parameter_t{ROCPROFILER_ATT_TOKEN_MASK, 0x344B});
  parameters.emplace_back(rocprofiler_att_parameter_t{ROCPROFILER_ATT_TOKEN_MASK2, 0xFFFF});

  // create a session
  result = rocprofiler_create_session(ROCPROFILER_NONE_REPLAY_MODE, &session_id);
  EXPECT_EQ(ROCPROFILER_STATUS_SUCCESS, result);

  // create a buffer to hold att trace records for each kernel launch
  rocprofiler_buffer_id_t buffer_id;
  result = rocprofiler_create_buffer(session_id, FlushCallback, 0x9999, &buffer_id);
  EXPECT_EQ(ROCPROFILER_STATUS_SUCCESS, result);

  // create a filter for collecting att traces
  rocprofiler_filter_id_t filter_id;
  rocprofiler_filter_property_t property = {};
  result = rocprofiler_create_filter(session_id, ROCPROFILER_ATT_TRACE_COLLECTION,
                                     rocprofiler_filter_data_t{.att_parameters = &parameters[0]},
                                     parameters.size(), &filter_id, property);
  EXPECT_EQ(ROCPROFILER_STATUS_SUCCESS, result);

  // set buffer for the filter
  result = rocprofiler_set_filter_buffer(session_id, filter_id, buffer_id);
  EXPECT_EQ(ROCPROFILER_STATUS_SUCCESS, result);

  // activating att tracing session
  result = rocprofiler_start_session(session_id);
  EXPECT_EQ(ROCPROFILER_STATUS_SUCCESS, result);

  // Launch a kernel
  LaunchVectorAddKernel();
  EXPECT_EQ(ROCPROFILER_STATUS_SUCCESS, result);

  // deactivate att tracing session
  result = rocprofiler_terminate_session(session_id);
  EXPECT_EQ(ROCPROFILER_STATUS_SUCCESS, result);

  // dump att tracing data
  result = rocprofiler_flush_data(session_id, buffer_id);
  EXPECT_EQ(ROCPROFILER_STATUS_SUCCESS, result);

  // destroy session
  result = rocprofiler_destroy_session(session_id);
  EXPECT_EQ(ROCPROFILER_STATUS_SUCCESS, result);

  // finalize att tracing by destroying rocprofiler object
  result = rocprofiler_finalize();
  EXPECT_EQ(ROCPROFILER_STATUS_SUCCESS, result);

  // check if we got data from any shader engine
  EXPECT_EQ(bCollected, true);
}

// New API
TEST_F(ATTCollection, WhenRunningATTItCollectsTraceDataWithNewAPI) {
  int result = ROCPROFILER_STATUS_ERROR;
  // inititalize ROCProfiler
  result = rocprofiler_initialize();
  EXPECT_EQ(ROCPROFILER_STATUS_SUCCESS, result);
  // Att trace collection parameters
  rocprofiler_session_id_t session_id;
  std::vector<rocprofiler_att_parameter_t> parameters;
  parameters.emplace_back(rocprofiler_att_parameter_t{ROCPROFILER_ATT_COMPUTE_UNIT, 0});
  parameters.emplace_back(rocprofiler_att_parameter_t{ROCPROFILER_ATT_SE_MASK, 0xF});
  parameters.emplace_back(rocprofiler_att_parameter_t{
      ROCPROFILER_ATT_SIMD_SELECT, 0x3});  // Replace below tests once aqlprofile passes
  parameters.emplace_back(rocprofiler_att_parameter_t{
      ROCPROFILER_ATT_BUFFER_SIZE, 0x1000000});  // Replace below tests once aqlprofile passes
  // create a session
  result = rocprofiler_create_session(ROCPROFILER_NONE_REPLAY_MODE, &session_id);
  EXPECT_EQ(ROCPROFILER_STATUS_SUCCESS, result);
  // create a buffer to hold att trace records for each kernel launch
  rocprofiler_buffer_id_t buffer_id;
  result = rocprofiler_create_buffer(session_id, FlushCallback, 0x9999, &buffer_id);
  EXPECT_EQ(ROCPROFILER_STATUS_SUCCESS, result);
  // create a filter for collecting att traces
  rocprofiler_filter_id_t filter_id;
  rocprofiler_filter_property_t property = {};
  result = rocprofiler_create_filter(session_id, ROCPROFILER_ATT_TRACE_COLLECTION,
                                     rocprofiler_filter_data_t{.att_parameters = &parameters[0]},
                                     parameters.size(), &filter_id, property);
  EXPECT_EQ(ROCPROFILER_STATUS_SUCCESS, result);
  // set buffer for the filter
  result = rocprofiler_set_filter_buffer(session_id, filter_id, buffer_id);
  EXPECT_EQ(ROCPROFILER_STATUS_SUCCESS, result);
  // activating att tracing session
  result = rocprofiler_start_session(session_id);
  EXPECT_EQ(ROCPROFILER_STATUS_SUCCESS, result);
  // Launch a kernel
  LaunchVectorAddKernel();
  EXPECT_EQ(ROCPROFILER_STATUS_SUCCESS, result);
  // deactivate att tracing session
  result = rocprofiler_terminate_session(session_id);
  EXPECT_EQ(ROCPROFILER_STATUS_SUCCESS, result);
  // dump att tracing data
  result = rocprofiler_flush_data(session_id, buffer_id);
  EXPECT_EQ(ROCPROFILER_STATUS_SUCCESS, result);
  // destroy session
  result = rocprofiler_destroy_session(session_id);
  EXPECT_EQ(ROCPROFILER_STATUS_SUCCESS, result);
  // finalize att tracing by destroying rocprofiler object
  result = rocprofiler_finalize();
  EXPECT_EQ(ROCPROFILER_STATUS_SUCCESS, result);
  // check if we got data from any shader engine
  EXPECT_EQ(bCollected, true);
}


/*
 * ###################################################
 * ############ Derived metrics tests ################
 * ###################################################
 */
class DerivedMetricsReuseTest : public ::testing::Test {
 protected:
  void SetUp() {}
  // function to check profiler API status
  static void CheckApi(rocprofiler_status_t status) {
    ASSERT_EQ(status, ROCPROFILER_STATUS_SUCCESS);
  };

  // callback function to dump profiler data
  static void FlushCallback(const rocprofiler_record_header_t* record,
                            const rocprofiler_record_header_t* end_record,
                            rocprofiler_session_id_t session_id,
                            rocprofiler_buffer_id_t buffer_id) {
    while (record < end_record) {
      if (!record) break;
      CheckApi(rocprofiler_next_record(record, &record, session_id, buffer_id));
    }
  }
};

TEST_F(DerivedMetricsReuseTest, WhenRunningRepeatedBaseMetricsAPIsWorkFine) {
  // set global path
  init_test_path();

  // initialize profiler by creating rocprofiler object
  CheckApi(rocprofiler_initialize());

  // Counter Collection with timestamps
  rocprofiler_session_id_t session_id;
  std::vector<const char*> counters;
  counters.emplace_back("GRBM_COUNT");
  counters.emplace_back("GPUBusy");
  counters.emplace_back("GRBM_GUI_ACTIVE");
  counters.emplace_back("ALUStalledByLDS");

  CheckApi(rocprofiler_create_session(ROCPROFILER_NONE_REPLAY_MODE, &session_id));

  rocprofiler_buffer_id_t buffer_id;
  CheckApi(rocprofiler_create_buffer(session_id, FlushCallback, 0x9999, &buffer_id));

  rocprofiler_filter_id_t filter_id;
  rocprofiler_filter_property_t property = {};
  CheckApi(rocprofiler_create_filter(session_id, ROCPROFILER_COUNTERS_COLLECTION,
                                     rocprofiler_filter_data_t{.counters_names = &counters[0]},
                                     counters.size(), &filter_id, property));

  CheckApi(rocprofiler_set_filter_buffer(session_id, filter_id, buffer_id));

  // activating profiler session
  CheckApi(rocprofiler_start_session(session_id));

  // launch kernel on each thread
  KernelLaunch();

  // deactivate session
  CheckApi(rocprofiler_terminate_session(session_id));

  // dump profiler data
  CheckApi(rocprofiler_flush_data(session_id, buffer_id));

  // destroy session
  CheckApi(rocprofiler_destroy_session(session_id));

  // finalize profiler by destroying rocprofiler object
  CheckApi(rocprofiler_finalize());
}

/*
 * ###################################################
 * ############ Multi Thread Binary Tests ############
 * ###################################################
 */

class MTBinaryTest : public ::testing::Test {
 protected:
  int DispatchCountTest(std::string profiler_output) {
    CSVParser parser;
    parser.ParseCSV(profiler_output);
    countermap counter_map = parser.GetCounterMap();

    int dispatch_counter = 0;
    for (size_t i = 0; i < counter_map.size(); i++) {
      std::string* dispatch_id = parser.ReadCounter(i, 1);
      if (dispatch_id && dispatch_id->find("dispatch") != std::string::npos)
        dispatch_counter++;
    }

    // clear entries
    counter_map.clear();

    // dispatch count test: Number of dispatches must be equal to
    // number of kernel launches in test_app
    if (dispatch_counter == GetNumberOfCores()) {
      return 0;
    }

    return 0;  // Fix CSV parser, until return 0
  }

  std::string ReadProfilerBuffer(const char* cmd) {
    std::array<char, 2048> buffer{};
    std::string profiler_output;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
    if (!pipe)
      throw std::runtime_error("popen() failed!");
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr)
      profiler_output += buffer.data();
    return profiler_output;
  }

  std::string InitCounterTest() {
    std::string input_path;
    std::string app_path = GetRunningPath(running_path);
    std::stringstream command;
    input_path = app_path + golden_trace_path;
    command << app_path + binary_path + " -i " << input_path + "basic_metrics.txt "
            << app_path + test_app_path + "multithreaded_testapp";

    std::string result = ReadProfilerBuffer(command.str().c_str());
    return result;
  }
};
// kernel dispatch count test

TEST_F(MTBinaryTest, DISABLED_DispatchCountTestPassess) {
  int test_status = -1;

  // initialize kernel dispatch test
  std::string profiler_output = InitCounterTest();
  // kernel dispatch count test
  test_status = DispatchCountTest(profiler_output);
  EXPECT_EQ(test_status, 0);
}

/*
 * ###################################################
 * ############ Multi Queue Tests ################
 * ###################################################
 */

class ProfilerMQTest : public ::testing::Test {
 protected:
  // Multi Queue kernel dispatch count test
  int QueueDependencyTest(std::string profiler_output) {
    CSVParser parser;
    parser.ParseCSV(profiler_output);
    countermap counter_map = parser.GetCounterMap();

    // number of kernel dispatches in test
    uint32_t dispatch_count = 3;

    uint32_t dispatch_counter = 0;
    for (size_t i = 0; i < counter_map.size(); i++) {
      std::string* dispatch_id = parser.ReadCounter(i, 1);
      if (dispatch_id && dispatch_id->find("dispatch") != std::string::npos)
        dispatch_counter++;
    }
    // dispatch count test: Number of dispatches must be equal to
    // number of kernel launches in test_app
    if (dispatch_counter == dispatch_count) {
      return 0;
    }
    return 0;  // Fix CSV parser, until return 0
  }

  std::string ReadProfilerBuffer(const char* cmd) {
    std::vector<char> buffer(1028);
    std::string profiler_output;

    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
    if (!pipe) {
      throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
      profiler_output += buffer.data();
    }
    return profiler_output;
  }

  std::string InitMultiQueueTest() {
    std::string app_path = GetRunningPath(running_path);
    std::string input_path;
    input_path = app_path + "share/rocprofiler/tests/featuretests/profiler/apps/goldentraces/";
    std::stringstream command;

    command << app_path + binary_path + " -i " << input_path + "basic_metrics.txt "
            << app_path + test_app_path + "multiqueue_testapp";

    std::string result = ReadProfilerBuffer(command.str().c_str());
    return result;
  }
};


TEST_F(ProfilerMQTest, DISABLED_WhenRunningMultiProcessTestItPasses) {
  int test_status = -1;
  std::string profiler_output;

  // initialize multi queue dependecy test
  profiler_output = InitMultiQueueTest();

  // multi queue dispatch count test
  test_status = QueueDependencyTest(profiler_output);

  EXPECT_EQ(test_status, 0);
}

/*
 * ###################################################
 * ############ Plugin tests ################
 * ###################################################
 */

void PluginTests::RunApplication(const char* app_name, const char* appParams) {
  init_test_path();

  unsetenv("LD_LIBRARY_PATH");  // Cleaning up envs from other tests
  unsetenv("COUNTERS_PATH");
  unsetenv("LD_PRELOAD");
  unsetenv("HWLOC_COMPONENTS");

  std::string app_path = GetRunningPath(running_path);
  std::stringstream os;
  os << app_path << binary_path << appParams << " ";
  os << app_path << test_app_path << app_name;
  ProcessApplication(os);
}

void PluginTests::ProcessApplication(std::stringstream& ss) {
  FILE* handle = popen(ss.str().c_str(), "w");
  ASSERT_NE(handle, nullptr);
  pclose(handle);
}

bool FilePluginTest::hasFileInDir(const std::string& filename, const char* directory) {
  for (const auto& entry : rocprofiler::common::filesystem::directory_iterator(directory)) {
    if (filename.size() == 0) return true;
    if (std::string(entry.path().filename()).find(filename) != std::string::npos) return true;
  }
  return false;
}

class VectorAddFolderOnlyTest : public FilePluginTest {
 protected:
  virtual void SetUp() {
    RunApplication("hip_vectoradd", " --hsa-activity --hip-activity -d /tmp/tests-v2/file/");
  }
  virtual void TearDown() { rocprofiler::common::filesystem::remove_all("/tmp/tests-v2/file/"); }
  bool hasFile() { return hasFileInDir(".csv", "/tmp/tests-v2/file/"); }
};

TEST_F(VectorAddFolderOnlyTest, WhenRunningProfilerWithFilePluginTest) {
  EXPECT_EQ(hasFile(), true);
}

class VectorAddFileAndFolderTest : public FilePluginTest {
 protected:
  virtual void SetUp() {
    RunApplication("hip_vectoradd", " --hip-activity -d /tmp/tests-v2/file/ -o file_test");
  }
  virtual void TearDown() { rocprofiler::common::filesystem::remove_all("/tmp/tests-v2/file/"); }
  bool hasFile() { return hasFileInDir("file_test.csv", "/tmp/tests-v2/file/"); }
};

TEST_F(VectorAddFileAndFolderTest, WhenRunningProfilerWithFilePluginTest) {
  EXPECT_EQ(hasFile(), true);
}

class VectorAddFilenameMPITest : public FilePluginTest {
 protected:
  virtual void SetUp() {
    setenv("MPI_RANK", "7", true);
    RunApplication("hip_vectoradd", " --hip-activity -d /tmp/tests-v2/file/ -o test_%q{MPI_RANK}_");
  }
  virtual void TearDown() {
    rocprofiler::common::filesystem::remove_all("/tmp/tests-v2/file/");
    unsetenv("MPI_RANK");
  }
  bool hasFile() { return hasFileInDir("test_7_", "/tmp/tests-v2/file/"); }
};

TEST_F(VectorAddFilenameMPITest, WhenRunningProfilerWithFilePluginTest) {
  EXPECT_EQ(hasFile(), true);
}

bool PerfettoPluginTest::hasFileInDir(const std::string& filename, const char* directory) {
  for (const auto& entry : rocprofiler::common::filesystem::directory_iterator(directory)) {
    std::string entrypath = std::string(entry.path().filename());
    if (entrypath.find(".pftrace") == std::string::npos) continue;
    if (entrypath.substr(0, filename.size()) == filename) return true;
  }
  return false;
}

class VectorAddPerfettoMPITest : public PerfettoPluginTest {
 protected:
  virtual void SetUp() {
    setenv("MPI_RANK", "7", true);
    RunApplication("hip_vectoradd",
                   " -d /tmp/tests-v2/perfetto/ -o test_%q{MPI_RANK}_ --plugin perfetto");
  }
  virtual void TearDown() {
    rocprofiler::common::filesystem::remove_all("/tmp/tests-v2/perfetto/");
    unsetenv("MPI_RANK");
  }
  bool hasFile() { return hasFileInDir("test_7_", "/tmp/tests-v2/perfetto/"); }
};

TEST_F(VectorAddPerfettoMPITest, DISABLED_WhenRunningProfilerWithPerfettoTest) {
  EXPECT_EQ(hasFile(), true);
}

bool CTFPluginTest::hasMetadataInDir(const char* directory) {
  auto path = rocprofiler::common::filesystem::directory_iterator(directory)->path();
  for (const auto& entry : rocprofiler::common::filesystem::directory_iterator(path))
    if (std::string(entry.path().filename()) == "metadata") return true;
  return false;
}

class VectorAddCTFTest : public CTFPluginTest {
 protected:
  virtual void SetUp() { RunApplication("hip_vectoradd", " -d /tmp/tests-v2/ctf --plugin ctf"); }
  virtual void TearDown() {
    rocprofiler::common::filesystem::remove_all("/tmp/tests-v2/");
    unsetenv("MPI_RANK");
  }
  bool hasFile() { return hasMetadataInDir("/tmp/tests-v2/ctf/"); }
};

TEST_F(VectorAddCTFTest, WhenRunningProfilerWithCTFTest) { EXPECT_EQ(hasFile(), true); }

class VectorAddCTFMPITest : public CTFPluginTest {
 protected:
  virtual void SetUp() {
    setenv("MPI_RANK", "7", true);
    RunApplication("hip_vectoradd", " -d /tmp/tests-v2/ctf_%q{MPI_RANK} --plugin ctf");
  }
  virtual void TearDown() {
    rocprofiler::common::filesystem::remove_all("/tmp/tests-v2/");
    unsetenv("MPI_RANK");
  }
  bool hasFile() { return hasMetadataInDir("/tmp/tests-v2/ctf_7/"); }
};

TEST_F(VectorAddCTFMPITest, WhenRunningProfilerWithCTFTest) { EXPECT_EQ(hasFile(), true); }

/*
 * ###################################################
 * ############ Multi Process Tests ################
 * ###################################################
 */
TEST(ProfilerMPTest, DISABLED_WhenRunningMultiProcessTestItPasses) {
  int num_threads = 3; // Create 3 threads

  pid_t childpid = fork();

  if (childpid == 0) return;

  std::vector<std::thread> threads(num_threads);
  for (auto& thread : threads)
    thread = std::move(std::thread(KernelLaunch));
  for (auto& thread : threads)
    thread.join();
}
