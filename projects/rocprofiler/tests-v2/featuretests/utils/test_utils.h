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
#ifndef TESTS_FEATURETESTS_PROFILER_UTILS_TEST_UTILS_H_
#define TESTS_FEATURETESTS_PROFILER_UTILS_TEST_UTILS_H_

#include <cxxabi.h>    // for __cxa_demangle
#include <dlfcn.h>     // for dladdr
#include <execinfo.h>  // for backtrace

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iterator>
#include <string>

namespace rocprofiler {
namespace tests {
namespace utility {

typedef struct {
  std::string dispatch_id;
  std::string gpu_id;
  std::string queue_id;
  std::string process_id;
  std::string thread_id;
  std::string grid_size;
  std::string workgroup_size;
  std::string lds_per_workgroup;
  std::string scratch_per_workitem;
  std::string arch_vgpr;
  std::string accum_vgpr;
  std::string sgpr;
  std::string wave_size;
  std::string kernel_name;
  std::string begin_time;
  std::string end_time;
  std::string correlation_id;
  std::string counter;
} profiler_kernel_info_t;

typedef struct {
  std::string domain;
  std::string function;
  std::string operation;
  std::string kernel_name;
  std::string begin_time;
  std::string end_time;
  std::string corelation_id;
  std::string roctx_id;
  std::string roxtx_msg;
} tracer_kernel_info_t;

// Get current running path
std::string GetRunningPath(std::string string_to_erase);

// Get Number of cores in the system
int GetNumberOfCores();

// Check if running path is /opt/rocm or not
bool is_installed_path();

// tokenize profiler output
void tokenize_profiler_output(std::string line, profiler_kernel_info_t& kinfo);

// tokenize tracer output
void tokenize_tracer_output(std::string line, tracer_kernel_info_t& kinfo);

// get numeric value of timestamp token
uint64_t get_timestamp_value(const std::string& str);

}  // namespace utility
}  // namespace tests
}  // namespace rocprofiler

// used for dl_addr to locate the running
// path for executable
int main(int argc, char** argv);

using rocprofiler::tests::utility::get_timestamp_value;
using rocprofiler::tests::utility::GetNumberOfCores;
using rocprofiler::tests::utility::GetRunningPath;
using rocprofiler::tests::utility::is_installed_path;
using rocprofiler::tests::utility::profiler_kernel_info_t;
using rocprofiler::tests::utility::tokenize_profiler_output;
using rocprofiler::tests::utility::tokenize_tracer_output;
using rocprofiler::tests::utility::tracer_kernel_info_t;

#endif  // TESTS_FEATURETESTS_PROFILER_UTILS_TEST_UTILS_H_
