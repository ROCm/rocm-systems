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
#include "test_utils.h"
#include <regex>

namespace rocprofiler {
namespace tests {
namespace utility {

// This function returns the running path of executable
std::string GetRunningPath(std::string string_to_erase) {
  std::string path;
  char* real_path;
  Dl_info dl_info;

  if (0 != dladdr(reinterpret_cast<void*>(main), &dl_info)) {
    std::string to_erase = string_to_erase;
    path = dl_info.dli_fname;
    real_path = realpath(path.c_str(), NULL);
    if (real_path == nullptr) {
      throw(std::string("Error! in extracting real path"));
    }
    path.clear();  // reset path
    path.append(real_path);

    size_t pos = path.find(to_erase);
    if (pos != std::string::npos) path.erase(pos, to_erase.length());
  } else {
    throw(std::string("Error! in extracting real path"));
  }
  return path;
}

// This function returns number of cores
// available in system
int GetNumberOfCores() {
  std::ifstream cpuinfo("/proc/cpuinfo");
  const int num_cpu_cores =
      std::count(std::istream_iterator<std::string>(cpuinfo), std::istream_iterator<std::string>(),
                 std::string("processor"));
  return num_cpu_cores;
}

bool is_installed_path() {
  std::string path;
  char* real_path;
  Dl_info dl_info;

  if (0 != dladdr(reinterpret_cast<void*>(main), &dl_info)) {
    path = dl_info.dli_fname;
    real_path = realpath(path.c_str(), NULL);
    if (real_path == nullptr) {
      throw(std::string("Error! in extracting real path"));
    }
    path.clear();  // reset path
    path.append(real_path);
    if (path.find("/opt") != std::string::npos) {
      return true;
    }
  }
  return false;
}

// tokenize profiler output
void tokenize_profiler_output(std::string line, profiler_kernel_info_t& kinfo) {
  std::stringstream tokenStream(line);
  std::string token;
  std::getline(tokenStream, token, ',');
  kinfo.dispatch_id = token;
  std::getline(tokenStream, token, ',');
  kinfo.gpu_id = token;
  std::getline(tokenStream, token, ',');
  kinfo.queue_id = token;
  std::getline(tokenStream, token, ',');
  kinfo.process_id = token;
  std::getline(tokenStream, token, ',');
  kinfo.thread_id = token;
  std::getline(tokenStream, token, ',');
  kinfo.grid_size = token;
  std::getline(tokenStream, token, ',');
  kinfo.workgroup_size = token;
  std::getline(tokenStream, token, ',');
  kinfo.lds_per_workgroup = token;
  std::getline(tokenStream, token, ',');
  kinfo.scratch_per_workitem = token;
  std::getline(tokenStream, token, ',');
  kinfo.arch_vgpr = token;
  std::getline(tokenStream, token, ',');
  kinfo.accum_vgpr = token;
  std::getline(tokenStream, token, ',');
  kinfo.sgpr = token;
  std::getline(tokenStream, token, ',');
  kinfo.wave_size = token;
  std::getline(tokenStream, token, ',');
  kinfo.kernel_name = token;
  std::getline(tokenStream, token, ',');
  kinfo.begin_time = token;
  std::getline(tokenStream, token, ',');
  kinfo.end_time = token;
  std::getline(tokenStream, token, ',');
  kinfo.correlation_id = token;
  std::getline(tokenStream, token, ',');
  kinfo.counter = token;
}

// tokenize tracer output
void tokenize_tracer_output(std::string line, tracer_kernel_info_t& kinfo) {
  std::stringstream tokenStream(line);
  std::string token;
  std::getline(tokenStream, token, ',');
  kinfo.domain = token;
  std::getline(tokenStream, token, ',');
  int version_position = token.find('R');
  if (version_position != std::string::npos) {
    token = token.substr(0, version_position) + ')';
  }
  kinfo.function = token;
  std::getline(tokenStream, token, ',');
  kinfo.begin_time = token;
  std::getline(tokenStream, token, ',');
  kinfo.end_time = token;
  std::getline(tokenStream, token, ',');
  kinfo.corelation_id = token;
  std::getline(tokenStream, token, ',');
  kinfo.roctx_id = token;
  std::getline(tokenStream, token, ',');
  kinfo.roxtx_msg = token;
}

// get numeric value of timestamp token
uint64_t get_timestamp_value(const std::string& str) {
  std::regex pattern("(\\d+)");
  std::smatch match;

  if (regex_search(str, match, pattern)) {
    return stoul(match[1]);
  } else {
    return -1;
  }
}


}  // namespace utility
}  // namespace tests
}  // namespace rocprofiler
