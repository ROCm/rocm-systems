
/* Copyright (c) 2022 Advanced Micro Devices, Inc.

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
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#include <gtest/gtest.h>
#include "src/core/hardware/hsa_info.h"
#include "api/rocprofiler_singleton.h"
#include "src/core/hsa/hsa_support.h"
#include "tests-v2/HSAToolLibrary/HSATool.h"

// used for dl_addr to locate the running
// path for executable
int main(int argc, char** argv);
std::string metrics_path;
std::string running_path;

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


static void init_test_path() {
  metrics_path = "libexec/rocprofiler/counters/derived_counters.xml";
  if (is_installed_path()) {
    running_path = "share/rocprofiler/tests/runCoreUnitTests";
  } else {
    running_path = "tests-v2/unittests/core/runCoreUnitTests";
  }
}


// This function returns the running path of executable
std::string GetRunningPath(std::string string_to_erase) {
  std::string path;
  const char* real_path;
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


int HSATool_onload_callback(void* table, uint64_t runtime_version, uint64_t failed_tool_count,
                            const char* const* failed_tool_names) {
  rocprofiler::HSASupport_Singleton& hsasupport_singleton =
      rocprofiler::HSASupport_Singleton::GetInstance();
  hsasupport_singleton.HSAInitialize(reinterpret_cast<HsaApiTable*>(table));
  return true;
}


int main(int argc, char** argv) {
  init_test_path();
  SetHSACallback(HSATool_onload_callback);
  metrics_path = "libexec/rocprofiler/counters/derived_counters.xml";
  std::string app_path = GetRunningPath(running_path);
  std::stringstream gfx_path;
  gfx_path << app_path << metrics_path;
  setenv("ROCPROFILER_METRICS_PATH", gfx_path.str().c_str(), true);
  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  hsa_init();
  int status = RUN_ALL_TESTS();
  hsa_shut_down();
  return status;
}
