/* Copyright (c) 2018-2022 Advanced Micro Devices, Inc.

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

#pragma once

#include <cxxabi.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <sys/types.h>

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <set>
#include <sstream>

#include "src/utils/helper.h"

// Macro to check ROCProfiler calls status
#define CHECK_ROCPROFILER(call)                                                                    \
  do {                                                                                             \
    if ((call) != ROCPROFILER_STATUS_SUCCESS)                                                      \
      rocprofiler::fatal("Error: ROCProfiler API Call Error!");                                    \
  } while (false)

namespace {

[[maybe_unused]] uint32_t GetPid() {
  static uint32_t pid = syscall(__NR_getpid);
  return pid;
}

[[maybe_unused]] uint64_t GetMachineID() { return gethostid(); }

[[maybe_unused]] std::set<std::string> GetKernelFilters() {
    std::set<std::string> ret;
    if (const char* line_c_str = getenv("ROCPROFILER_KERNEL_FILTER")) {
      std::stringstream ss(std::string{line_c_str});
      std::string filter_name;
      while(std::getline(ss, filter_name, ' '))
      {
        if (filter_name.find("kernel:") != std::string::npos) {
          continue;
        }
        ret.insert(filter_name);
      }
    }
    return ret;
}

}  // namespace
