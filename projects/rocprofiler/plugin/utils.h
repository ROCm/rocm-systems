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
#include <systemd/sd-id128.h>

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <string>

#include "src/utils/helper.h"

// Macro to check ROCProfiler calls status
#define CHECK_ROCPROFILER(call)                                                                      \
  do {                                                                                             \
    if ((call) != ROCPROFILER_STATUS_SUCCESS) rocmtools::fatal("Error: ROCProfiler API Call Error!");  \
  } while (false)

namespace {

[[maybe_unused]] uint32_t GetPid() {
  static uint32_t pid = syscall(__NR_getpid);
  return pid;
}

[[maybe_unused]] uint64_t GetMachineID() {
  char hostname[1023] = "\0";
  gethostname(hostname, 1023);
  sd_id128_t ret;
  char machine_id[SD_ID128_STRING_MAX];
  [[maybe_unused]] int status = sd_id128_get_machine(&ret);
  assert(status == 0 && "Error: Couldn't get machine id!");
  if (sd_id128_to_string(ret, machine_id)) return std::hash<std::string>{}(machine_id);
  return std::rand();
}

}  // namespace
