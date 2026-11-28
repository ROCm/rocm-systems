/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime_api.h>
#include <dlfcn.h>
#include <unistd.h>
#include <cstdlib>
#include <iostream>

// Exit codes, also used by Unit_hipDispatchTable_ToolCallsHipDuringRegistration
constexpr int kPassed = 0;
constexpr int kFailed = 1;
constexpr int kToolNotLoaded = 2;

// Initializes HIP with dispatchTableReentryTool loaded as profiler library (set up by the test
// through ROCPROFILER_REGISTER_LIBRARY). The tool calls HIP while the runtime registers its
// dispatch tables, which must neither deadlock nor abort; the alarm turns a hang into a failure.
int main() {
  alarm(60);
  int count = 0;
  const hipError_t status = hipGetDeviceCount(&count);
  if (status != hipSuccess) {
    std::cout << "hipGetDeviceCount failed: " << status << std::endl;
    return kFailed;
  }

  const char* tool_path = std::getenv("ROCPROFILER_REGISTER_LIBRARY");
  void* tool = tool_path != nullptr ? dlopen(tool_path, RTLD_NOW | RTLD_NOLOAD) : nullptr;
  const auto* tables = tool != nullptr
                           ? static_cast<unsigned*>(dlsym(tool, "dispatch_table_reentry_tables"))
                           : nullptr;
  if (tables == nullptr || *tables == 0) {
    std::cout << "the tool library was not handed any dispatch table" << std::endl;
    return kToolNotLoaded;
  }
  // The "hip" table is the one whose registration the tool re-enters. The others reach the
  // tool only when registered outside its callback ("hip_tools" is registered by HIP init,
  // which here runs inside it), so they aren't required.
  constexpr unsigned kHipTable = 0x1;
  if ((*tables & kHipTable) == 0) {
    std::cout << "tables handed to the tool: 0x" << std::hex << *tables << std::endl;
    return kFailed;
  }
  const auto* tool_status =
      static_cast<hipError_t*>(dlsym(tool, "dispatch_table_reentry_status"));
  if (tool_status == nullptr || *tool_status != hipSuccess) {
    std::cout << "HIP call from the tool failed: " << *tool_status << std::endl;
    return kFailed;
  }
  return kPassed;
}
