/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Stands in for a profiler library: rocprofiler-register loads it (ROCPROFILER_REGISTER_LIBRARY)
// and hands it the HIP dispatch tables while the runtime registers them. Like tool libraries
// that initialize themselves with HIP calls, it calls back into HIP from there.

#include <hip/hip_runtime_api.h>
#include <cstdint>
#include <cstring>

extern "C" {
//! Tables handed to the tool: bit 0 "hip", bit 1 "hip_compiler", bit 2 "hip_tools"
__attribute__((visibility("default"))) unsigned dispatch_table_reentry_tables = 0;
//! First failing status of the HIP calls made from the tool
__attribute__((visibility("default"))) hipError_t dispatch_table_reentry_status = hipSuccess;

__attribute__((visibility("default"))) int rocprofiler_set_api_table(const char* name, uint64_t,
                                                                    uint64_t, void**, uint64_t) {
  static const char* const kTables[] = {"hip", "hip_compiler", "hip_tools"};
  for (unsigned i = 0; i < 3; ++i) {
    if (std::strcmp(name, kTables[i]) == 0) {
      dispatch_table_reentry_tables |= 1u << i;
    }
  }
  int version = 0;
  const hipError_t status = hipRuntimeGetVersion(&version);
  if (dispatch_table_reentry_status == hipSuccess) {
    dispatch_table_reentry_status = status;
  }
  return 0;
}
}
