/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup HRR HRR dispatch-table test tool
 * @{
 * @ingroup HRRTest
 * A minimal rocprofiler-register tool for hrr_dispatch_swap_test.cc.
 *
 * rocprofiler-register loads the library named by ROCPROFILER_REGISTER_LIBRARY
 * and passes it each runtime's tables, which attaches it to the process the way
 * a profiler is. This one keeps the HIP table, so the workload can tell whether
 * it got one.
 */

#include <atomic>
#include <cstdint>
#include <cstring>

#define HRR_TOOL_EXPORT extern "C" __attribute__((visibility("default")))

namespace {
std::atomic<void*> g_hip_table{nullptr};
}  // namespace

HRR_TOOL_EXPORT int rocprofiler_set_api_table(const char* name, uint64_t /*lib_version*/,
                                              uint64_t /*lib_instance*/, void** tables,
                                              uint64_t num_tables) {
  if (name && std::strcmp(name, "hip") == 0 && tables && num_tables > 0)
    g_hip_table.store(tables[0]);
  return 0;
}

// The live HipDispatchTable, or null if rocprofiler-register never passed one.
HRR_TOOL_EXPORT void* hrr_dispatch_tool_hip_table() { return g_hip_table.load(); }

/** @} */
