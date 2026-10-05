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
 * libamdhip64 does not export its live dispatch table. A profiler gets it from
 * rocprofiler-register, which loads the library named by
 * ROCPROFILER_REGISTER_LIBRARY and passes it each runtime's tables. This one
 * keeps the HIP table so the workload can wrap a slot the way a profiler
 * attaching after capture would.
 *
 * It also registers an exit handler when the HIP table arrives. That happens on
 * the first use of the table, before hip::init() registers
 * hip_capture_shutdown(), so the handler runs after the capture has shut down
 * and can inspect what shutdown left behind.
 *
 * Preloaded with LD_PRELOAD, it also stands in front of fopen(), so a workload
 * can act at the moment the capture opens a file, e.g. between the trailer and
 * the manifest. Without a hook it only forwards the call.
 */

#include <dlfcn.h>

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#define HRR_TOOL_EXPORT extern "C" __attribute__((visibility("default")))

namespace {
std::atomic<void*> g_hip_table{nullptr};
std::atomic<void (*)()> g_late_hook{nullptr};

void run_late_hook() {
  if (void (*hook)() = g_late_hook.load(std::memory_order_acquire)) hook();
}

using FopenFn = void* (*)(const char*, const char*);
std::atomic<void (*)(const char*)> g_fopen_hook{nullptr};

// Runs the hook, then the next definition of `name`. FILE is left out on
// purpose: with _FILE_OFFSET_BITS=64 <cstdio> renames fopen() to fopen64().
void* hooked_fopen(const char* name, std::atomic<FopenFn>& next, const char* path,
                   const char* mode) {
  if (void (*hook)(const char*) = g_fopen_hook.load(std::memory_order_acquire)) hook(path);
  FopenFn fn = next.load(std::memory_order_acquire);
  if (!fn) {
    fn = reinterpret_cast<FopenFn>(dlsym(RTLD_NEXT, name));
    next.store(fn, std::memory_order_release);
  }
  return fn ? fn(path, mode) : nullptr;
}

std::atomic<FopenFn> g_next_fopen{nullptr};
std::atomic<FopenFn> g_next_fopen64{nullptr};
}  // namespace

HRR_TOOL_EXPORT void* hrr_dispatch_tool_fopen(const char* path, const char* mode) __asm__("fopen");
HRR_TOOL_EXPORT void* hrr_dispatch_tool_fopen64(const char* path, const char* mode)
    __asm__("fopen64");

void* hrr_dispatch_tool_fopen(const char* path, const char* mode) {
  return hooked_fopen("fopen", g_next_fopen, path, mode);
}

void* hrr_dispatch_tool_fopen64(const char* path, const char* mode) {
  return hooked_fopen("fopen64", g_next_fopen64, path, mode);
}

HRR_TOOL_EXPORT int rocprofiler_set_api_table(const char* name, uint64_t /*lib_version*/,
                                              uint64_t /*lib_instance*/, void** tables,
                                              uint64_t num_tables) {
  if (name && std::strcmp(name, "hip") == 0 && tables && num_tables > 0) {
    void* expected = nullptr;
    if (g_hip_table.compare_exchange_strong(expected, tables[0])) std::atexit(run_late_hook);
  }
  return 0;
}

// The live HipDispatchTable, or null if rocprofiler-register never passed one.
HRR_TOOL_EXPORT void* hrr_dispatch_tool_hip_table() { return g_hip_table.load(); }

// Runs `hook` from the exit handler above, after hip_capture_shutdown().
HRR_TOOL_EXPORT void hrr_dispatch_tool_set_late_hook(void (*hook)()) {
  g_late_hook.store(hook, std::memory_order_release);
}

// Runs `hook` with the path on every fopen() while the library is preloaded.
HRR_TOOL_EXPORT void hrr_dispatch_tool_set_fopen_hook(void (*hook)(const char*)) {
  g_fopen_hook.store(hook, std::memory_order_release);
}

/** @} */
