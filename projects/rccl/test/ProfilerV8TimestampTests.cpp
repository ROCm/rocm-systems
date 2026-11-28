/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// ProfilerV8Timestamp: a plugin exporting only ncclProfiler_v8 that selects a CPU clock in
// ncclProfilerConfig_v8_t must be loaded, and must get every host API event start, stop and
// state transition stamped by RCCL on that clock (NCCL 2.32.6 profiler v8).

#include <gtest/gtest.h>
#include <rccl/rccl.h>
#include <hip/hip_runtime.h>

#include <dlfcn.h>
#include <time.h>
#include <unistd.h>

#include <filesystem>
#include <string>

#include "common/ProcessIsolatedTestRunner.hpp"
#include "plugin/profiler_v8_stamp_plugin.h"

namespace RcclUnitTesting {
namespace {

// MAX_STR_LEN in src/plugin/plugin_open.cc: a longer NCCL_PROFILER_PLUGIN is truncated.
constexpr size_t kPluginPathLimit = 255;

// Built and installed beside the test binary.
std::string v8PluginPath() {
  std::error_code ec;
  std::filesystem::path exe = std::filesystem::read_symlink("/proc/self/exe", ec);
  if (ec) return "";
  return (exe.parent_path() / RCCL_TEST_PROFILER_V8_NAME).string();
}

uint64_t monotonicNs() {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}

} // namespace

// Isolated because the profiler plugin is loaded once per process, at the first communicator.
TEST(ProfilerV8Timestamp, HostApiEventsCarryNcclStampedCpuClock) {
  RUN_ISOLATED_TEST("ProfilerV8Timestamp.HostApiEventsCarryNcclStampedCpuClock", []() {
    // Single-rank collectives return before any API event is emitted, so two ranks are needed.
    if (int n = 0; hipGetDeviceCount(&n) != hipSuccess || n < 2) GTEST_SKIP() << "requires at least 2 GPUs";
    const std::string path = v8PluginPath();
    ASSERT_FALSE(path.empty()) << "could not resolve the test binary's own directory";
    if (path.size() >= kPluginPathLimit) GTEST_SKIP() << "plugin path exceeds the plugin path limit: " << path;
    // RCCL's own dlopen of the same path returns this handle, so the counters below are the ones it fills.
    void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    ASSERT_NE(handle, nullptr) << "cannot dlopen " << path << ": " << dlerror();
    auto snapshot = reinterpret_cast<RcclProfV8SnapshotFn>(dlsym(handle, RCCL_TEST_PROF_V8_SNAPSHOT));
    ASSERT_NE(snapshot, nullptr);
    ASSERT_EQ(setenv("NCCL_PROFILER_PLUGIN", path.c_str(), 1), 0);

    constexpr int kRanks = 2;
    constexpr size_t kCount = 1024;
    ncclComm_t comms[kRanks];
    hipStream_t streams[kRanks];
    float* bufs[kRanks];
    // Ahead of comm init, so any event RCCL emits while setting up also falls inside the window.
    const uint64_t before = monotonicNs();
    ASSERT_EQ(ncclCommInitAll(comms, kRanks, nullptr), ncclSuccess);
    for (int r = 0; r < kRanks; ++r) {
      ASSERT_EQ(hipSetDevice(r), hipSuccess);
      ASSERT_EQ(hipStreamCreate(&streams[r]), hipSuccess);
      ASSERT_EQ(hipMalloc(&bufs[r], kCount * sizeof(float)), hipSuccess);
    }
    ASSERT_EQ(ncclGroupStart(), ncclSuccess);
    for (int r = 0; r < kRanks; ++r)
      ASSERT_EQ(ncclAllReduce(bufs[r], bufs[r], kCount, ncclFloat, ncclSum, comms[r], streams[r]), ncclSuccess);
    ASSERT_EQ(ncclGroupEnd(), ncclSuccess);
    for (int r = 0; r < kRanks; ++r) ASSERT_EQ(hipStreamSynchronize(streams[r]), hipSuccess);
    const uint64_t after = monotonicNs();
    RcclProfV8Stats s;
    snapshot(&s);
    for (int r = 0; r < kRanks; ++r) {
      ASSERT_EQ(ncclCommDestroy(comms[r]), ncclSuccess);
      ASSERT_EQ(hipStreamDestroy(streams[r]), hipSuccess);
      ASSERT_EQ(hipFree(bufs[r]), hipSuccess);
    }

    ASSERT_GE(s.inits, (uint64_t)kRanks) << "RCCL did not initialize the ncclProfiler_v8 plugin on every comm";
    EXPECT_EQ(s.initsNoConfig, 0u) << "init() got no ncclProfilerConfig_v8_t to opt in through";
    // One group API event with its start/end transitions, and one collective API event per rank.
    ASSERT_GE(s.starts, (uint64_t)kRanks + 1) << "API events missing";
    EXPECT_GE(s.stops, (uint64_t)kRanks + 1) << "API event stops missing";
    EXPECT_GE(s.states, 2u) << "group API state transitions missing";
    EXPECT_EQ(s.unstamped, 0u) << "events delivered without an NCCL-stamped CLOCK_MONOTONIC tsCpu";
    EXPECT_EQ(s.stopBeforeStart, 0u) << "stop stamped before its own start";
    EXPECT_GE(s.minTs, before) << "tsCpu earlier than the calls that produced it: not CLOCK_MONOTONIC";
    EXPECT_LE(s.maxTs, after) << "tsCpu later than the calls that produced it: not CLOCK_MONOTONIC";
    dlclose(handle);
  });
}

} // namespace RcclUnitTesting
