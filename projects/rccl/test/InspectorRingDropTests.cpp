/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// InspectorRingDrop: the Inspector profiler plugin counts completed collectives
// that overflow its per-communicator ring before a dump drains them, and each
// JSON dump carries a per-communicator "dump_stats" record with those counts
// (plugins/profiler/inspector, NCCL 2.32.3). WritesCollRecords runs the same
// workload and checks only that the Inspector recorded it.

#include <gtest/gtest.h>
#include <rccl/rccl.h>
#include <hip/hip_runtime.h>

#include <unistd.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "common/ProcessIsolatedTestRunner.hpp"

#ifndef RCCL_INSPECTOR_PLUGIN_SO
#define RCCL_INSPECTOR_PLUGIN_SO ""  // no in-tree Inspector build: the tests skip
#endif

namespace RcclUnitTesting {
namespace {

constexpr int kNumGpus = 2;         // a one-rank AllReduce is a local copy with no profiler events
constexpr int kIterations = 16;     // AllReduces per rank, all completed before the only dump
constexpr size_t kCount = 1 << 16;  // 256 KiB of floats, above the 8 KiB tracking floor

// Integer value of `key` in one line of the Inspector's compact JSON, or -1.
long long jsonField(const std::string& line, const std::string& key) {
  const std::string tag = "\"" + key + "\":";
  const size_t pos = line.find(tag);
  return pos == std::string::npos ? -1 : std::strtoll(line.c_str() + pos + tag.size(), nullptr, 10);
}

// Runs kIterations AllReduces per rank through a one-entry collective ring that only
// communicator teardown drains, then hands the JSON the Inspector wrote to `check`.
void runInspector(void (*check)(const std::vector<std::string>& lines)) {
  const char* pluginPath = RCCL_INSPECTOR_PLUGIN_SO;
  int nGpus = 0;
  if (hipGetDeviceCount(&nGpus) != hipSuccess || nGpus < kNumGpus) GTEST_SKIP() << "requires 2 GPUs";
  if (access(pluginPath, R_OK) != 0) GTEST_SKIP() << "Inspector plugin not built: " << pluginPath;

  char tmpl[] = "/tmp/rccl_inspector_drop_XXXXXX";
  ASSERT_NE(mkdtemp(tmpl), nullptr);
  const std::filesystem::path dir(tmpl), dump = dir / "dump", plugin = dir / "inspector.so";
  // NCCL_PROFILER_PLUGIN is cut at 254 bytes; a short link loads the plugin from any build path.
  std::error_code ec;
  std::filesystem::create_symlink(pluginPath, plugin, ec);
  ASSERT_FALSE(ec) << ec.message();
  setenv("NCCL_PROFILER_PLUGIN", plugin.c_str(), 1);
  setenv("NCCL_INSPECTOR_ENABLE", "1", 1);
  setenv("NCCL_INSPECTOR_DUMP_DIR", dump.c_str(), 1);
  setenv("NCCL_INSPECTOR_DUMP_COLL_RING_SIZE", "1", 1);
  setenv("NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS", "-1", 1);
  setenv("NCCL_INSPECTOR_REQUIRE_KERNEL_TIMING", "0", 1);

  int devs[kNumGpus] = {0, 1};
  ncclComm_t comms[kNumGpus];
  void* bufs[kNumGpus];
  hipStream_t streams[kNumGpus];
  ASSERT_EQ(ncclCommInitAll(comms, kNumGpus, devs), ncclSuccess);
  for (int i = 0; i < kNumGpus; i++) {
    ASSERT_EQ(hipSetDevice(i), hipSuccess);
    ASSERT_EQ(hipMalloc(&bufs[i], kCount * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipStreamCreate(&streams[i]), hipSuccess);
  }
  // Grouped: RCCL's DDA, CE 2-shot and GIN SDMA AllReduce paths, which emit no
  // collective profiler events, only run ungrouped.
  for (int it = 0; it < kIterations; it++) {
    ASSERT_EQ(ncclGroupStart(), ncclSuccess);
    for (int i = 0; i < kNumGpus; i++)
      ASSERT_EQ(ncclAllReduce(bufs[i], bufs[i], kCount, ncclFloat, ncclSum, comms[i], streams[i]), ncclSuccess);
    ASSERT_EQ(ncclGroupEnd(), ncclSuccess);
    for (int i = 0; i < kNumGpus; i++) ASSERT_EQ(hipStreamSynchronize(streams[i]), hipSuccess);
  }
  // The profiler thread turns kernel completions into records asynchronously, and
  // teardown discards any it has not reached yet.
  usleep(500 * 1000);
  for (int i = 0; i < kNumGpus; i++) {
    ASSERT_EQ(ncclCommDestroy(comms[i]), ncclSuccess);
    EXPECT_EQ(hipFree(bufs[i]), hipSuccess);
    EXPECT_EQ(hipStreamDestroy(streams[i]), hipSuccess);
  }
  std::vector<std::string> lines;
  for (const auto& entry : std::filesystem::directory_iterator(dump, ec)) {
    std::ifstream in(entry.path());
    for (std::string line; std::getline(in, line);) lines.push_back(line);
  }
  std::filesystem::remove_all(dir, ec);
  check(lines);
}

} // namespace

// Isolated because the profiler plugin and its settings are latched by the first
// communicator a process creates.
TEST(InspectorRingDrop, DumpStatsReportDrops) {
  RUN_ISOLATED_TEST("InspectorRingDrop.DumpStatsReportDrops", []() {
    runInspector([](const std::vector<std::string>& lines) {
      long long dropped = -1;
      for (const std::string& line : lines) {
        if (line.find("\"dump_stats\":") == std::string::npos) continue;
        EXPECT_LE(jsonField(line, "coll_records"), 1) << line;
        dropped = std::max(dropped, jsonField(line, "coll_dropped_total"));
      }
      ASSERT_NE(dropped, -1) << "no dump_stats record in " << lines.size() << " lines of Inspector output";
      EXPECT_GT(dropped, 0) << kIterations << " AllReduces per rank through a one-entry ring, none reported dropped";
    });
  });
}

TEST(InspectorRingDrop, WritesCollRecords) {
  RUN_ISOLATED_TEST("InspectorRingDrop.WritesCollRecords", []() {
    runInspector([](const std::vector<std::string>& lines) {
      EXPECT_TRUE(std::any_of(lines.begin(), lines.end(), [](const std::string& line) {
        return line.find("\"coll_perf\":") != std::string::npos;
      })) << "no coll_perf record in " << lines.size() << " lines of Inspector output";
    });
  });
}

} // namespace RcclUnitTesting
