/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// InspectorCollAlgo: the Inspector profiler plugin's JSON output carries, in each
// collective's "coll_perf" record, the algorithm and protocol RCCL's core chose for it
// ("coll_algo"/"coll_proto", plugins/profiler/inspector, NCCL 2.32.6). WritesCollRecords
// runs the same AllReduces and only checks that the Inspector wrote collective records.

#include <gtest/gtest.h>
#include <rccl/rccl.h>
#include <hip/hip_runtime.h>

#include <unistd.h>

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
constexpr int kIterations = 4;
constexpr size_t kCount = 1 << 16;  // 256 KiB of floats, above the 8 KiB tracking floor

// String value of `key` in one line of the Inspector's compact JSON; `found` says whether the key is there.
std::string jsonStrField(const std::string& line, const std::string& key, bool& found) {
  const std::string tag = "\"" + key + "\":\"";
  const size_t pos = line.find(tag);
  found = pos != std::string::npos;
  if (!found) return "";
  const size_t start = pos + tag.size();
  return line.substr(start, line.find('"', start) - start);
}

// Runs grouped AllReduces with the Inspector loaded and returns the "coll_perf" lines it wrote.
void runInspector(std::vector<std::string>& collLines) {
  const char* pluginPath = RCCL_INSPECTOR_PLUGIN_SO;
  int nGpus = 0;
  if (hipGetDeviceCount(&nGpus) != hipSuccess || nGpus < kNumGpus) GTEST_SKIP() << "requires 2 GPUs";
  if (access(pluginPath, R_OK) != 0) GTEST_SKIP() << "Inspector plugin not built: " << pluginPath;

  char tmpl[] = "/tmp/rccl_inspector_algo_XXXXXX";
  ASSERT_NE(mkdtemp(tmpl), nullptr);
  const std::filesystem::path dir(tmpl), dump = dir / "dump", plugin = dir / "inspector.so";
  // NCCL_PROFILER_PLUGIN is cut at 254 bytes; a short link loads the plugin from any build path.
  std::error_code ec;
  std::filesystem::create_symlink(pluginPath, plugin, ec);
  ASSERT_FALSE(ec) << ec.message();
  setenv("NCCL_PROFILER_PLUGIN", plugin.c_str(), 1);
  setenv("NCCL_INSPECTOR_ENABLE", "1", 1);
  setenv("NCCL_INSPECTOR_DUMP_DIR", dump.c_str(), 1);
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
  for (const auto& entry : std::filesystem::directory_iterator(dump, ec)) {
    std::ifstream in(entry.path());
    for (std::string line; std::getline(in, line);)
      if (line.find("\"coll_perf\":") != std::string::npos) collLines.push_back(line);
  }
  std::filesystem::remove_all(dir, ec);
}

} // namespace

// Isolated because the profiler plugin and its settings are latched by the first
// communicator a process creates.
TEST(InspectorCollAlgo, CollPerfCarriesAlgoProto) {
  RUN_ISOLATED_TEST("InspectorCollAlgo.CollPerfCarriesAlgoProto", []() {
    std::vector<std::string> collLines;
    runInspector(collLines);
    if (testing::Test::HasFatalFailure() || testing::Test::IsSkipped()) return;
    ASSERT_FALSE(collLines.empty()) << "no coll_perf record in the Inspector output";
    for (const std::string& line : collLines) {
      bool hasAlgo = false, hasProto = false;
      const std::string algo = jsonStrField(line, "coll_algo", hasAlgo);
      jsonStrField(line, "coll_proto", hasProto);
      EXPECT_TRUE(hasAlgo) << "coll_perf without coll_algo: " << line;
      EXPECT_TRUE(hasProto) << "coll_perf without coll_proto: " << line;
      // RCCL's core reports the algorithm of every collective it launches.
      EXPECT_FALSE(algo.empty() || algo == "unknown") << "coll_algo not reported: " << line;
    }
  });
}

TEST(InspectorCollAlgo, WritesCollRecords) {
  RUN_ISOLATED_TEST("InspectorCollAlgo.WritesCollRecords", []() {
    std::vector<std::string> collLines;
    runInspector(collLines);
    if (testing::Test::HasFatalFailure() || testing::Test::IsSkipped()) return;
    EXPECT_FALSE(collLines.empty()) << "no coll_perf record in the Inspector output";
  });
}

} // namespace RcclUnitTesting
