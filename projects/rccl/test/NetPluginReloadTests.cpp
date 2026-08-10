/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Regression test for AICOMRCCL-1534 (NCCL GitHub issue #1978): a non-default
// NCCL_NET_PLUGIN must remain reloadable after the communicator that first used
// it is destroyed. Before the fix, ncclNetPluginUnload memset() the plugin
// struct and wiped the plugin name parsed once from NCCL_NET_PLUGIN, so the
// second communicator silently fell back to the default net plugin.
//
// The test uses a tiny net plugin (plugin/net_reload_plugin.cpp) compiled into
// this test binary that records every real init() call, then creates and
// destroys a communicator twice in one process. The plugin must be initialised
// twice (once per comm).

#include <gtest/gtest.h>
#include <rccl/rccl.h>
#include <hip/hip_runtime.h>

#include <unistd.h>

#include <cstdlib>
#include <fstream>
#include <string>

#include "common/ProcessIsolatedTestRunner.hpp"

namespace RcclUnitTesting {
namespace {

// One create/destroy cycle per iteration; the plugin must be (re)initialised
// once per cycle, so the expected init count equals this.
constexpr int kNumCommCycles = 2;

// Creates a unique file from an mkstemp template and unlinks it on scope exit,
// so an early-returning ASSERT_ cannot leak it into /tmp
class ScopedTempFile {
 public:
  explicit ScopedTempFile(const char* pathTemplate) : path_(pathTemplate) {
    int fd = mkstemp(path_.data());
    if (fd >= 0) {
      valid_ = true;
      close(fd);
    }
  }

  ~ScopedTempFile() {
    if (valid_) unlink(path_.c_str());
  }

  ScopedTempFile(const ScopedTempFile&) = delete;
  ScopedTempFile& operator=(const ScopedTempFile&) = delete;

  bool valid() const { return valid_; }

  const std::string& path() const { return path_; }

 private:
  std::string path_;
  bool valid_ = false;
};

int countLines(const std::string& path) {
  std::ifstream f(path);
  int count = 0;
  std::string line;
  while (std::getline(f, line))
    if (!line.empty()) ++count;
  return count;
}

} // namespace

TEST(NetPluginReload, CustomPluginReloadsAfterCommDestroy) {
  RUN_ISOLATED_TEST("NetPluginReload.CustomPluginReloadsAfterCommDestroy", []() {
    int deviceCount = 0;
    if (hipGetDeviceCount(&deviceCount) != hipSuccess || deviceCount < 1)
      GTEST_SKIP() << "requires at least one GPU";

    ScopedTempFile counterFile("/tmp/rccl_net_reload_XXXXXX");
    ASSERT_TRUE(counterFile.valid()) << "failed to create counter file";

    ASSERT_EQ(setenv("RCCL_NET_RELOAD_COUNTER_FILE", counterFile.path().c_str(), 1), 0);

    // The test plugin is compiled into this binary and its ncclNetPlugin_v10
    // symbol is exported (see test/CMakeLists.txt). NCCL_NET_PLUGIN=STATIC_PLUGIN
    // makes the loader dlopen(NULL) and resolve that in-process symbol
    ASSERT_EQ(setenv("NCCL_NET_PLUGIN", "STATIC_PLUGIN", 1), 0);

    ASSERT_EQ(hipSetDevice(0), hipSuccess);

    for (int cycle = 0; cycle < kNumCommCycles; ++cycle) {
      ncclUniqueId id;
      ASSERT_EQ(ncclGetUniqueId(&id), ncclSuccess);

      ncclComm_t comm = nullptr;
      ASSERT_EQ(ncclCommInitRank(&comm, 1, id, 0), ncclSuccess)
        << "comm init cycle " << cycle << " failed";

      ASSERT_EQ(ncclCommDestroy(comm), ncclSuccess);
    }

    int loads = countLines(counterFile.path());

    // Fixed: plugin reloaded for every cycle -> kNumCommCycles init calls.
    // Pre-fix: name wiped on unload, later cycles use the default plugin -> 1.
    EXPECT_EQ(loads, kNumCommCycles)
        << "custom net plugin init count = " << loads << " (expected "
        << kNumCommCycles << "; fewer means the plugin was not reloaded after destroy)";
  });
}

} // namespace RcclUnitTesting
