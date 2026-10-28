/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Whitebox tests for in-process net plugin loading (NCCL_NET_PLUGIN=STATIC_PLUGIN)
// using plugin/net_reload_plugin.cpp.
//
// NetPluginReload — AICOMRCCL-1534 (NCCL GitHub issue #1978): a non-default
// NCCL_NET_PLUGIN must remain reloadable after the communicator that first used
// it is destroyed.
//
// NetPluginAssignFail — NCCL 2.28.7 assignment-failure cleanup (src/plugin/net.cc):
// when a plugin inits but fails assignment (e.g. incompatible UNPACK version),
// ncclNetPluginFinalize() must run before probing the next plugin; netName mismatch
// must skip init entirely.
//
// NetPluginInitFail — AICOMRCCL-1891 (NCCL 2.29.7 net.cc fix): ncclNetPluginInit()
// used to call finalize() unconditionally on its failure path, so a plugin whose
// init() failed was finalized without ever having been initialized. For a v10 (or
// older) plugin that crashes outright: the compat layer in src/plugin/net/net_v10.cc
// only fills ncclNet.finalize once ncclNet_v10->init() has succeeded, so the call
// went through a null function pointer. The guard must suppress finalize() after a
// failed init() while still running it when init() succeeded and devices() failed.
//
// NetPluginSharedNic — NCCL 2.32.3 "topo: create separate XML nodes for network backends"
// (AICOMRCCL-3378): ncclTopoPopulateNics() gives GIN and RMA devices their own <gin> and <rma>
// nodes instead of gin="1"/rma="1" on the <net> node of the same NIC name, so each backend keeps
// its own "dev" index. The GIN/RMA stub's only NIC is net device 1 in shared_nic mode, and
// NCCL_TOPO_DUMP_FILE shows the node and index each backend got for it.

#include <gtest/gtest.h>
#include <rccl/rccl.h>
#include <hip/hip_runtime.h>

#include <unistd.h>

#include <cstdlib>
#include <fstream>
#include <map>
#include <string>

#include "common/ProcessIsolatedTestRunner.hpp"

namespace RcclUnitTesting {
namespace {

constexpr int kNumCommCycles = 2;

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

// Returns the reason this host cannot run the test, or "" when it can.
// GTEST_SKIP() must be issued by the caller: it expands to a bare return and
// would otherwise only leave this helper, letting the test body run on.
std::string gpuSkipReason() {
  int deviceCount = 0;
  if (hipGetDeviceCount(&deviceCount) != hipSuccess || deviceCount < 1)
    return "requires at least one GPU";
  return "";
}

void initAndDestroyComm() {
  ncclUniqueId id;
  ASSERT_EQ(ncclGetUniqueId(&id), ncclSuccess);

  ncclComm_t comm = nullptr;
  ASSERT_EQ(ncclCommInitRank(&comm, 1, id, 0), ncclSuccess);
  ASSERT_EQ(ncclCommDestroy(comm), ncclSuccess);
}

// The GIN/RMA stub's device name in plugin/net_reload_plugin.cpp.
const char* const kSharedNic = "RmaReloadStub";

// Value of attribute `key` in a dumped XML element, or "" when it has none.
std::string xmlAttr(const std::string& element, const std::string& key) {
  const std::string needle = " " + key + "=\"";
  size_t start = element.find(needle);
  if (start == std::string::npos) return "";
  start += needle.size();
  return element.substr(start, element.find('"', start) - start);
}

// Inits and destroys a single-rank comm whose NET, RMA and GIN backends all list kSharedNic, with
// NCCL_TOPO_DUMP_FILE set, and returns the dumped elements named kSharedNic by tag ("net", ...).
void dumpSharedNic(std::map<std::string, std::string>& nodes) {
  ScopedTempFile dumpFile("/tmp/rccl_topo_dump_XXXXXX");
  ASSERT_TRUE(dumpFile.valid()) << "failed to create topology dump file";
  ASSERT_EQ(setenv("RCCL_NET_TEST_PLUGIN_MODE", "shared_nic", 1), 0);
  ASSERT_EQ(setenv("NCCL_NET_PLUGIN", "STATIC_PLUGIN", 1), 0);
  ASSERT_EQ(unsetenv("NCCL_NET"), 0);
  // RMA comes from the stub; GIN from RCCL's GIN proxy, which runs over the RMA plugin.
  ASSERT_EQ(setenv("NCCL_RMA_PLUGIN", "STATIC_PLUGIN", 1), 0);
  ASSERT_EQ(setenv("NCCL_GIN_PLUGIN", "none", 1), 0);
  ASSERT_EQ(setenv("NCCL_GIN_ENABLE", "1", 1), 0);
  // Keep the two net devices separate: NIC fusion would try to merge them through makeVDevice.
  ASSERT_EQ(setenv("NCCL_NET_MERGE_LEVEL", "LOC", 1), 0);
  ASSERT_EQ(setenv("NCCL_TOPO_DUMP_FILE", dumpFile.path().c_str(), 1), 0);
  ASSERT_NO_FATAL_FAILURE(initAndDestroyComm());

  std::ifstream f(dumpFile.path());
  std::string line;
  while (std::getline(f, line)) {
    if (xmlAttr(line, "name") != kSharedNic) continue;
    size_t tag = line.find('<') + 1;
    nodes[line.substr(tag, line.find(' ', tag) - tag)] = line.substr(tag - 1);
  }
}

// All dumped elements, for failure messages.
std::string describe(const std::map<std::string, std::string>& nodes) {
  std::string all;
  for (const auto& node : nodes) all += "\n  " + node.second;
  return all.empty() ? " (none)" : all;
}

} // namespace

TEST(NetPluginReload, CustomPluginReloadsAfterCommDestroy) {
  RUN_ISOLATED_TEST("NetPluginReload.CustomPluginReloadsAfterCommDestroy", []() {
    if (auto reason = gpuSkipReason(); !reason.empty()) GTEST_SKIP() << reason;
    ASSERT_EQ(hipSetDevice(0), hipSuccess);

    ScopedTempFile counterFile("/tmp/rccl_net_reload_XXXXXX");
    ASSERT_TRUE(counterFile.valid()) << "failed to create counter file";

    ASSERT_EQ(setenv("RCCL_NET_RELOAD_COUNTER_FILE", counterFile.path().c_str(), 1), 0);
    // ncclNetPlugin_v12 is exported from this binary (see test/CMakeLists.txt).
    ASSERT_EQ(setenv("NCCL_NET_PLUGIN", "STATIC_PLUGIN", 1), 0);

    for (int cycle = 0; cycle < kNumCommCycles; ++cycle) {
      ncclUniqueId id;
      ASSERT_EQ(ncclGetUniqueId(&id), ncclSuccess);

      ncclComm_t comm = nullptr;
      ASSERT_EQ(ncclCommInitRank(&comm, 1, id, 0), ncclSuccess)
          << "comm init cycle " << cycle << " failed";

      ASSERT_EQ(ncclCommDestroy(comm), ncclSuccess);
    }

    const int loads = countLines(counterFile.path());
    EXPECT_EQ(loads, kNumCommCycles)
        << "custom net plugin init count = " << loads << " (expected "
        << kNumCommCycles << "; fewer means the plugin was not reloaded after destroy)";
  });
}

TEST(NetPluginAssignFail, FinalizesOnFailedAssignment) {
  RUN_ISOLATED_TEST("NetPluginAssignFail.FinalizesOnFailedAssignment", []() {
    if (auto reason = gpuSkipReason(); !reason.empty()) GTEST_SKIP() << reason;
    ASSERT_EQ(hipSetDevice(0), hipSuccess);

    ScopedTempFile initFile("/tmp/rccl_net_assign_init_XXXXXX");
    ScopedTempFile finalizeFile("/tmp/rccl_net_assign_fin_XXXXXX");
    ASSERT_TRUE(initFile.valid()) << "failed to create init counter file";
    ASSERT_TRUE(finalizeFile.valid()) << "failed to create finalize counter file";

    ASSERT_EQ(setenv("RCCL_NET_TEST_PLUGIN_MODE", "assign_fail", 1), 0);
    ASSERT_EQ(setenv("RCCL_NET_ASSIGN_FAIL_INIT_FILE", initFile.path().c_str(), 1), 0);
    ASSERT_EQ(setenv("RCCL_NET_ASSIGN_FAIL_FINALIZE_FILE", finalizeFile.path().c_str(), 1), 0);
    ASSERT_EQ(setenv("NCCL_NET_PLUGIN", "STATIC_PLUGIN", 1), 0);

    initAndDestroyComm();

    EXPECT_EQ(countLines(initFile.path()), 1)
        << "external plugin init must run once before assignment failure";
    EXPECT_EQ(countLines(finalizeFile.path()), 1)
        << "ncclNetPluginFinalize() must run when assignment fails";
  });
}

TEST(NetPluginAssignFail, NetNameMismatchSkipsInit) {
  RUN_ISOLATED_TEST("NetPluginAssignFail.NetNameMismatchSkipsInit", []() {
    if (auto reason = gpuSkipReason(); !reason.empty()) GTEST_SKIP() << reason;
    ASSERT_EQ(hipSetDevice(0), hipSuccess);

    ScopedTempFile initFile("/tmp/rccl_net_assign_init_XXXXXX");
    ScopedTempFile finalizeFile("/tmp/rccl_net_assign_fin_XXXXXX");
    ASSERT_TRUE(initFile.valid()) << "failed to create init counter file";
    ASSERT_TRUE(finalizeFile.valid()) << "failed to create finalize counter file";

    ASSERT_EQ(setenv("RCCL_NET_TEST_PLUGIN_MODE", "assign_fail", 1), 0);
    ASSERT_EQ(setenv("RCCL_NET_ASSIGN_FAIL_INIT_FILE", initFile.path().c_str(), 1), 0);
    ASSERT_EQ(setenv("RCCL_NET_ASSIGN_FAIL_FINALIZE_FILE", finalizeFile.path().c_str(), 1), 0);
    ASSERT_EQ(setenv("NCCL_NET_PLUGIN", "STATIC_PLUGIN", 1), 0);
    ASSERT_EQ(setenv("NCCL_IB_DISABLE", "1", 1), 0);

    ncclUniqueId id;
    ASSERT_EQ(ncclGetUniqueId(&id), ncclSuccess);

    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    config.netName = const_cast<char*>("Socket");

    ncclComm_t comm = nullptr;
    ASSERT_EQ(ncclCommInitRankConfig(&comm, 1, id, 0, &config), ncclSuccess);
    ASSERT_EQ(ncclCommDestroy(comm), ncclSuccess);

    EXPECT_EQ(countLines(initFile.path()), 0)
        << "ReloadTest must not be inited when netName requests Socket";
    EXPECT_EQ(countLines(finalizeFile.path()), 0)
        << "ReloadTest must not be finalized when it was never inited";
  });
}

TEST(NetPluginAssignFail, SurvivesMultipleCommCycles) {
  RUN_ISOLATED_TEST("NetPluginAssignFail.SurvivesMultipleCommCycles", []() {
    if (auto reason = gpuSkipReason(); !reason.empty()) GTEST_SKIP() << reason;
    ASSERT_EQ(hipSetDevice(0), hipSuccess);

    ScopedTempFile initFile("/tmp/rccl_net_assign_init_XXXXXX");
    ScopedTempFile finalizeFile("/tmp/rccl_net_assign_fin_XXXXXX");
    ASSERT_TRUE(initFile.valid());
    ASSERT_TRUE(finalizeFile.valid());

    ASSERT_EQ(setenv("RCCL_NET_TEST_PLUGIN_MODE", "assign_fail", 1), 0);
    ASSERT_EQ(setenv("RCCL_NET_ASSIGN_FAIL_INIT_FILE", initFile.path().c_str(), 1), 0);
    ASSERT_EQ(setenv("RCCL_NET_ASSIGN_FAIL_FINALIZE_FILE", finalizeFile.path().c_str(), 1), 0);
    ASSERT_EQ(setenv("NCCL_NET_PLUGIN", "STATIC_PLUGIN", 1), 0);

    for (int cycle = 0; cycle < kNumCommCycles; ++cycle) initAndDestroyComm();

    EXPECT_EQ(countLines(initFile.path()), kNumCommCycles)
        << "each comm cycle must init the external plugin once";
    EXPECT_EQ(countLines(finalizeFile.path()), kNumCommCycles)
        << "each failed assignment must finalize the external plugin once";
  });
}

TEST(NetPluginInitFail, DoesNotFinalizeAfterFailedInit) {
  RUN_ISOLATED_TEST("NetPluginInitFail.DoesNotFinalizeAfterFailedInit", []() {
    if (auto reason = gpuSkipReason(); !reason.empty()) GTEST_SKIP() << reason;
    ASSERT_EQ(hipSetDevice(0), hipSuccess);

    ScopedTempFile initFile("/tmp/rccl_net_init_fail_init_XXXXXX");
    ScopedTempFile finalizeFile("/tmp/rccl_net_init_fail_fin_XXXXXX");
    ASSERT_TRUE(initFile.valid()) << "failed to create init counter file";
    ASSERT_TRUE(finalizeFile.valid()) << "failed to create finalize counter file";

    ASSERT_EQ(setenv("RCCL_NET_TEST_PLUGIN_MODE", "init_fail", 1), 0);
    ASSERT_EQ(setenv("RCCL_NET_TEST_INIT_FILE", initFile.path().c_str(), 1), 0);
    ASSERT_EQ(setenv("RCCL_NET_TEST_FINALIZE_FILE", finalizeFile.path().c_str(), 1), 0);
    ASSERT_EQ(setenv("NCCL_NET_PLUGIN", "STATIC_PLUGIN", 1), 0);

    // The comm must still come up: a plugin that fails init is disabled, and RCCL
    // falls back to the internal IB/Socket plugins.
    initAndDestroyComm();

    EXPECT_EQ(countLines(initFile.path()), 1)
        << "external plugin init must be attempted once";
    EXPECT_EQ(countLines(finalizeFile.path()), 0)
        << "finalize() must not run for an init() that failed";
  });
}

TEST(NetPluginInitFail, FailedPluginIsNotRetriedOrFinalized) {
  RUN_ISOLATED_TEST("NetPluginInitFail.FailedPluginIsNotRetriedOrFinalized", []() {
    if (auto reason = gpuSkipReason(); !reason.empty()) GTEST_SKIP() << reason;
    ASSERT_EQ(hipSetDevice(0), hipSuccess);

    ScopedTempFile initFile("/tmp/rccl_net_init_fail_init_XXXXXX");
    ScopedTempFile finalizeFile("/tmp/rccl_net_init_fail_fin_XXXXXX");
    ASSERT_TRUE(initFile.valid());
    ASSERT_TRUE(finalizeFile.valid());

    ASSERT_EQ(setenv("RCCL_NET_TEST_PLUGIN_MODE", "init_fail", 1), 0);
    ASSERT_EQ(setenv("RCCL_NET_TEST_INIT_FILE", initFile.path().c_str(), 1), 0);
    ASSERT_EQ(setenv("RCCL_NET_TEST_FINALIZE_FILE", finalizeFile.path().c_str(), 1), 0);
    ASSERT_EQ(setenv("NCCL_NET_PLUGIN", "STATIC_PLUGIN", 1), 0);

    for (int cycle = 0; cycle < kNumCommCycles; ++cycle) initAndDestroyComm();

    // netPluginLibs[] is process-global, so the failure marks the plugin disabled
    // for good: later comms skip it instead of re-running init.
    EXPECT_EQ(countLines(initFile.path()), 1)
        << "a plugin disabled by a failed init must not be re-inited by later comms";
    EXPECT_EQ(countLines(finalizeFile.path()), 0)
        << "no comm cycle may finalize a plugin that was never initialized";
  });
}

TEST(NetPluginInitFail, FinalizesWhenDevicesFailsAfterInit) {
  RUN_ISOLATED_TEST("NetPluginInitFail.FinalizesWhenDevicesFailsAfterInit", []() {
    if (auto reason = gpuSkipReason(); !reason.empty()) GTEST_SKIP() << reason;
    ASSERT_EQ(hipSetDevice(0), hipSuccess);

    ScopedTempFile initFile("/tmp/rccl_net_dev_fail_init_XXXXXX");
    ScopedTempFile finalizeFile("/tmp/rccl_net_dev_fail_fin_XXXXXX");
    ASSERT_TRUE(initFile.valid());
    ASSERT_TRUE(finalizeFile.valid());

    ASSERT_EQ(setenv("RCCL_NET_TEST_PLUGIN_MODE", "devices_fail", 1), 0);
    ASSERT_EQ(setenv("RCCL_NET_TEST_INIT_FILE", initFile.path().c_str(), 1), 0);
    ASSERT_EQ(setenv("RCCL_NET_TEST_FINALIZE_FILE", finalizeFile.path().c_str(), 1), 0);
    ASSERT_EQ(setenv("NCCL_NET_PLUGIN", "STATIC_PLUGIN", 1), 0);

    initAndDestroyComm();

    // Same failure path as above, but init() did succeed, so the context exists and
    // has to be released. This is what keeps the guard from becoming a leak.
    EXPECT_EQ(countLines(initFile.path()), 1)
        << "external plugin init must run before devices() is probed";
    EXPECT_EQ(countLines(finalizeFile.path()), 1)
        << "finalize() must release the context when devices() fails after a good init()";
  });
}

TEST(NetPluginSharedNic, DumpMarksNicForEveryBackend) {
  RUN_ISOLATED_TEST("NetPluginSharedNic.DumpMarksNicForEveryBackend", []() {
    if (auto reason = gpuSkipReason(); !reason.empty()) GTEST_SKIP() << reason;
    ASSERT_EQ(hipSetDevice(0), hipSuccess);

    std::map<std::string, std::string> nodes;
    ASSERT_NO_FATAL_FAILURE(dumpSharedNic(nodes));
    // Holds whether the backends share one <net> node or each has a node of its own.
    for (const char* backend : {"net", "gin", "rma"}) {
      bool marked = false;
      for (const auto& node : nodes) marked |= xmlAttr(node.second, backend) == "1";
      EXPECT_TRUE(marked) << "the " << backend << " plugin did not import " << kSharedNic
                          << " into the topology; dumped:" << describe(nodes);
    }
  });
}

TEST(NetPluginSharedNic, GinAndRmaKeepTheirOwnDevIndex) {
  RUN_ISOLATED_TEST("NetPluginSharedNic.GinAndRmaKeepTheirOwnDevIndex", []() {
    if (auto reason = gpuSkipReason(); !reason.empty()) GTEST_SKIP() << reason;
    ASSERT_EQ(hipSetDevice(0), hipSuccess);

    std::map<std::string, std::string> nodes;
    ASSERT_NO_FATAL_FAILURE(dumpSharedNic(nodes));
    // kSharedNic is GIN/RMA device 0 but net device 1. A <net> node shared by all three backends
    // has one "dev", which the net plugin, imported last, sets to 1.
    const std::pair<const char*, const char*> devByTag[] = {{"gin", "0"}, {"rma", "0"}, {"net", "1"}};
    for (const auto& [tag, dev] : devByTag) {
      auto it = nodes.find(tag);
      EXPECT_EQ(it == nodes.end() ? "" : xmlAttr(it->second, "dev"), dev)
          << "dev of the <" << tag << "> node for " << kSharedNic << " (\"\": no such node); dumped:" << describe(nodes);
    }
  });
}

} // namespace RcclUnitTesting
