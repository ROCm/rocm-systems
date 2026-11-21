/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// GinRmaPluginVersion — NCCL 2.32.6 bumped the GIN plugin API to v15 and the RMA
// plugin API to v17 (src/plugin/gin.cc, src/plugin/rma.cc). An external plugin
// that exports only ncclGinPlugin_v15 / ncclRmaPlugin_v17 must be loaded and
// initialized. plugin/gin_rma_latest_plugin.cpp is such a plugin; it reports no
// devices, so RCCL runs init() then finalize() and falls back to its own backends.

#include <gtest/gtest.h>
#include <rccl/rccl.h>
#include <hip/hip_runtime.h>

#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

#include "common/ProcessIsolatedTestRunner.hpp"

namespace RcclUnitTesting {
namespace {

// MAX_STR_LEN in src/plugin/plugin_open.cc: a longer plugin path is truncated and never loads.
constexpr size_t kPluginPathLimit = 255;

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

// The plugin is built and installed beside the test binary.
std::string latestPluginPath() {
  std::error_code ec;
  std::filesystem::path exe = std::filesystem::read_symlink("/proc/self/exe", ec);
  if (ec) return "";
  return (exe.parent_path() / RCCL_TEST_GIN_RMA_LATEST_PLUGIN_NAME).string();
}

// Loads the plugin as the external GIN ("GIN") or RMA ("RMA") plugin of a
// single-rank communicator and checks that its init() and finalize() ran once.
void expectLatestPluginInitialized(const char* kind) {
  int deviceCount = 0;
  if (hipGetDeviceCount(&deviceCount) != hipSuccess || deviceCount < 1)
    GTEST_SKIP() << "requires at least one GPU";
  ASSERT_EQ(hipSetDevice(0), hipSuccess);

  const std::string pluginPath = latestPluginPath();
  ASSERT_FALSE(pluginPath.empty()) << "could not resolve the test binary's own directory";
  ASSERT_EQ(access(pluginPath.c_str(), R_OK), 0) << "test plugin missing: " << pluginPath;
  if (pluginPath.size() >= kPluginPathLimit)
    GTEST_SKIP() << "plugin path exceeds the plugin path limit: " << pluginPath;

  ScopedTempFile initFile("/tmp/rccl_latest_plugin_init_XXXXXX");
  ScopedTempFile finalizeFile("/tmp/rccl_latest_plugin_fin_XXXXXX");
  ASSERT_TRUE(initFile.valid());
  ASSERT_TRUE(finalizeFile.valid());

  const std::string k(kind);
  ASSERT_EQ(setenv(("RCCL_TEST_" + k + "_LATEST_INIT_FILE").c_str(), initFile.path().c_str(), 1), 0);
  ASSERT_EQ(setenv(("RCCL_TEST_" + k + "_LATEST_FINALIZE_FILE").c_str(), finalizeFile.path().c_str(), 1), 0);
  ASSERT_EQ(unsetenv("NCCL_GIN_PLUGIN"), 0);
  ASSERT_EQ(unsetenv("NCCL_RMA_PLUGIN"), 0);
  ASSERT_EQ(setenv(("NCCL_" + k + "_PLUGIN").c_str(), pluginPath.c_str(), 1), 0);
  ASSERT_EQ(setenv("NCCL_GIN_ENABLE", "1", 1), 0);

  ncclUniqueId id;
  ASSERT_EQ(ncclGetUniqueId(&id), ncclSuccess);
  ncclComm_t comm = nullptr;
  ASSERT_EQ(ncclCommInitRank(&comm, 1, id, 0), ncclSuccess);
  ASSERT_EQ(ncclCommDestroy(comm), ncclSuccess);

  EXPECT_EQ(countLines(initFile.path()), 1)
      << k << " plugin exporting only the newest API symbol was not loaded and initialized";
  EXPECT_EQ(countLines(finalizeFile.path()), 1)
      << k << " plugin reporting no devices must be finalized after init";
}

}  // namespace

TEST(GinRmaPluginVersion, LoadsGinPluginV15) {
  RUN_ISOLATED_TEST("GinRmaPluginVersion.LoadsGinPluginV15", []() { expectLatestPluginInitialized("GIN"); });
}

TEST(GinRmaPluginVersion, LoadsRmaPluginV17) {
  RUN_ISOLATED_TEST("GinRmaPluginVersion.LoadsRmaPluginV17", []() { expectLatestPluginInitialized("RMA"); });
}

// Control: the same comm init still reaches an external RMA plugin through the
// v13 loader, using plugin/net_reload_plugin.cpp's in-binary ncclGinPlugin_v13 stub.
TEST(GinRmaPluginVersion, LoadsLegacyRmaPluginV13) {
  RUN_ISOLATED_TEST("GinRmaPluginVersion.LoadsLegacyRmaPluginV13", []() {
    int deviceCount = 0;
    if (hipGetDeviceCount(&deviceCount) != hipSuccess || deviceCount < 1)
      GTEST_SKIP() << "requires at least one GPU";
    ASSERT_EQ(hipSetDevice(0), hipSuccess);

    ScopedTempFile initFile("/tmp/rccl_legacy_rma_init_XXXXXX");
    ScopedTempFile finalizeFile("/tmp/rccl_legacy_rma_fin_XXXXXX");
    ASSERT_TRUE(initFile.valid());
    ASSERT_TRUE(finalizeFile.valid());
    ASSERT_EQ(setenv("RCCL_GIN_TEST_PLUGIN_MODE", "devices_zero", 1), 0);
    ASSERT_EQ(setenv("RCCL_GIN_TEST_INIT_FILE", initFile.path().c_str(), 1), 0);
    ASSERT_EQ(setenv("RCCL_GIN_TEST_FINALIZE_FILE", finalizeFile.path().c_str(), 1), 0);
    ASSERT_EQ(unsetenv("NCCL_GIN_PLUGIN"), 0);
    ASSERT_EQ(setenv("NCCL_RMA_PLUGIN", "STATIC_PLUGIN", 1), 0);

    ncclUniqueId id;
    ASSERT_EQ(ncclGetUniqueId(&id), ncclSuccess);
    ncclComm_t comm = nullptr;
    ASSERT_EQ(ncclCommInitRank(&comm, 1, id, 0), ncclSuccess);
    ASSERT_EQ(ncclCommDestroy(comm), ncclSuccess);

    EXPECT_EQ(countLines(initFile.path()), 1) << "v13 RMA plugin was not loaded and initialized";
    EXPECT_EQ(countLines(finalizeFile.path()), 1) << "v13 RMA plugin reporting no devices must be finalized";
  });
}

}  // namespace RcclUnitTesting
