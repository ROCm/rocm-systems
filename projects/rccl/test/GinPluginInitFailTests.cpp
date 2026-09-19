/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Whitebox tests for in-process GIN plugin loading (NCCL_GIN_PLUGIN=STATIC_PLUGIN)
// using plugin/net_reload_plugin.cpp's ncclGinPlugin_v13 stub.
//
// GinPluginInitFail — NCCL 2.30.7 / NVIDIA/nccl#2179: ncclGinPluginInit()
// used to drop the per-comm context allocated by init() when the following
// devices() probe failed or reported ndev <= 0. The plugin was disabled, but
// finalize() was never called (Valgrind: 8 bytes definitely lost / rank from
// ncclGinIbInitType). The guard added in that fix must still run finalize()
// after a successful init, and must not run it when init() itself failed.

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

void setGinStubEnv(const char* mode, const std::string& initPath, const std::string& finalizePath) {
  ASSERT_EQ(setenv("RCCL_GIN_TEST_PLUGIN_MODE", mode, 1), 0);
  ASSERT_EQ(setenv("RCCL_GIN_TEST_INIT_FILE", initPath.c_str(), 1), 0);
  ASSERT_EQ(setenv("RCCL_GIN_TEST_FINALIZE_FILE", finalizePath.c_str(), 1), 0);
  ASSERT_EQ(setenv("NCCL_GIN_PLUGIN", "STATIC_PLUGIN", 1), 0);
  ASSERT_EQ(setenv("NCCL_GIN_ENABLE", "1", 1), 0);
}

}  // namespace

TEST(GinPluginInitFail, FinalizesWhenDevicesFailsAfterInit) {
  RUN_ISOLATED_TEST("GinPluginInitFail.FinalizesWhenDevicesFailsAfterInit", []() {
    if (auto reason = gpuSkipReason(); !reason.empty()) GTEST_SKIP() << reason;
    ASSERT_EQ(hipSetDevice(0), hipSuccess);

    ScopedTempFile initFile("/tmp/rccl_gin_dev_fail_init_XXXXXX");
    ScopedTempFile finalizeFile("/tmp/rccl_gin_dev_fail_fin_XXXXXX");
    ASSERT_TRUE(initFile.valid());
    ASSERT_TRUE(finalizeFile.valid());

    setGinStubEnv("devices_fail", initFile.path(), finalizeFile.path());
    initAndDestroyComm();

    EXPECT_EQ(countLines(initFile.path()), 1)
        << "external GIN plugin init must run before devices() is probed";
    EXPECT_EQ(countLines(finalizeFile.path()), 1)
        << "finalize() must release the context when devices() fails after a good init()";
  });
}

TEST(GinPluginInitFail, FinalizesWhenDevicesReportsZero) {
  RUN_ISOLATED_TEST("GinPluginInitFail.FinalizesWhenDevicesReportsZero", []() {
    if (auto reason = gpuSkipReason(); !reason.empty()) GTEST_SKIP() << reason;
    ASSERT_EQ(hipSetDevice(0), hipSuccess);

    ScopedTempFile initFile("/tmp/rccl_gin_dev_zero_init_XXXXXX");
    ScopedTempFile finalizeFile("/tmp/rccl_gin_dev_zero_fin_XXXXXX");
    ASSERT_TRUE(initFile.valid());
    ASSERT_TRUE(finalizeFile.valid());

    setGinStubEnv("devices_zero", initFile.path(), finalizeFile.path());
    initAndDestroyComm();

    EXPECT_EQ(countLines(initFile.path()), 1)
        << "external GIN plugin init must run before devices() is probed";
    EXPECT_EQ(countLines(finalizeFile.path()), 1)
        << "finalize() must release the context when devices() reports ndev <= 0";
  });
}

TEST(GinPluginInitFail, DoesNotFinalizeAfterFailedInit) {
  RUN_ISOLATED_TEST("GinPluginInitFail.DoesNotFinalizeAfterFailedInit", []() {
    if (auto reason = gpuSkipReason(); !reason.empty()) GTEST_SKIP() << reason;
    ASSERT_EQ(hipSetDevice(0), hipSuccess);

    ScopedTempFile initFile("/tmp/rccl_gin_init_fail_init_XXXXXX");
    ScopedTempFile finalizeFile("/tmp/rccl_gin_init_fail_fin_XXXXXX");
    ASSERT_TRUE(initFile.valid());
    ASSERT_TRUE(finalizeFile.valid());

    setGinStubEnv("init_fail", initFile.path(), finalizeFile.path());
    initAndDestroyComm();

    EXPECT_EQ(countLines(initFile.path()), 1)
        << "external GIN plugin init must be attempted once";
    EXPECT_EQ(countLines(finalizeFile.path()), 0)
        << "finalize() must not run for an init() that failed";
  });
}

}  // namespace RcclUnitTesting
