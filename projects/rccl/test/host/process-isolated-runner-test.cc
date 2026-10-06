/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only tests for test/common/ProcessIsolatedTestRunner.cpp itself.

#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <optional>
#include <string>

#include <gtest/gtest.h>

#include "../common/ProcessIsolatedTestRunner.hpp"

namespace {

constexpr const char* kMarkerEnvVar = "RCCL_PIT_TEST_MARKER";

// Sets an environment variable for the scope, then restores the previous value or absence.
class ScopedEnv {
public:
  ScopedEnv(const char* name, const char* value) : name_(name) {
    if (const char* old = std::getenv(name)) {
      old_ = old;
    }
    setenv(name, value, 1);
  }
  ~ScopedEnv() {
    if (old_) {
      setenv(name_, old_->c_str(), 1);
    } else {
      unsetenv(name_);
    }
  }
  ScopedEnv(const ScopedEnv&) = delete;
  ScopedEnv& operator=(const ScopedEnv&) = delete;

private:
  const char* name_;
  std::optional<std::string> old_;
};

}  // namespace

// gtest assigns the child's single filtered test to shard 0, so a non-zero shard index must not reach the child.
TEST(ProcessIsolatedRunnerMicrotest, NonZeroShard_ChildStillRunsBody) {
  // Private 0700 dir so another local user cannot pre-plant a symlink at the marker or status-file path.
  std::string dir = ::testing::TempDir() + "rccl_pit_XXXXXX";
  // The re-exec child re-enters this body only to reach its lambda and then _exit()s, so it must not create a dir.
  if (std::getenv(RcclUnitTesting::ProcessIsolatedTestRunner::kReexecMarkerEnvVar) == nullptr) {
    ASSERT_NE(nullptr, mkdtemp(&dir[0])) << "mkdtemp failed under " << ::testing::TempDir();
  }
  const std::string marker = dir + "/marker";
  const std::string statusFile = dir + "/shard_status";
  {
    ScopedEnv totalShards("GTEST_TOTAL_SHARDS", "2");
    ScopedEnv shardIndex("GTEST_SHARD_INDEX", "1");
    // Set after this process's gtest started, so only a child that inherits it would create the file.
    ScopedEnv shardStatusFile("GTEST_SHARD_STATUS_FILE", statusFile.c_str());
    RUN_ISOLATED_TEST_WITH_ENV(
      "Pit_NonZeroShard_ChildStillRunsBody",
      []() {
        const char* path = std::getenv(kMarkerEnvVar);
        ASSERT_NE(nullptr, path);
        std::FILE* f = std::fopen(path, "w");
        ASSERT_NE(nullptr, f);
        std::fclose(f);
      },
      {{kMarkerEnvVar, marker}});
  }
  const bool markerWritten = access(marker.c_str(), F_OK) == 0;
  const bool statusFileWritten = access(statusFile.c_str(), F_OK) == 0;
  std::remove(marker.c_str());
  std::remove(statusFile.c_str());
  rmdir(dir.c_str());
  EXPECT_TRUE(markerWritten) << "isolated body never ran: the child ran 0 tests and exited 0";
  EXPECT_FALSE(statusFileWritten) << "child inherited GTEST_SHARD_STATUS_FILE and its gtest wrote the status file";
}
