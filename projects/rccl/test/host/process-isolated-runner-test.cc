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
  const std::string marker = ::testing::TempDir() + "rccl_pit_shard_marker_" + std::to_string(getpid());
  std::remove(marker.c_str());
  {
    ScopedEnv totalShards("GTEST_TOTAL_SHARDS", "2");
    ScopedEnv shardIndex("GTEST_SHARD_INDEX", "1");
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
  EXPECT_EQ(0, access(marker.c_str(), F_OK)) << "isolated body never ran: the child ran 0 tests and exited 0";
  std::remove(marker.c_str());
}
