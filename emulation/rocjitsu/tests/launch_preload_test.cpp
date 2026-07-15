// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "launch_preload.h"

#include <cstdlib>
#include <gtest/gtest.h>
#include <optional>
#include <string>

#if defined(RJ_EXPECT_SHARED_ASAN_RUNTIME)
#include <filesystem>
#include <system_error>
#endif

namespace {

class ScopedEnvVar {
public:
  explicit ScopedEnvVar(const char *name) : name_(name) {
    if (const char *value = std::getenv(name_))
      old_value_ = value;
  }

  ~ScopedEnvVar() {
    if (old_value_) {
      setenv(name_, old_value_->c_str(), 1);
      return;
    }
    unsetenv(name_);
  }

private:
  const char *name_;
  std::optional<std::string> old_value_;
};

void expect_ld_preload_eq(const std::string &expected) {
  const char *ld_preload = std::getenv("LD_PRELOAD");
  ASSERT_NE(nullptr, ld_preload);
  EXPECT_EQ(expected, ld_preload);
}

#if defined(RJ_EXPECT_SHARED_ASAN_RUNTIME)
std::string canonical_existing_path(const std::filesystem::path &path) {
  std::error_code ec;
  std::filesystem::path canonical = std::filesystem::canonical(path, ec);
  return ec ? std::string{} : canonical.string();
}
#endif

} // namespace

TEST(LaunchPreloadTest, NoAsanPrependsInterposerBeforeExistingPreload) {
#if defined(RJ_EXPECT_SHARED_ASAN_RUNTIME)
  GTEST_SKIP() << "shared-ASan builds exercise the ASan ordering case";
#else
  ScopedEnvVar restore_ld_preload("LD_PRELOAD");
  const std::string interposer = "/tmp/librocjitsu.so";
  const std::string existing = "/tmp/libexisting.so";

  ASSERT_TRUE(rocjitsu::cli::find_loaded_asan_runtime().empty());
  ASSERT_EQ(0, setenv("LD_PRELOAD", existing.c_str(), 1));

  rocjitsu::cli::prepend_launch_preloads(interposer);

  expect_ld_preload_eq(interposer + ":" + existing);
#endif
}

TEST(LaunchPreloadTest, SharedAsanPrependsAsanBeforeInterposerAndExistingPreload) {
#if !defined(RJ_EXPECT_SHARED_ASAN_RUNTIME)
  GTEST_SKIP() << "requires a shared-ASan build";
#else
  ScopedEnvVar restore_ld_preload("LD_PRELOAD");
  const std::string interposer = "/tmp/librocjitsu.so";
  const std::string existing = "/tmp/libexisting.so";
  const std::string expected_asan = canonical_existing_path(RJ_EXPECTED_SHARED_ASAN_RUNTIME);

  ASSERT_FALSE(expected_asan.empty()) << RJ_EXPECTED_SHARED_ASAN_RUNTIME;
  EXPECT_EQ(expected_asan, rocjitsu::cli::find_loaded_asan_runtime());
  ASSERT_EQ(0, setenv("LD_PRELOAD", existing.c_str(), 1));

  rocjitsu::cli::prepend_launch_preloads(interposer);

  expect_ld_preload_eq(expected_asan + ":" + interposer + ":" + existing);
#endif
}
