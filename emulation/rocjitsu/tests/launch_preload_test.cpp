// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "launch_preload.h"

#include <cstdlib>
#include <filesystem>
#include <gtest/gtest.h>
#include <optional>
#include <string>
#include <system_error>

#include <sys/wait.h>
#include <unistd.h>

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

#if defined(RJ_EXPECT_SHARED_ASAN_RUNTIME) || defined(RJ_EXPECT_SHARED_TSAN_RUNTIME)
std::string canonical_existing_path(const std::filesystem::path &path) {
  std::error_code ec;
  std::filesystem::path canonical = std::filesystem::canonical(path, ec);
  return ec ? std::string{} : canonical.string();
}
#endif

#if !defined(RJ_EXPECT_SHARED_ASAN_RUNTIME) && !defined(RJ_EXPECT_SHARED_TSAN_RUNTIME)
void expect_no_sanitizer_preload_order() {
  ScopedEnvVar restore_ld_preload("LD_PRELOAD");
  const std::string interposer = "/tmp/librocjitsu.so";
  const std::string existing = "/tmp/libexisting.so";

  ASSERT_TRUE(rocjitsu::cli::find_loaded_asan_runtime().empty());
  ASSERT_TRUE(rocjitsu::cli::find_loaded_tsan_runtime().empty());
  ASSERT_EQ(0, setenv("LD_PRELOAD", existing.c_str(), 1));

  rocjitsu::cli::prepend_launch_preloads(interposer);

  expect_ld_preload_eq(interposer + ":" + existing);
}
#endif

} // namespace

TEST(LaunchPreloadTest, NoAsanPrependsInterposerBeforeExistingPreload) {
#if defined(RJ_EXPECT_SHARED_ASAN_RUNTIME) || defined(RJ_EXPECT_SHARED_TSAN_RUNTIME)
  GTEST_SKIP() << "shared sanitizer builds exercise sanitizer ordering cases";
#else
  expect_no_sanitizer_preload_order();
#endif
}

TEST(LaunchPreloadTest, SanitizerNamedExecutableAliasDoesNotPreloadExecutable) {
#if defined(RJ_EXPECT_SHARED_ASAN_RUNTIME) || defined(RJ_EXPECT_SHARED_TSAN_RUNTIME)
  GTEST_SKIP() << "shared sanitizer builds exercise sanitizer ordering cases";
#else
  if (std::getenv("RJ_LAUNCH_PRELOAD_ALIAS_CHILD")) {
    expect_no_sanitizer_preload_order();
    return;
  }

  std::error_code ec;
  const std::filesystem::path exe = std::filesystem::read_symlink("/proc/self/exe", ec);
  ASSERT_FALSE(ec) << ec.message();

  const std::filesystem::path temp_dir =
      std::filesystem::temp_directory_path() /
      ("rocjitsu_launch_preload_test_" + std::to_string(getpid()));
  ASSERT_TRUE(std::filesystem::create_directories(temp_dir, ec) ||
              std::filesystem::exists(temp_dir))
      << ec.message();

  static constexpr const char *alias_names[] = {"launch-asan-preload-test",
                                                "launch-tsan-preload-test"};
  for (const char *alias_name : alias_names) {
    const std::filesystem::path alias = temp_dir / alias_name;
    std::filesystem::remove(alias, ec);
    ec.clear();
    std::filesystem::create_symlink(exe, alias, ec);
    ASSERT_FALSE(ec) << ec.message();

    pid_t pid = fork();
    ASSERT_NE(-1, pid);
    if (pid == 0) {
      setenv("RJ_LAUNCH_PRELOAD_ALIAS_CHILD", "1", 1);
      const std::string alias_string = alias.string();
      execl(alias_string.c_str(), alias_string.c_str(),
            "--gtest_filter=LaunchPreloadTest."
            "SanitizerNamedExecutableAliasDoesNotPreloadExecutable",
            nullptr);
      _exit(127);
    }

    int status = 0;
    ASSERT_EQ(pid, waitpid(pid, &status, 0));
    ASSERT_TRUE(WIFEXITED(status));
    EXPECT_EQ(0, WEXITSTATUS(status));
  }
  std::filesystem::remove_all(temp_dir, ec);
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

TEST(LaunchPreloadTest, SharedTsanPrependsTsanBeforeInterposerAndExistingPreload) {
#if !defined(RJ_EXPECT_SHARED_TSAN_RUNTIME)
  GTEST_SKIP() << "requires a shared-TSan build";
#else
  ScopedEnvVar restore_ld_preload("LD_PRELOAD");
  const std::string interposer = "/tmp/librocjitsu.so";
  const std::string existing = "/tmp/libexisting.so";
  const std::string expected_tsan = canonical_existing_path(RJ_EXPECTED_SHARED_TSAN_RUNTIME);

  ASSERT_FALSE(expected_tsan.empty()) << RJ_EXPECTED_SHARED_TSAN_RUNTIME;
  EXPECT_EQ(expected_tsan, rocjitsu::cli::find_loaded_tsan_runtime());
  ASSERT_EQ(0, setenv("LD_PRELOAD", existing.c_str(), 1));

  rocjitsu::cli::prepend_launch_preloads(interposer);

  expect_ld_preload_eq(expected_tsan + ":" + interposer + ":" + existing);
#endif
}
