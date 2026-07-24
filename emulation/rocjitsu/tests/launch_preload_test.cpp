// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "launch_preload.h"
#include "scoped_temp.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <gtest/gtest.h>
#include <optional>
#include <string>
#include <system_error>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

namespace {

class ScopedUnsetLdPreload {
public:
  ScopedUnsetLdPreload() {
    if (const char *value = std::getenv("LD_PRELOAD"))
      original_value_ = value;
    unset_result_ = unsetenv("LD_PRELOAD");
  }

  ~ScopedUnsetLdPreload() {
    if (original_value_)
      setenv("LD_PRELOAD", original_value_->c_str(), 1);
  }

  int unset_result() const { return unset_result_; }

private:
  std::optional<std::string> original_value_;
  int unset_result_ = -1;
};

std::vector<std::string> copy_environment(char *const *envp) {
  std::vector<std::string> entries;
  while (*envp != nullptr) {
    entries.emplace_back(*envp);
    ++envp;
  }
  EXPECT_EQ(nullptr, *envp);
  return entries;
}

void expect_ld_preload_eq(const rocjitsu::cli::LaunchEnvironment &environment,
                          const std::string &expected) {
  const char *ld_preload = environment.get("LD_PRELOAD");
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
  const std::string interposer = "/tmp/librocjitsu.so";
  const std::string existing = "/tmp/libexisting.so";
  rocjitsu::cli::LaunchEnvironment environment;
  environment.set("LD_PRELOAD", existing);

  ASSERT_TRUE(rocjitsu::cli::find_loaded_asan_runtime().empty());
  ASSERT_TRUE(rocjitsu::cli::find_loaded_tsan_runtime().empty());

  rocjitsu::cli::prepend_launch_preloads(environment, interposer);

  expect_ld_preload_eq(environment, interposer + ":" + existing);
}
#endif

} // namespace

TEST(LaunchPreloadTest, EnvpCacheRebuildsAfterSet) {
  const std::string name = "RJ_LAUNCH_ENV_CACHE_TEST_" + std::to_string(getpid());
  const std::string before = name + "=before";
  const std::string after = name + "=after";
  rocjitsu::cli::LaunchEnvironment environment;
  environment.set(name, "before");

  const std::vector<std::string> initial = copy_environment(environment.envp());
  EXPECT_NE(initial.end(), std::find(initial.begin(), initial.end(), before));

  environment.set(name, "after");

  const std::vector<std::string> rebuilt = copy_environment(environment.envp());
  EXPECT_EQ(rebuilt.end(), std::find(rebuilt.begin(), rebuilt.end(), before));
  EXPECT_NE(rebuilt.end(), std::find(rebuilt.begin(), rebuilt.end(), after));
}

TEST(LaunchPreloadTest, NoAsanPrependsInterposerBeforeExistingPreload) {
#if defined(RJ_EXPECT_SHARED_ASAN_RUNTIME) || defined(RJ_EXPECT_SHARED_TSAN_RUNTIME)
  GTEST_SKIP() << "shared sanitizer builds exercise sanitizer ordering cases";
#else
  expect_no_sanitizer_preload_order();
#endif
}

TEST(LaunchPreloadTest, UnsetLdPreloadUsesInterposerOnly) {
#if defined(RJ_EXPECT_SHARED_ASAN_RUNTIME) || defined(RJ_EXPECT_SHARED_TSAN_RUNTIME)
  GTEST_SKIP() << "shared sanitizer builds include the sanitizer runtime";
#else
  ScopedUnsetLdPreload scoped_unset;
  ASSERT_EQ(0, scoped_unset.unset_result());

  const std::string interposer = "/tmp/librocjitsu.so";
  rocjitsu::cli::LaunchEnvironment environment;
  ASSERT_EQ(nullptr, environment.get("LD_PRELOAD"));

  rocjitsu::cli::prepend_launch_preloads(environment, interposer);

  expect_ld_preload_eq(environment, interposer);
#endif
}

TEST(LaunchPreloadTest, EmptyLdPreloadUsesInterposerOnly) {
#if defined(RJ_EXPECT_SHARED_ASAN_RUNTIME) || defined(RJ_EXPECT_SHARED_TSAN_RUNTIME)
  GTEST_SKIP() << "shared sanitizer builds include the sanitizer runtime";
#else
  const std::string interposer = "/tmp/librocjitsu.so";
  rocjitsu::cli::LaunchEnvironment environment;
  environment.set("LD_PRELOAD", "");

  rocjitsu::cli::prepend_launch_preloads(environment, interposer);

  expect_ld_preload_eq(environment, interposer);
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

  const rocjitsu::test::ScopedTempDirectory temp_dir_owner("rocjitsu-launch-preload-");
  const std::filesystem::path temp_dir(temp_dir_owner.path());

  static constexpr const char *alias_names[] = {"launch-asan-preload-test",
                                                "launch-tsan-preload-test"};
  for (const char *alias_name : alias_names) {
    const std::filesystem::path alias = temp_dir / alias_name;
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
#endif
}

TEST(LaunchPreloadTest, SharedAsanPrependsAsanBeforeInterposerAndExistingPreload) {
#if !defined(RJ_EXPECT_SHARED_ASAN_RUNTIME)
  GTEST_SKIP() << "requires a shared-ASan build";
#else
  const std::string interposer = "/tmp/librocjitsu.so";
  const std::string existing = "/tmp/libexisting.so";
  const std::string expected_asan = canonical_existing_path(RJ_EXPECTED_SHARED_ASAN_RUNTIME);
  rocjitsu::cli::LaunchEnvironment environment;
  environment.set("LD_PRELOAD", existing);

  ASSERT_FALSE(expected_asan.empty()) << RJ_EXPECTED_SHARED_ASAN_RUNTIME;
  EXPECT_EQ(expected_asan, rocjitsu::cli::find_loaded_asan_runtime());

  rocjitsu::cli::prepend_launch_preloads(environment, interposer);

  expect_ld_preload_eq(environment, expected_asan + ":" + interposer + ":" + existing);
#endif
}

TEST(LaunchPreloadTest, SharedTsanPrependsTsanBeforeInterposerAndExistingPreload) {
#if !defined(RJ_EXPECT_SHARED_TSAN_RUNTIME)
  GTEST_SKIP() << "requires a shared-TSan build";
#else
  const std::string interposer = "/tmp/librocjitsu.so";
  const std::string existing = "/tmp/libexisting.so";
  const std::string expected_tsan = canonical_existing_path(RJ_EXPECTED_SHARED_TSAN_RUNTIME);
  rocjitsu::cli::LaunchEnvironment environment;
  environment.set("LD_PRELOAD", existing);

  ASSERT_FALSE(expected_tsan.empty()) << RJ_EXPECTED_SHARED_TSAN_RUNTIME;
  EXPECT_EQ(expected_tsan, rocjitsu::cli::find_loaded_tsan_runtime());

  rocjitsu::cli::prepend_launch_preloads(environment, interposer);

  expect_ld_preload_eq(environment, expected_tsan + ":" + interposer + ":" + existing);
#endif
}
