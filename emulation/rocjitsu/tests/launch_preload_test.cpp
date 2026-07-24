// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "launch_preload.h"
#include "rocm_visibility.h"
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

std::vector<rocjitsu::cli::VisibleGpu> test_gpus() {
  return {{0, 100, 90402, 0x1111111111111111ULL},
          {1, 101, 90402, 0x2222222222222222ULL},
          {2, 102, 120001, 0x3333333333333333ULL}};
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

TEST(RocmVisibilityTest, UnsetAndEmptyRocrSelectorsDiffer) {
  const auto gpus = test_gpus();
  EXPECT_EQ(3u, rocjitsu::cli::filter_rocr_visible_gpus(gpus, std::nullopt).size());
  EXPECT_TRUE(rocjitsu::cli::filter_rocr_visible_gpus(gpus, std::string_view{}).empty());
}

TEST(RocmVisibilityTest, KfdEnumerationSkipsZeroGpuIdsAndCompactsOrdinals) {
  const std::vector<rocjitsu::cli::VisibleGpu> candidates{{0, 0, 90402, 0},
                                                          {1, 101, 90402, 0x2222222222222222ULL}};
  const auto gpus = rocjitsu::cli::enumerate_kfd_gpus(candidates);

  ASSERT_EQ(1u, gpus.size());
  EXPECT_EQ(0u, gpus[0].ordinal);
  EXPECT_EQ(101u, gpus[0].gpu_id);
}

TEST(RocmVisibilityTest, NumericAndUuidSelectorsReorderDevices) {
  const auto gpus = test_gpus();
  const auto numeric = rocjitsu::cli::filter_rocr_visible_gpus(gpus, "1,0");
  const auto uuid = rocjitsu::cli::filter_rocr_visible_gpus(gpus, "gpu-2222");

  ASSERT_EQ(2u, numeric.size());
  EXPECT_EQ(101u, numeric[0].gpu_id);
  EXPECT_EQ(100u, numeric[1].gpu_id);
  ASSERT_EQ(1u, uuid.size());
  EXPECT_EQ(101u, uuid[0].gpu_id);
}

TEST(RocmVisibilityTest, AmbiguousRocrUuidTerminatesAfterValidPrefix) {
  const std::vector<rocjitsu::cli::VisibleGpu> gpus{{0, 100, 90402, 0x1111111111111111ULL},
                                                    {1, 101, 90402, 0x2222222211111111ULL},
                                                    {2, 102, 90402, 0x2222222244444444ULL}};
  const auto selected = rocjitsu::cli::filter_rocr_visible_gpus(gpus, "0,GPU-2222,1");

  ASSERT_EQ(1u, selected.size());
  EXPECT_EQ(100u, selected[0].gpu_id);
}

TEST(RocmVisibilityTest, ClientUuidUsesCaseSensitiveFirstMatch) {
  const std::vector<rocjitsu::cli::VisibleGpu> gpus{{0, 100, 90402, 0x1111111111111111ULL},
                                                    {1, 101, 90402, 0x1111111111111122ULL}};
  const auto first = rocjitsu::cli::filter_client_visible_gpus(gpus, "GPU-1111");
  const auto lowercase = rocjitsu::cli::filter_client_visible_gpus(gpus, "gpu-1111");

  ASSERT_EQ(1u, first.size());
  EXPECT_EQ(100u, first[0].gpu_id);
  EXPECT_TRUE(lowercase.empty());
}

TEST(RocmVisibilityTest, InvalidRocrTokenPreservesValidPrefix) {
  const auto selected = rocjitsu::cli::filter_rocr_visible_gpus(test_gpus(), "1,invalid,0");
  ASSERT_EQ(1u, selected.size());
  EXPECT_EQ(101u, selected[0].gpu_id);
}

TEST(RocmVisibilityTest, MalformedSelectorsTerminateSelection) {
  EXPECT_TRUE(rocjitsu::cli::filter_rocr_visible_gpus(test_gpus(), "GPU-").empty());
  EXPECT_TRUE(rocjitsu::cli::filter_client_visible_gpus(test_gpus(), "01").empty());
  EXPECT_TRUE(rocjitsu::cli::filter_client_visible_gpus(test_gpus(), "00").empty());

  const auto rocr_prefix = rocjitsu::cli::filter_rocr_visible_gpus(test_gpus(), "0,GPU-,1");
  ASSERT_EQ(1u, rocr_prefix.size());
  EXPECT_EQ(100u, rocr_prefix[0].gpu_id);

  const auto client_prefix = rocjitsu::cli::filter_client_visible_gpus(test_gpus(), "1,00,2");
  ASSERT_EQ(1u, client_prefix.size());
  EXPECT_EQ(101u, client_prefix[0].gpu_id);
}

TEST(RocmVisibilityTest, DuplicateAndReorderedSelectorsMatchRuntimeBehavior) {
  const auto rocr_duplicate = rocjitsu::cli::filter_rocr_visible_gpus(test_gpus(), "0,0,1");
  ASSERT_EQ(1u, rocr_duplicate.size());
  EXPECT_EQ(100u, rocr_duplicate[0].gpu_id);

  const auto client_duplicate = rocjitsu::cli::filter_client_visible_gpus(test_gpus(), "0,0,1");
  ASSERT_EQ(2u, client_duplicate.size());
  EXPECT_EQ(100u, client_duplicate[0].gpu_id);
  EXPECT_EQ(101u, client_duplicate[1].gpu_id);

  const auto client_reordered = rocjitsu::cli::filter_client_visible_gpus(test_gpus(), "2,0");
  ASSERT_EQ(2u, client_reordered.size());
  EXPECT_EQ(102u, client_reordered[0].gpu_id);
  EXPECT_EQ(100u, client_reordered[1].gpu_id);
}

TEST(RocmVisibilityTest, NegativeRocrOrdinalPreservesValidPrefix) {
  const auto selected = rocjitsu::cli::filter_rocr_visible_gpus(test_gpus(), "0,-1,1");
  ASSERT_EQ(1u, selected.size());
  EXPECT_EQ(100u, selected[0].gpu_id);
}

TEST(RocmVisibilityTest, HipSelectorUsesPostRocrOrdinals) {
  const auto selected =
      rocjitsu::cli::effective_visible_gpus(test_gpus(), "1,0", "1", std::nullopt);
  ASSERT_EQ(1u, selected.size());
  EXPECT_EQ(100u, selected[0].gpu_id);
}

TEST(RocmVisibilityTest, HipSelectorTakesPrecedenceOverCudaFallback) {
  const auto hip = rocjitsu::cli::effective_visible_gpus(test_gpus(), std::nullopt, "1", "2");
  const auto cuda =
      rocjitsu::cli::effective_visible_gpus(test_gpus(), std::nullopt, std::string_view{}, "2");

  ASSERT_EQ(1u, hip.size());
  EXPECT_EQ(101u, hip[0].gpu_id);
  ASSERT_EQ(1u, cuda.size());
  EXPECT_EQ(102u, cuda[0].gpu_id);
}

TEST(RocmVisibilityTest, NormalizesClientUuidToPostRocrOrdinal) {
  const auto normalized = rocjitsu::cli::normalized_client_visible_devices(
      test_gpus(), "1,0", "GPU-1111111111111111", std::nullopt);
  ASSERT_TRUE(normalized);
  EXPECT_EQ("HIP_VISIBLE_DEVICES", normalized->name);
  EXPECT_EQ("1", normalized->value);
}

TEST(RocmVisibilityTest, NormalizesClientSelectorAgainstExpandedRocrOrder) {
  const auto normalized =
      rocjitsu::cli::normalized_client_visible_devices(test_gpus(), "1,0,3", "1", std::nullopt);
  ASSERT_TRUE(normalized);
  EXPECT_EQ("HIP_VISIBLE_DEVICES", normalized->name);
  EXPECT_EQ("1", normalized->value);
}

TEST(RocmVisibilityTest, NormalizesSelectedDbtHostToClientDeviceZero) {
  const std::vector<rocjitsu::cli::VisibleGpu> heterogeneous_gpus{
      {0, 100, 110000, 0x1111111111111111ULL}, {1, 101, 90402, 0x2222222222222222ULL}};
  const auto normalized = rocjitsu::cli::normalized_client_visible_devices(
      heterogeneous_gpus, std::nullopt, "0,1", std::nullopt, 101);

  ASSERT_TRUE(normalized);
  EXPECT_EQ("HIP_VISIBLE_DEVICES", normalized->name);
  EXPECT_EQ("1,0", normalized->value);
}

TEST(RocmVisibilityTest, SelectsDbtHostFromClientVisibleGpusBeforeNormalization) {
  const auto visible =
      rocjitsu::cli::effective_visible_gpus(test_gpus(), std::nullopt, "1", std::nullopt);
  const auto automatic = rocjitsu::cli::select_host_gpu(visible, 0, 90402);
  ASSERT_EQ(rocjitsu::cli::HostSelectionStatus::Selected, automatic.status);
  EXPECT_EQ(101u, automatic.gpu_id);

  const auto normalized = rocjitsu::cli::normalized_client_visible_devices(
      test_gpus(), std::nullopt, "1", std::nullopt, automatic.gpu_id);
  ASSERT_TRUE(normalized);
  EXPECT_EQ("1", normalized->value);

  const auto hidden_explicit = rocjitsu::cli::select_host_gpu(visible, 100, 90402);
  EXPECT_EQ(rocjitsu::cli::HostSelectionStatus::ExplicitGpuHidden, hidden_explicit.status);
}

TEST(RocmVisibilityTest, ExpandsRocrSelectionWithGuestOrdinal) {
  const auto expanded = rocjitsu::cli::expanded_rocr_visible_devices(test_gpus(), "GPU-2222");
  ASSERT_TRUE(expanded);
  EXPECT_EQ("1,3", *expanded);
}

TEST(RocmVisibilityTest, DoesNotExpandEmptyRocrSelection) {
  EXPECT_FALSE(rocjitsu::cli::expanded_rocr_visible_devices(test_gpus(), "5"));
}

TEST(RocmVisibilityTest, SelectsFirstVisibleIsaMatch) {
  const rocjitsu::cli::HostSelection selection =
      rocjitsu::cli::select_host_gpu(test_gpus(), 0, 90402);
  EXPECT_EQ(rocjitsu::cli::HostSelectionStatus::Selected, selection.status);
  EXPECT_EQ(100u, selection.gpu_id);
}

TEST(RocmVisibilityTest, SelectsExplicitGpuWithMatchingIsa) {
  const rocjitsu::cli::HostSelection selection =
      rocjitsu::cli::select_host_gpu(test_gpus(), 100, 90402);
  EXPECT_EQ(rocjitsu::cli::HostSelectionStatus::Selected, selection.status);
  EXPECT_EQ(100u, selection.gpu_id);
}

TEST(RocmVisibilityTest, RejectsHiddenExplicitGpu) {
  const rocjitsu::cli::HostSelection selection =
      rocjitsu::cli::select_host_gpu({test_gpus()[0]}, 101, 90402);
  EXPECT_EQ(rocjitsu::cli::HostSelectionStatus::ExplicitGpuHidden, selection.status);
  EXPECT_EQ(101u, selection.gpu_id);
}

TEST(RocmVisibilityTest, RejectsExplicitGpuWithDifferentIsa) {
  const rocjitsu::cli::HostSelection selection =
      rocjitsu::cli::select_host_gpu(test_gpus(), 102, 90402);
  EXPECT_EQ(rocjitsu::cli::HostSelectionStatus::ExplicitGpuIsaMismatch, selection.status);
  EXPECT_EQ(102u, selection.gpu_id);
}

TEST(RocmVisibilityTest, ReportsMissingIsaMatch) {
  const rocjitsu::cli::HostSelection selection =
      rocjitsu::cli::select_host_gpu(test_gpus(), 0, 999999);
  EXPECT_EQ(rocjitsu::cli::HostSelectionStatus::NoIsaMatch, selection.status);
  EXPECT_EQ(0u, selection.gpu_id);
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
