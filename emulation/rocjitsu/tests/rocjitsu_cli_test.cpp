// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#define main rocjitsu_cli_main_for_test
#include "../tools/rocjitsu/main.cpp"
#undef main

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace {

class ScopedEnv {
public:
  ScopedEnv(const char *name, const char *value) : name_(name) {
    if (const char *prior = std::getenv(name_.c_str()))
      prior_ = prior;
    if (value)
      setenv(name_.c_str(), value, 1);
    else
      unsetenv(name_.c_str());
  }

  ~ScopedEnv() {
    if (prior_)
      setenv(name_.c_str(), prior_->c_str(), 1);
    else
      unsetenv(name_.c_str());
  }

private:
  std::string name_;
  std::optional<std::string> prior_;
};

std::vector<std::string> split_preload(std::string_view preload) {
  std::vector<std::string> entries;
  while (!preload.empty()) {
    auto pos = preload.find(':');
    entries.emplace_back(preload.substr(0, pos));
    if (pos == std::string_view::npos)
      break;
    preload.remove_prefix(pos + 1);
  }
  return entries;
}

size_t find_entry(const std::vector<std::string> &entries, std::string_view needle) {
  auto it = std::find(entries.begin(), entries.end(), needle);
  return it == entries.end() ? entries.size() : static_cast<size_t>(it - entries.begin());
}

TEST(RocjitsuCliTest, DetectsSanitizerRuntimeBasenames) {
  EXPECT_TRUE(is_sanitizer_runtime("libclang_rt.asan-x86_64.so"));
  EXPECT_TRUE(is_sanitizer_runtime("libclang_rt.tsan-x86_64.so"));
  EXPECT_TRUE(is_sanitizer_runtime("libclang_rt.ubsan_standalone-x86_64.so"));
  EXPECT_TRUE(is_sanitizer_runtime("libasan.so.8"));
  EXPECT_TRUE(is_sanitizer_runtime("libtsan.so.2"));
  EXPECT_TRUE(is_sanitizer_runtime("libubsan.so.1"));

  EXPECT_FALSE(is_sanitizer_runtime("libhsa-runtime64.so"));
  EXPECT_FALSE(is_sanitizer_runtime("librocjitsu.so"));
}

TEST(RocjitsuCliTest, BuildsDeduplicatedPreloadWithInterposerBeforeExistingEntries) {
  constexpr const char *kInterposer = "/tmp/librocjitsu.so";
  constexpr const char *kExisting = "/tmp/existing.so";
  constexpr const char *kExtra = "/tmp/extra.so";
  ScopedEnv preload("LD_PRELOAD",
                    "/tmp/existing.so:/tmp/librocjitsu.so::/tmp/extra.so:/tmp/existing.so");

  auto entries = split_preload(build_ld_preload(kInterposer));

  EXPECT_EQ(std::count(entries.begin(), entries.end(), kInterposer), 1);
  EXPECT_EQ(std::count(entries.begin(), entries.end(), kExisting), 1);
  EXPECT_EQ(std::count(entries.begin(), entries.end(), kExtra), 1);

  const size_t interposer_pos = find_entry(entries, kInterposer);
  const size_t existing_pos = find_entry(entries, kExisting);
  const size_t extra_pos = find_entry(entries, kExtra);
  ASSERT_NE(interposer_pos, entries.size());
  ASSERT_NE(existing_pos, entries.size());
  ASSERT_NE(extra_pos, entries.size());
  EXPECT_LT(interposer_pos, existing_pos);
  EXPECT_LT(existing_pos, extra_pos);
}

} // namespace
