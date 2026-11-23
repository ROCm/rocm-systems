/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/misc/kernel_config.cc's candidate search. snprintf is seamed in this
// TU so every absolute candidate path is re-rooted under a scratch dir, and the test owns the search.

#include <gtest/gtest.h>

#include <stdlib.h>
#include <sys/utsname.h>

#include <cstdarg>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

#include "kernel_config.h"

namespace {
std::string g_root;
__attribute__((format(printf, 3, 4))) int KernelConfigSnprintf(char* buf, size_t n, const char* fmt, ...) {
  char formatted[512];
  va_list ap;
  va_start(ap, fmt);
  std::vsnprintf(formatted, sizeof(formatted), fmt, ap);
  va_end(ap);
  const bool candidate = fmt[0] == '/';  // the path templates; "zcat %s" and "%s" pass through
  return std::snprintf(buf, n, "%s%s", candidate ? g_root.c_str() : "", formatted);
}
}  // namespace

#define snprintf KernelConfigSnprintf
#include KERNEL_CONFIG_CC_PATH
#undef snprintf

namespace {

// Only the candidates a test writes exist under the scratch root, so the search reaches them in
// template order: /usr/lib/modules/%s/config before /lib/modules/%s/build/.config.
class KernelConfigSearchMicrotest : public ::testing::Test {
 protected:
  std::string release_;

  void SetUp() override {
    char tmpl[] = "/tmp/rccl-kcfg-XXXXXX";
    ASSERT_NE(nullptr, mkdtemp(tmpl));
    g_root = tmpl;
    struct utsname u;
    ASSERT_EQ(0, ::uname(&u));
    release_ = u.release;
  }
  void TearDown() override {
    if (!g_root.empty()) std::filesystem::remove_all(g_root);
    g_root.clear();
  }
  std::string ModulesConfig() const { return "/usr/lib/modules/" + release_ + "/config"; }
  std::string BuildConfig() const { return "/lib/modules/" + release_ + "/build/.config"; }
  static void Write(const std::string& path, const std::string& text) {
    std::filesystem::create_directories(std::filesystem::path(g_root + path).parent_path());
    std::ofstream file(g_root + path);
    ASSERT_TRUE(file.is_open()) << path;
    file << text;
    file.close();
    ASSERT_TRUE(file.good()) << path;
  }
};

TEST_F(KernelConfigSearchMicrotest, ReadFirstAvailable_SkipsEmptyCandidateForLaterOne) {
  ASSERT_NO_FATAL_FAILURE(Write(ModulesConfig(), ""));
  ASSERT_NO_FATAL_FAILURE(Write(BuildConfig(), "CONFIG_PCI_P2PDMA=y\n"));
  std::string content;
  char path[256] = {};

  ASSERT_TRUE(ncclKernelConfigReadFirstAvailable(&content, path, sizeof(path)));

  EXPECT_EQ(g_root + BuildConfig(), path) << "an empty config file was taken as the kernel config";
  EXPECT_TRUE(ncclKernelConfigContentHasOption(content, "CONFIG_PCI_P2PDMA=y"));
}

TEST_F(KernelConfigSearchMicrotest, ReadFirstAvailable_FirstNonEmptyCandidateWins) {
  ASSERT_NO_FATAL_FAILURE(Write(ModulesConfig(), "CONFIG_A=y\n"));
  ASSERT_NO_FATAL_FAILURE(Write(BuildConfig(), "CONFIG_B=y\n"));
  std::string content;
  char path[256] = {};

  ASSERT_TRUE(ncclKernelConfigReadFirstAvailable(&content, path, sizeof(path)));

  EXPECT_EQ(g_root + ModulesConfig(), path);
  EXPECT_EQ("CONFIG_A=y\n", content);
}

TEST_F(KernelConfigSearchMicrotest, ReadFirstAvailable_OnlyEmptyCandidatesIsUnavailable) {
  ASSERT_NO_FATAL_FAILURE(Write(ModulesConfig(), ""));
  ASSERT_NO_FATAL_FAILURE(Write(BuildConfig(), ""));
  std::string content;
  char path[256] = {};

  EXPECT_FALSE(ncclKernelConfigReadFirstAvailable(&content, path, sizeof(path)))
      << "returned true with empty content from " << path;
}

}  // namespace
