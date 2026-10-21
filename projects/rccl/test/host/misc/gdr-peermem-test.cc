/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host-only tests for src/misc/gdr_peermem.cc. The scan's client matching is covered by
 * test/GdrPeerMemTests.cpp (rccl-UnitTestsGdr); these cover the opened bases, PATH_MAX and
 * the default paths.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>

#include <dirent.h>
#include <limits.h>
#include <stdlib.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

#include "ScopedHook.h"

// Records every directory the scan opens; the default opens it for real. Local to this test:
// gdr_peermem.cc is the only unit that scans directories.
static std::vector<std::string> g_openedDirs;
static std::function<DIR*(const char*)> g_opendir = [](const char* path) { return opendir(path); };
static DIR* GdrPeermemTestOpendir(const char* path) {
  g_openedDirs.push_back(path);
  return g_opendir(path);
}

#define opendir GdrPeermemTestOpendir
#include GDR_PEERMEM_CC_PATH
#undef opendir

namespace {

namespace fs = std::filesystem;

class GdrPeermemScanMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    std::string tmpl = ::testing::TempDir() + "gdr-peermem-test-XXXXXX";
    ASSERT_NE(mkdtemp(tmpl.data()), nullptr);
    root_ = tmpl;
    g_openedDirs.clear();
  }
  void TearDown() override { fs::remove_all(root_); }

  // Creates <root>/<base>/ and returns its path.
  std::string MakeBase(const std::string& base) {
    std::string path = root_ + "/" + base;
    fs::create_directories(path);
    return path;
  }
  static void Touch(const std::string& path) { std::ofstream(path).put('1'); }
  static void AddClient(const std::string& base, const std::string& name) {
    fs::create_directory(base + "/" + name);
    Touch(base + "/" + name + "/version");
  }

  int Scan(std::vector<std::string> bases) {
    std::vector<const char*> paths;
    for (const auto& base : bases) paths.push_back(base.c_str());
    paths.push_back(nullptr);
    return ncclIbScanPeerMemClients(paths.data());
  }

  std::string root_;
};

TEST_F(GdrPeermemScanMicrotest, Scan_EmptyList_FindsNothing) {
  EXPECT_EQ(Scan({}), 0);
  EXPECT_TRUE(g_openedDirs.empty());
}

TEST_F(GdrPeermemScanMicrotest, Scan_ClientInFirstBase_DoesNotOpenLaterBases) {
  std::string first = MakeBase("first");
  std::string second = MakeBase("second");
  AddClient(first, "amdkfd");
  AddClient(second, "amdkfd");

  EXPECT_EQ(Scan({first, second}), 1);
  EXPECT_EQ(g_openedDirs, (std::vector<std::string>{first}));
}

TEST_F(GdrPeermemScanMicrotest, Scan_NoClientInFirstBase_OpensEveryBase) {
  std::string first = MakeBase("first");
  std::string second = MakeBase("second");

  EXPECT_EQ(Scan({first, second}), 0);
  EXPECT_EQ(g_openedDirs, (std::vector<std::string>{first, second}));
}

// "<base>/<entry>/version" is exactly PATH_MAX characters, one too many for the buffer.
// The truncated path names a real file, so only the length check keeps it from matching.
TEST_F(GdrPeermemScanMicrotest, Scan_VersionPathDoesNotFitPathMax_SkipsEntry) {
  const size_t kEntryPathLen = PATH_MAX - std::string("/version").size();
  std::string base = root_;
  while (kEntryPathLen - base.size() - 1 > 200) base += "/" + std::string(150, 'd');
  fs::create_directories(base);
  std::string entry = base + "/" + std::string(kEntryPathLen - base.size() - 1, 'e');
  ASSERT_EQ(entry.size(), kEntryPathLen);
  fs::create_directory(entry);
  Touch(entry + "/versio");  // what snprintf leaves of "<entry>/version"
  ASSERT_TRUE(fs::exists(entry + "/versio"));

  EXPECT_EQ(Scan({base}), 0);
}

TEST(GdrPeermemDefaultScanMicrotest, ScanDefault_NoClient_ProbesEveryStandardMemoryPeersPath) {
  ScopedHook noDirs(g_opendir, [](const char*) -> DIR* { return nullptr; });
  g_openedDirs.clear();

  EXPECT_EQ(ncclIbScanDefaultPeerMemClients(), 0);
  std::sort(g_openedDirs.begin(), g_openedDirs.end());
  EXPECT_EQ(g_openedDirs,
            (std::vector<std::string>{"/sys/kernel/memory_peers", "/sys/kernel/mm/memory_peers", "/sys/memory_peers"}));
}

}  // namespace
