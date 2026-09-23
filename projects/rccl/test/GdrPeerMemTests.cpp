/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "gdr_peermem.h"
#include "debug.h"

#include <gtest/gtest.h>

#include <cstdio>
#include <filesystem>
#include <string>
#include <system_error>
#include <unistd.h>
#include <vector>

// Weak stub for ncclDebugLog, required because gdr_peermem.cc uses the debug.h
// INFO() macro. Debug builds: ncclDebugLog is exported from librccl.so
// (-fvisibility=default), so the strong shared-library definition wins at link
// time and this stub is unused. Release builds: ncclDebugLog is hidden in
// librccl.so (-fvisibility=hidden), so this stub provides the symbol and drops logs.
// Mirrors the pattern in test/AltRsmiTests.cpp.
void __attribute__((weak)) ncclDebugLog(ncclDebugLogLevel, unsigned long, const char*, int, const char*, ...) {}

namespace RcclUnitTesting {

namespace {
namespace fs = std::filesystem;

// Each test gets its own PID-scoped sandbox directory so parallel CI runs and
// leftover state from a previous run cannot collide.
class GdrPeerMem : public ::testing::Test {
 protected:
  std::string root_;

  void SetUp() override {
    const char* name = ::testing::UnitTest::GetInstance()->current_test_info()->name();
    root_ = (fs::temp_directory_path() / ("gdr_peermem_test_" + std::to_string(getpid())) / name).string();
    std::error_code ec;
    fs::remove_all(root_, ec);
    fs::create_directories(root_);
  }

  void TearDown() override {
    std::error_code ec;
    fs::remove_all(root_, ec);
  }

  // Absolute path to `rel` inside this test's sandbox.
  std::string Path(const std::string& rel) { return (fs::path(root_) / rel).string(); }

  void MakeDir(const std::string& rel) { fs::create_directories(Path(rel)); }

  void MakeFile(const std::string& rel) {
    std::FILE* f = std::fopen(Path(rel).c_str(), "w");
    ASSERT_NE(f, nullptr) << "failed to create " << Path(rel);
    std::fclose(f);
  }

  // A fully registered client: a named subdirectory holding a `version` attribute.
  void MakeClient(const std::string& rel) {
    MakeDir(rel);
    MakeFile(rel + "/version");
  }
};

}  // namespace

// A registered client (named subdirectory with a `version` attribute) is detected.
TEST_F(GdrPeerMem, ClientPresent_SubdirDetected) {
  MakeClient("memory_peers/amdkfd");
  const std::string base = Path("memory_peers");
  const char* paths[] = {base.c_str(), nullptr};
  EXPECT_EQ(ncclIbScanPeerMemClients(paths), 1);
}

// An existing but empty memory_peers directory means no client is loaded. This
// is the false-positive case the sysfs approach fixes relative to the kallsyms scan.
TEST_F(GdrPeerMem, NoClient_EmptyDir) {
  MakeDir("memory_peers");
  const std::string base = Path("memory_peers");
  const char* paths[] = {base.c_str(), nullptr};
  EXPECT_EQ(ncclIbScanPeerMemClients(paths), 0);
}

// opendir() returns NULL for an absent base path; detection must simply skip it.
TEST_F(GdrPeerMem, NoClient_AbsentPath) {
  const std::string base = Path("does_not_exist");
  const char* paths[] = {base.c_str(), nullptr};
  EXPECT_EQ(ncclIbScanPeerMemClients(paths), 0);
}

// A regular file in memory_peers (e.g. a stray `version`) is not a client and
// must not trigger detection.
TEST_F(GdrPeerMem, FilesOnly_NotDetected) {
  MakeDir("memory_peers");
  MakeFile("memory_peers/version");
  const std::string base = Path("memory_peers");
  const char* paths[] = {base.c_str(), nullptr};
  EXPECT_EQ(ncclIbScanPeerMemClients(paths), 0);
}

// Detection is client-name agnostic: a client that is not `amdkfd` is still found.
TEST_F(GdrPeerMem, ClientPresent_NonAmdkfdName) {
  MakeClient("memory_peers/some_other_client");
  const std::string base = Path("memory_peers");
  const char* paths[] = {base.c_str(), nullptr};
  EXPECT_EQ(ncclIbScanPeerMemClients(paths), 1);
}

// A subdirectory without a `version` attribute is not a registered client. Keying on
// `version` rather than on the dirent type is what makes readdir's DT_UNKNOWN irrelevant.
TEST_F(GdrPeerMem, SubdirWithoutVersion_NotDetected) {
  MakeDir("memory_peers/not_a_client");
  const std::string base = Path("memory_peers");
  const char* paths[] = {base.c_str(), nullptr};
  EXPECT_EQ(ncclIbScanPeerMemClients(paths), 0);
}

// A hidden subdirectory (leading dot) is skipped even when it looks like a complete
// client, proving the dotfile filter rejects real entries and not just "." / "..".
TEST_F(GdrPeerMem, HiddenSubdirIgnored) {
  MakeClient("memory_peers/.hidden");
  const std::string base = Path("memory_peers");
  const char* paths[] = {base.c_str(), nullptr};
  EXPECT_EQ(ncclIbScanPeerMemClients(paths), 0);
}

// The scan walks the list in order: a client found in a later base path is
// detected even when earlier ones are absent.
TEST_F(GdrPeerMem, MultipleBasePaths_SecondHasClient) {
  MakeClient("second/amdkfd");
  const std::string p0 = Path("first_absent");
  const std::string p1 = Path("second");
  const char* paths[] = {p0.c_str(), p1.c_str(), nullptr};
  EXPECT_EQ(ncclIbScanPeerMemClients(paths), 1);
}

// A match in the first base path short-circuits: detection succeeds without
// requiring the remaining paths to exist.
TEST_F(GdrPeerMem, FirstMatchShortCircuits) {
  MakeClient("first/amdkfd");
  const std::string p0 = Path("first");
  const std::string p1 = Path("second_absent");
  const char* paths[] = {p0.c_str(), p1.c_str(), nullptr};
  EXPECT_EQ(ncclIbScanPeerMemClients(paths), 1);
}

namespace {
// Stands in for the per-device registration probe: fails on device `failDev` (-1 = never) and
// records which devices it was asked about.
struct FakeRegProbe {
  int failDev = -1;
  std::vector<int> calls;
};

int FakeRegProbeFn(int dev, void* ctx) {
  FakeRegProbe* fake = static_cast<FakeRegProbe*>(ctx);
  fake->calls.push_back(dev);
  return dev != fake->failDev;
}
}  // namespace

// With no IB devices there is nothing to register on, so peermem is not enabled and the probe
// is never invoked.
TEST(GdrPeerMemProbe, NoDevices_NotEnabled) {
  FakeRegProbe fake;
  EXPECT_EQ(ncclIbProbePeerMemAllDevs(0, FakeRegProbeFn, &fake), 0);
  EXPECT_TRUE(fake.calls.empty());
}

// Registration succeeding on every device enables peermem, and every device was probed.
TEST(GdrPeerMemProbe, AllDevicesRegister_Enabled) {
  FakeRegProbe fake;
  EXPECT_EQ(ncclIbProbePeerMemAllDevs(4, FakeRegProbeFn, &fake), 1);
  EXPECT_EQ(fake.calls, (std::vector<int>{0, 1, 2, 3}));
}

// A host with no GPU-memory registration support fails on the first device and keeps the
// DMA-BUF fallback. This is the configuration the sysfs scan was introduced to fix.
TEST(GdrPeerMemProbe, FirstDeviceFails_NotEnabled) {
  FakeRegProbe fake;
  fake.failDev = 0;
  EXPECT_EQ(ncclIbProbePeerMemAllDevs(4, FakeRegProbeFn, &fake), 0);
  EXPECT_EQ(fake.calls, (std::vector<int>{0}));
}

// Mixed-HCA host: registration works on the first devices but not on a later one. Peermem must
// stay disabled, and probing stops at the failing device.
TEST(GdrPeerMemProbe, LaterDeviceFails_NotEnabled) {
  FakeRegProbe fake;
  fake.failDev = 2;
  EXPECT_EQ(ncclIbProbePeerMemAllDevs(4, FakeRegProbeFn, &fake), 0);
  EXPECT_EQ(fake.calls, (std::vector<int>{0, 1, 2}));
}

// The last device alone failing is enough to disable peermem.
TEST(GdrPeerMemProbe, LastDeviceFails_NotEnabled) {
  FakeRegProbe fake;
  fake.failDev = 3;
  EXPECT_EQ(ncclIbProbePeerMemAllDevs(4, FakeRegProbeFn, &fake), 0);
  EXPECT_EQ(fake.calls, (std::vector<int>{0, 1, 2, 3}));
}

}  // namespace RcclUnitTesting
