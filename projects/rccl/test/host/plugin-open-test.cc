/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/plugin/plugin_open.cc: plugin library name resolution, the not-found report, and
// sharing the NET/GIN library with other plugin types. dlopen, dlerror, dlclose and dlinfo are seamed in this TU.

#include <gtest/gtest.h>

#include <csignal>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "../common/LogCapture.hpp"
#include "debug.h"
#include "os.h"
#include "plugin.h"

using RcclUnitTesting::CaptureLog;
using RcclUnitTesting::LogHas;

void DisableCoreDumpsForExpectedCrash();  // plugin-net-test.cc

// ASan turns a SIGSEGV into a report plus exit(1), so KilledBySignal never matches under it.
#if defined(__SANITIZE_ADDRESS__) || (defined(__has_feature) && __has_feature(address_sanitizer))
#define DEATH_BY_SEGV ::testing::ExitedWithCode(1)
#else
#define DEATH_BY_SEGV ::testing::KilledBySignal(SIGSEGV)
#endif

namespace {

// A library either loads, is missing, or fails with some other dlopen error.
struct FakeDl {
  std::map<std::string, void*> installed;
  std::map<std::string, std::string> brokenErrors;
  std::vector<std::string> opened;  // "" for dlopen(NULL)
  std::map<void*, int> openCount;
  std::map<void*, int> closeCount;
  std::string error;
  bool nullError = false;
  bool dlinfoFails = false;
};
FakeDl g_dl;

void* const kSelf = reinterpret_cast<void*>(0x5e1f);

void* FakeDlopen(const char* name) {
  g_dl.opened.push_back(name ? name : "");
  void* handle = name ? nullptr : kSelf;
  if (name) {
    auto it = g_dl.installed.find(name);
    if (it != g_dl.installed.end()) handle = it->second;
    else if (g_dl.brokenErrors.count(name)) g_dl.error = g_dl.brokenErrors[name];
    else g_dl.error = std::string(name) + ": cannot open shared object file: No such file or directory";
  }
  if (handle) ++g_dl.openCount[handle];
  return handle;
}

const char* FakeDlerror() { return g_dl.nullError ? nullptr : g_dl.error.c_str(); }

void FakeDlclose(void* handle) { ++g_dl.closeCount[handle]; }

int FakeDlinfo(void* handle, int request, void* info) {
  static struct link_map lm;
  static std::string path;
  if (g_dl.dlinfoFails || request != RTLD_DI_LINKMAP) return -1;
  path = "/fake/" + std::to_string(reinterpret_cast<uintptr_t>(handle));
  lm.l_name = const_cast<char*>(path.c_str());
  *static_cast<struct link_map**>(info) = &lm;
  return 0;
}

}  // namespace

#define ncclOsDlopen FakeDlopen
#define ncclOsDlerror FakeDlerror
#define ncclOsDlclose FakeDlclose
#define dlinfo FakeDlinfo

#include PLUGIN_OPEN_CC_PATH

#undef dlinfo
#undef ncclOsDlclose
#undef ncclOsDlerror
#undef ncclOsDlopen

namespace {

void* const kNet = reinterpret_cast<void*>(0x1);
void* const kGin = reinterpret_cast<void*>(0x2);

std::string PathOf(void* handle) { return "/fake/" + std::to_string(reinterpret_cast<uintptr_t>(handle)); }

class PluginOpenMicrotest : public ::testing::Test {
 protected:
  RcclUnitTesting::ScopedDebugLogging logAll_{-1, ~uint64_t(0)};  // nccl_fakes.cc mutes INFO by default
  void SetUp() override {
    ResetSlots();
    g_dl = FakeDl{};
  }
  void TearDown() override { ResetSlots(); }

  static void ResetSlots() {
    for (int t = 0; t < NUM_LIBS; t++) {
      free(libNames[t]);
      free(ncclPluginLibPaths[t]);
      libNames[t] = ncclPluginLibPaths[t] = nullptr;
      libHandles[t] = nullptr;
    }
  }
};

TEST_F(PluginOpenMicrotest, NoNameOpensTheTypesDefaultLibrary) {
  const char* expected[NUM_LIBS] = {"librccl-net.so",   "librccl-gin.so",      "librccl-rma.so",
                                    "librccl-tuner.so", "librccl-profiler.so", "librccl-env.so"};
  void* (*open[NUM_LIBS])(const char*) = {ncclOpenNetPluginLib,   ncclOpenGinPluginLib,      ncclOpenRmaPluginLib,
                                          ncclOpenTunerPluginLib, ncclOpenProfilerPluginLib, ncclOpenEnvPluginLib};
  for (int t = 0; t < NUM_LIBS; t++) {
    void* handle = reinterpret_cast<void*>(uintptr_t(0x100 + t));
    g_dl.installed[expected[t]] = handle;
    ASSERT_EQ(open[t](t % 2 ? "" : nullptr), handle) << expected[t];
    EXPECT_STREQ(ncclGetPluginLibName(static_cast<ncclPluginType>(t)), expected[t]);
    EXPECT_EQ(std::string(ncclPluginLibPaths[t]), PathOf(handle));
  }
  EXPECT_EQ(g_dl.opened.size(), size_t(NUM_LIBS));
}

TEST_F(PluginOpenMicrotest, ABareNameIsRetriedWithTheTypePrefix) {
  g_dl.installed["librccl-tuner-csv.so"] = kGin;
  ASSERT_EQ(ncclOpenTunerPluginLib("csv"), kGin);

  EXPECT_EQ(g_dl.opened, (std::vector<std::string>{"csv", "librccl-tuner-csv.so"}));
  EXPECT_STREQ(ncclGetPluginLibName(ncclPluginTypeTuner), "librccl-tuner-csv.so");
  EXPECT_EQ(std::string(ncclPluginLibPaths[ncclPluginTypeTuner]), PathOf(kGin));
}

TEST_F(PluginOpenMicrotest, MissingLibrariesAreReportedTogether) {
  std::string log = CaptureLog([&] { EXPECT_EQ(ncclOpenNetPluginLib("ib2"), nullptr); });

  EXPECT_EQ(g_dl.opened, (std::vector<std::string>{"ib2", "librccl-net-ib2.so"}));
  EXPECT_TRUE(LogHas(log, "NET/Plugin: Could not find: ib2 librccl-net-ib2.so\n")) << log;
  EXPECT_EQ(ncclGetPluginLibName(ncclPluginTypeNet), nullptr);
  EXPECT_EQ(ncclPluginLibPaths[ncclPluginTypeNet], nullptr);
}

// Only a "No such file" naming the library itself counts as missing; a missing dependency is a load error.
TEST_F(PluginOpenMicrotest, OtherLoadErrorsAreLoggedSeparately) {
  g_dl.brokenErrors["ib2"] = "ib2: undefined symbol: ibv_foo";
  g_dl.brokenErrors["librccl-net-ib2.so"] = "libdep.so: cannot open shared object file: No such file or directory";
  std::string log = CaptureLog([&] { EXPECT_EQ(ncclOpenNetPluginLib("ib2"), nullptr); });

  EXPECT_TRUE(LogHas(log, "NET/Plugin: ib2: ib2: undefined symbol: ibv_foo\n")) << log;
  EXPECT_TRUE(LogHas(log, "NET/Plugin: librccl-net-ib2.so: libdep.so: cannot open shared object file")) << log;
  EXPECT_FALSE(LogHas(log, "Could not find")) << log;

  g_dl.brokenErrors.clear();
  log = CaptureLog([&] { EXPECT_EQ(ncclOpenGinPluginLib("x"), nullptr); });
  EXPECT_TRUE(LogHas(log, "GIN/Plugin: Could not find: x librccl-gin-x.so\n")) << log;
}

TEST_F(PluginOpenMicrotest, ANullDlerrorIsLoggedAsEmpty) {
  g_dl.nullError = true;
  std::string log = CaptureLog([&] { EXPECT_EQ(ncclOpenRmaPluginLib("/opt/rma.so"), nullptr); });
  EXPECT_TRUE(LogHas(log, "RMA/Plugin: /opt/rma.so: \n")) << log;
}

TEST_F(PluginOpenMicrotest, PathsAndFullLibraryNamesAreTriedOnlyAsGiven) {
  for (const char* name : {"/opt/lib/x.so", "./x.so", "sub/x", "libx.so"}) {
    g_dl.opened.clear();
    EXPECT_EQ(ncclOpenProfilerPluginLib(name), nullptr);
    EXPECT_EQ(g_dl.opened, std::vector<std::string>{name});
  }
}

// Despite the comment, only a name with both "lib" and ".so" skips the prefixed retry.
// Pins AICOMRCCL-2990; a fix flips this pin.
TEST_F(PluginOpenMicrotest, OpenPluginLib_KnownBug_AHalfLibraryNameIsStillRetriedWithThePrefix) {
  EXPECT_EQ(ncclOpenNetPluginLib("x.so"), nullptr);
  EXPECT_EQ(ncclOpenNetPluginLib("libx"), nullptr);
  EXPECT_EQ(g_dl.opened, (std::vector<std::string>{"x.so", "librccl-net-x.so.so", "libx", "librccl-net-libx.so"}));
}

TEST_F(PluginOpenMicrotest, StaticPluginOpensTheProcessItself) {
  ASSERT_EQ(ncclOpenTunerPluginLib("static_plugin"), kSelf);
  EXPECT_EQ(g_dl.opened, std::vector<std::string>{""});
  EXPECT_STREQ(ncclGetPluginLibName(ncclPluginTypeTuner), "static_plugin");
}

// The STATIC_PLUGIN check compares only strlen(name) chars, so any prefix opens the process.
// Pins AICOMRCCL-2989; a fix flips this pin.
TEST_F(PluginOpenMicrotest, TryOpenLib_KnownBug_AnyPrefixOfStaticPluginOpensTheProcess) {
  for (const char* name : {"s", "Stat", "STATIC_"}) {
    g_dl.opened.clear();
    ResetSlots();
    EXPECT_EQ(ncclOpenNetPluginLib(name), kSelf) << name;
    EXPECT_EQ(g_dl.opened, std::vector<std::string>{""}) << name;
  }
  g_dl.opened.clear();
  EXPECT_EQ(ncclOpenNetPluginLib("STATIC_PLUGIN_X"), nullptr);
  EXPECT_EQ(g_dl.opened[0], "STATIC_PLUGIN_X");
}

TEST_F(PluginOpenMicrotest, AFailedDlinfoLeavesNoPath) {
  g_dl.installed["librccl-net.so"] = kNet;
  g_dl.dlinfoFails = true;
  ASSERT_EQ(ncclOpenNetPluginLib(nullptr), kNet);
  EXPECT_EQ(ncclPluginLibPaths[ncclPluginTypeNet], nullptr);
  EXPECT_STREQ(ncclGetPluginLibName(ncclPluginTypeNet), "librccl-net.so");

  g_dl.installed["librccl-net-x.so"] = kGin;
  ResetSlots();
  ASSERT_EQ(ncclOpenNetPluginLib("x"), kGin);
  EXPECT_EQ(ncclPluginLibPaths[ncclPluginTypeNet], nullptr);
}

TEST_F(PluginOpenMicrotest, CloseOnlyReleasesTheHandleOpenedForThatType) {
  g_dl.installed["librccl-net.so"] = kNet;
  ASSERT_EQ(ncclOpenNetPluginLib(nullptr), kNet);

  EXPECT_EQ(ncclClosePluginLib(nullptr, ncclPluginTypeNet), ncclSuccess);
  EXPECT_EQ(ncclClosePluginLib(kNet, ncclPluginTypeTuner), ncclSuccess);
  EXPECT_EQ(ncclClosePluginLib(kGin, ncclPluginTypeNet), ncclSuccess);
  EXPECT_TRUE(g_dl.closeCount.empty());

  EXPECT_EQ(ncclClosePluginLib(kNet, ncclPluginTypeNet), ncclSuccess);
  EXPECT_EQ(g_dl.closeCount[kNet], 1);
  EXPECT_EQ(libHandles[ncclPluginTypeNet], nullptr);
  EXPECT_EQ(ncclGetPluginLibName(ncclPluginTypeNet), nullptr);
  EXPECT_EQ(ncclPluginLibPaths[ncclPluginTypeNet], nullptr);
  EXPECT_EQ(ncclClosePluginLib(kNet, ncclPluginTypeNet), ncclSuccess);
  EXPECT_EQ(g_dl.closeCount[kNet], 1);
}

TEST_F(PluginOpenMicrotest, SharingTheNetOrGinLibraryTakesItsOwnReference) {
  EXPECT_EQ(ncclGetNetPluginLib(ncclPluginTypeTuner), nullptr);
  EXPECT_EQ(ncclGetGinPluginLib(ncclPluginTypeRma), nullptr);
  EXPECT_TRUE(g_dl.opened.empty());

  g_dl.installed["librccl-net.so"] = kNet;
  g_dl.installed["librccl-gin.so"] = kGin;
  ASSERT_EQ(ncclOpenNetPluginLib(nullptr), kNet);
  ASSERT_EQ(ncclOpenGinPluginLib(nullptr), kGin);
  ASSERT_EQ(ncclGetNetPluginLib(ncclPluginTypeTuner), kNet);
  ASSERT_EQ(ncclGetGinPluginLib(ncclPluginTypeProfiler), kGin);

  EXPECT_EQ(g_dl.openCount[kNet], 2);
  EXPECT_EQ(g_dl.openCount[kGin], 2);
  EXPECT_STREQ(ncclGetPluginLibName(ncclPluginTypeTuner), "librccl-net.so");
  EXPECT_STREQ(ncclGetPluginLibName(ncclPluginTypeProfiler), "librccl-gin.so");
  EXPECT_EQ(std::string(ncclPluginLibPaths[ncclPluginTypeTuner]), PathOf(kNet));
  EXPECT_EQ(std::string(ncclPluginLibPaths[ncclPluginTypeProfiler]), PathOf(kGin));
  EXPECT_NE(ncclPluginLibPaths[ncclPluginTypeTuner], ncclPluginLibPaths[ncclPluginTypeNet]);  // a copy

  ASSERT_EQ(ncclClosePluginLib(kNet, ncclPluginTypeTuner), ncclSuccess);
  EXPECT_EQ(g_dl.closeCount[kNet], 1);
  EXPECT_EQ(libHandles[ncclPluginTypeNet], kNet);
}

// rma.cc shares GIN then NET into the RMA slot, and NET overwrites GIN's unreleased reference.
// Pins AICOMRCCL-2987; a fix flips this pin.
TEST_F(PluginOpenMicrotest, GetNetPluginLib_KnownBug_GinThenNetIntoOneSlotLeaksTheGinReference) {
  g_dl.installed["librccl-net.so"] = kNet;
  g_dl.installed["librccl-gin.so"] = kGin;
  ASSERT_EQ(ncclOpenNetPluginLib(nullptr), kNet);
  ASSERT_EQ(ncclOpenGinPluginLib(nullptr), kGin);
  ASSERT_EQ(ncclGetGinPluginLib(ncclPluginTypeRma), kGin);
  // The next call drops these copies; free them here so the leak under test stays out of LeakSanitizer.
  char* ginName = libNames[ncclPluginTypeRma];
  char* ginPath = ncclPluginLibPaths[ncclPluginTypeRma];
  ASSERT_EQ(ncclGetNetPluginLib(ncclPluginTypeRma), kNet);
  free(ginName);
  free(ginPath);
  EXPECT_STREQ(ncclGetPluginLibName(ncclPluginTypeRma), "librccl-net.so");

  ASSERT_EQ(ncclClosePluginLib(kGin, ncclPluginTypeRma), ncclSuccess);
  ASSERT_EQ(ncclClosePluginLib(kGin, ncclPluginTypeGin), ncclSuccess);
  EXPECT_EQ(g_dl.openCount[kGin], 2);
  EXPECT_EQ(g_dl.closeCount[kGin], 1);
}

// Sharing strdup's the NET library's path, which is NULL when dlinfo failed.
// Pins AICOMRCCL-2988; a fix flips this pin.
TEST_F(PluginOpenMicrotest, GetNetPluginLib_KnownBug_CrashesWhenTheNetLibraryHasNoPath) {
  g_dl.installed["librccl-net.so"] = kNet;
  g_dl.dlinfoFails = true;
  ASSERT_EQ(ncclOpenNetPluginLib(nullptr), kNet);
  const std::string style = ::testing::GTEST_FLAG(death_test_style);
  ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
  EXPECT_EXIT(
      {
        DisableCoreDumpsForExpectedCrash();
        (void)ncclGetNetPluginLib(ncclPluginTypeTuner);
      },
      DEATH_BY_SEGV, "");
  ::testing::GTEST_FLAG(death_test_style) = style;
}

}  // namespace
