/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/debug.cc: NCCL_DEBUG* parsing, the log line format, NCCL_DEBUG_FILE
// expansion, reset, and thread naming. The environment, clock, host/pid/tid and device are seamed here.

#include <gtest/gtest.h>

#include <pthread.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <condition_variable>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "../common/ScopedEnv.hpp"
#include "utils.h"

namespace {
// Not fakes/env_fakes.cc: DebugFile_KnownBug_TrailingPercentReadsPastTheName needs a value followed by a second NUL.
std::map<std::string, std::string> g_env;
bool g_envPluginReady = true;
std::string g_hostName;
ncclResult_t g_hostNameResult = ncclSuccess;
uint64_t g_pid = 0;
uint64_t g_tid = 0;
int g_device = 0;
int g_getDeviceCalls = 0;
timespec g_now = {};
int64_t g_setThreadName = 0;
std::vector<std::string> g_registeredKeys;
}  // namespace

bool ncclEnvPluginInitialized() { return g_envPluginReady; }
const char* ncclGetEnv(const char* name) {
  auto it = g_env.find(name);
  return it == g_env.end() ? nullptr : it->second.c_str();
}
ncclResult_t getHostName(char* hostname, int maxlen, const char /*delim*/) {
  if (g_hostNameResult != ncclSuccess) return g_hostNameResult;
  std::snprintf(hostname, maxlen, "%s", g_hostName.c_str());
  return ncclSuccess;
}
uint64_t ncclOsGetPid() { return g_pid; }
uint64_t ncclOsGetTid() { return g_tid; }
std::tm* ncclOsLocaltime(const time_t* timer, std::tm* buf) { return gmtime_r(timer, buf); }
hipError_t hipGetDevice(int* dev) {
  ++g_getDeviceCalls;
  *dev = g_device;
  return hipSuccess;
}
int64_t ncclLoadParam(char const* /*env*/, int64_t /*deftVal*/, int64_t /*uninitialized*/, int64_t* /*cache*/,
                      int8_t* /*noCache*/) {
  return g_setThreadName;
}

#include "param/param_registry.h"
ncclResult_t ncclParamRegistry::add(std::string key, ncclParamInfo_t /*info*/, ncclParamInterface* /*param*/) {
  g_registeredKeys.push_back(key);
  return ncclSuccess;
}
extern "C" const char* ncclParamEnvPluginGet(const char* /*key*/, bool /*envInit*/) { return nullptr; }
extern "C" bool ncclParamIsCacheDisabled(const char* /*key*/) { return false; }

// utils.h, included above, declares and defines the real clockRealtime, so the rename reaches only debug.cc's calls.
static void fakeClockRealtime(timespec* ts) { *ts = g_now; }
#define clockRealtime fakeClockRealtime
#include DEBUG_CC_PATH
#undef clockRealtime

namespace {

constexpr uint32_t Bit(int level) { return 1u << level; }
constexpr uint32_t kError = Bit(NCCL_LOG_ERROR), kVersion = Bit(NCCL_LOG_VERSION), kWarn = Bit(NCCL_LOG_WARN),
                   kAttn = Bit(NCCL_LOG_ATTN), kInfo = Bit(NCCL_LOG_INFO), kAbort = Bit(NCCL_LOG_ABORT),
                   kTrace = Bit(NCCL_LOG_TRACE);
constexpr uint64_t kDefaultSubsys = NCCL_INIT | NCCL_BOOTSTRAP | NCCL_ENV;

std::string ReadFile(const std::string& path) {
  std::ifstream in(path);
  std::stringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

class DebugMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    g_env.clear();
    g_envPluginReady = true;
    g_hostName = "node1";
    g_hostNameResult = ncclSuccess;
    g_pid = 4242;
    g_tid = 77;
    g_device = 3;
    g_getDeviceCalls = 0;
    g_now = {0, 123456789};  // 1970-01-01 00:00:00.123456789 UTC
    g_setThreadName = 0;
    tid = -1;
    ncclDebugNoWarn = 0;
    ncclWarnSetDebugInfo = false;
    ncclLastError[0] = '\0';
    capture_ = std::tmpfile();
    ASSERT_NE(capture_, nullptr);
  }

  void TearDown() override {
    if (ncclDebugFile != stdout && ncclDebugFile != capture_) std::fclose(ncclDebugFile);
    ncclDebugFile = stdout;
    std::fclose(capture_);
    ncclDebugLevelMask = NCCL_DEBUG_LEVEL_MASK_UNINITIALIZED;
    ncclDebugMask = 0;
    if (!dir_.empty()) std::filesystem::remove_all(dir_);
  }

  // A fresh directory for NCCL_DEBUG_FILE, made on first use.
  const std::string& Dir() {
    if (dir_.empty()) {
      char dirTemplate[] = "/tmp/rccl-debug-test-XXXXXX";
      EXPECT_NE(mkdtemp(dirTemplate), nullptr);
      dir_ = dirTemplate;
    }
    return dir_;
  }

  // Re-reads the environment the way a reset does, then captures output unless NCCL_DEBUG_FILE took it.
  void Init() {
    if (ncclDebugFile == capture_) ncclDebugFile = stdout;  // a reset would fclose() it
    ncclDebugLevelMask = NCCL_DEBUG_LEVEL_MASK_RESET_TRIGGERED;
    pthread_mutex_lock(&ncclDebugLock);
    ncclDebugInit();
    pthread_mutex_unlock(&ncclDebugLock);
    if (ncclDebugFile == stdout) ncclDebugFile = capture_;
  }

  uint32_t LevelsFor(const char* debug) {
    g_env["NCCL_DEBUG"] = debug;
    Init();
    return ncclDebugLevelMask;
  }

  uint64_t SubsysFor(const char* subsys) {
    g_env["NCCL_DEBUG_SUBSYS"] = subsys;
    Init();
    return ncclDebugMask;
  }

  uint32_t TimestampLevelsFor(const char* levels) {
    g_env["NCCL_DEBUG_TIMESTAMP_LEVELS"] = levels;
    Init();
    return ncclDebugTimestampLevels;
  }

  std::string Output() {
    std::fflush(capture_);
    std::rewind(capture_);
    std::string out;
    char buf[4096];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof(buf), capture_)) > 0) out.append(buf, n);
    return out;
  }

  std::string dir_;
  FILE* capture_ = nullptr;
};

TEST_F(DebugMicrotest, Levels_UnsetNcclDebugShowsOnlyFatalErrors) {
  Init();
  EXPECT_EQ(ncclDebugLevelMask, kError);
}

TEST_F(DebugMicrotest, Levels_EachNcclDebugValueIncludesTheLessVerboseOnes) {
  EXPECT_EQ(LevelsFor("NONE"), 0u);
  EXPECT_EQ(LevelsFor("VERSION"), kError | kVersion);
  EXPECT_EQ(LevelsFor("WARN"), kError | kVersion | kWarn);
  EXPECT_EQ(LevelsFor("ATTN"), kError | kVersion | kWarn | kAttn);
  EXPECT_EQ(LevelsFor("INFO"), kError | kVersion | kWarn | kAttn | kInfo);
  EXPECT_EQ(LevelsFor("ABORT"), kError | kVersion | kWarn | kAttn | kInfo | kAbort);
  EXPECT_EQ(LevelsFor("TRACE"), kError | kVersion | kWarn | kAttn | kInfo | kAbort | kTrace);
  EXPECT_EQ(LevelsFor("info"), kError | kVersion | kWarn | kAttn | kInfo);
}

TEST_F(DebugMicrotest, Levels_UnknownNcclDebugEnablesNothing) {
  EXPECT_EQ(LevelsFor("LOUD"), 0u);
}

TEST_F(DebugMicrotest, Levels_NcclDebugLevelsAddsEachNamedLevel) {
  g_env["NCCL_DEBUG"] = "NONE";
  const std::pair<const char*, uint32_t> cases[] = {
    {"ERROR", kError}, {"VERSION", kVersion}, {"WARN", kWarn},   {"ATTN", kAttn},
    {"INFO", kInfo},   {"ABORT", kAbort},     {"TRACE", kTrace},
    {"ALL", kError | kVersion | kWarn | kAttn | kInfo | kAbort | kTrace}};
  for (const auto& c : cases) {
    g_env["NCCL_DEBUG_LEVELS"] = c.first;
    Init();
    EXPECT_EQ(ncclDebugLevelMask, c.second) << c.first;
  }
  g_env["NCCL_DEBUG_LEVELS"] = "warn,bogus,Trace";
  Init();
  EXPECT_EQ(ncclDebugLevelMask, kWarn | kTrace);
}

TEST_F(DebugMicrotest, Levels_NcclDebugLevelsIsAddedOnTopOfNcclDebug) {
  g_env["NCCL_DEBUG_LEVELS"] = "TRACE";
  EXPECT_EQ(LevelsFor("WARN"), kError | kVersion | kWarn | kTrace);
}

TEST_F(DebugMicrotest, Subsys_DefaultIsInitBootstrapEnv) {
  Init();
  EXPECT_EQ(ncclDebugMask, kDefaultSubsys);
}

TEST_F(DebugMicrotest, Subsys_EachNameSelectsItsBit) {
  const std::pair<const char*, uint64_t> cases[] = {
    {"INIT", NCCL_INIT},     {"COLL", NCCL_COLL},       {"P2P", NCCL_P2P},     {"SHM", NCCL_SHM},
    {"NET", NCCL_NET},       {"GRAPH", NCCL_GRAPH},     {"TUNING", NCCL_TUNING}, {"ENV", NCCL_ENV},
    {"ALLOC", NCCL_ALLOC},   {"CALL", NCCL_CALL},       {"PROXY", NCCL_PROXY}, {"NVLS", NCCL_NVLS},
    {"BOOTSTRAP", NCCL_BOOTSTRAP}, {"REG", NCCL_REG},   {"PROFILE", NCCL_PROFILE}, {"RAS", NCCL_RAS},
    {"VERBS", NCCL_VERBS},   {"DESTROY", NCCL_DESTROY}, {"ALL", (uint64_t)NCCL_ALL}};
  for (const auto& c : cases) EXPECT_EQ(SubsysFor(c.first), c.second) << c.first;
}

TEST_F(DebugMicrotest, Subsys_ListIsCaseInsensitiveAndSkipsUnknownNames) {
  EXPECT_EQ(SubsysFor("coll,bogus,Net"), (uint64_t)(NCCL_COLL | NCCL_NET));
}

TEST_F(DebugMicrotest, Subsys_CaretExcludesTheListedNames) {
  EXPECT_EQ(SubsysFor("^INIT,NET"), ~(uint64_t)(NCCL_INIT | NCCL_NET));
}

// The published NCCL_DEBUG_SUBSYS options have ALLOC_HOST but not VERBS; the runtime parser has VERBS but not
// ALLOC_HOST. Pins AICOMRCCL-2950; a fix flips this pin.
TEST_F(DebugMicrotest, Subsys_KnownBug_AllocHostIsIgnored) {
  EXPECT_EQ(SubsysFor("ALLOC_HOST"), 0u);
  uint64_t published = 0;
  EXPECT_EQ(ncclParamDebugSubsys.parser.resolve("ALLOC_HOST", published), ncclSuccess);
  EXPECT_EQ(published, (uint64_t)NCCL_ALLOC_HOST);
  EXPECT_EQ(SubsysFor("VERBS"), (uint64_t)NCCL_VERBS);
  EXPECT_EQ(ncclParamDebugSubsys.parser.resolve("VERBS", published), ncclInvalidArgument);
}

TEST_F(DebugMicrotest, WarnEnableDebugInfo_ParsesAnInteger) {
  g_env["NCCL_WARN_ENABLE_DEBUG_INFO"] = "0x1";
  Init();
  EXPECT_TRUE(ncclWarnSetDebugInfo);
  g_env["NCCL_WARN_ENABLE_DEBUG_INFO"] = "0";
  Init();
  EXPECT_FALSE(ncclWarnSetDebugInfo);
}

TEST_F(DebugMicrotest, WarnEnableDebugInfo_EmptyOrOutOfRangeKeepsThePreviousValue) {
  ncclWarnSetDebugInfo = true;
  g_env["NCCL_WARN_ENABLE_DEBUG_INFO"] = "";
  Init();
  EXPECT_TRUE(ncclWarnSetDebugInfo);
  ncclWarnSetDebugInfo = false;
  g_env["NCCL_WARN_ENABLE_DEBUG_INFO"] = "99999999999999999999";  // ERANGE
  Init();
  EXPECT_FALSE(ncclWarnSetDebugInfo);
}

TEST_F(DebugMicrotest, WarnEnableDebugInfo_UnsetKeepsThePreviousValue) {
  g_env["NCCL_WARN_ENABLE_DEBUG_INFO"] = "1";
  Init();
  g_env.erase("NCCL_WARN_ENABLE_DEBUG_INFO");
  Init();
  EXPECT_TRUE(ncclWarnSetDebugInfo);
}

// The documented default is WARN,ATTN; the runtime default is WARN only.
TEST_F(DebugMicrotest, TimestampLevels_DefaultIsWarnOnly) {
  Init();
  EXPECT_EQ(ncclDebugTimestampLevels, kWarn);
}

TEST_F(DebugMicrotest, TimestampLevels_EachNameSelectsItsLevel) {
  EXPECT_EQ(TimestampLevelsFor("VERSION"), kVersion);
  EXPECT_EQ(TimestampLevelsFor("WARN"), kWarn);
  EXPECT_EQ(TimestampLevelsFor("INFO"), kInfo);
  EXPECT_EQ(TimestampLevelsFor("ABORT"), kAbort);
  EXPECT_EQ(TimestampLevelsFor("TRACE"), kTrace);
  EXPECT_EQ(TimestampLevelsFor("ALL"), ~0u);
  EXPECT_EQ(TimestampLevelsFor("info,bogus,Trace"), kInfo | kTrace);
}

TEST_F(DebugMicrotest, TimestampLevels_CaretExcludesTheListedLevels) {
  EXPECT_EQ(TimestampLevelsFor("^WARN,INFO"), ~(kWarn | kInfo));
}

// ATTN and ERROR are documented levels the runtime parser ignores.
TEST_F(DebugMicrotest, TimestampLevels_AttnAndErrorAreIgnored) {
  EXPECT_EQ(TimestampLevelsFor("ATTN,ERROR"), 0u);
}

TEST_F(DebugMicrotest, TimestampFormat_DefaultHasNoSubseconds) {
  Init();
  EXPECT_STREQ(ncclDebugTimestampFormat, "[%F %T] ");
  EXPECT_EQ(ncclDebugTimestampSubsecondsStart, -1);
}

TEST_F(DebugMicrotest, TimestampFormat_UnderscoresBecomeSpaces) {
  g_env["NCCL_DEBUG_TIMESTAMP_FORMAT"] = "%T_|_";
  Init();
  EXPECT_STREQ(ncclDebugTimestampFormat, "%T | ");
}

TEST_F(DebugMicrotest, TimestampFormat_ReservesSpaceForSubseconds) {
  g_env["NCCL_DEBUG_TIMESTAMP_FORMAT"] = "%T.%3f ";
  Init();
  EXPECT_STREQ(ncclDebugTimestampFormat, "%T.    ");
  EXPECT_EQ(ncclDebugTimestampSubsecondsStart, 3);
  EXPECT_EQ(ncclDebugTimestampSubsecondDigits, 3);
  EXPECT_EQ(ncclDebugTimestampMaxSubseconds, 1000u);
}

TEST_F(DebugMicrotest, TimestampFormat_EscapedOrZeroDigitIsNotSubseconds) {
  for (const char* fmt : {"%%3f", "%0f", "%3x"}) {
    g_env["NCCL_DEBUG_TIMESTAMP_FORMAT"] = fmt;
    Init();
    EXPECT_EQ(ncclDebugTimestampSubsecondsStart, -1) << fmt;
    EXPECT_STREQ(ncclDebugTimestampFormat, fmt);
  }
}

TEST_F(DebugMicrotest, TimestampFormat_TooLongFallsBack) {
  // 247 chars + 9 subsecond digits - 3 for "%9f" is 253: still fits.
  std::string fits = std::string(244, 'x') + "%9f";
  g_env["NCCL_DEBUG_TIMESTAMP_FORMAT"] = fits;
  Init();
  EXPECT_EQ(ncclDebugTimestampSubsecondsStart, 244);
  // One char past what fits with subseconds: the format is kept, but without subseconds.
  std::string noRoom = std::string(250, 'x') + "%9f";
  g_env["NCCL_DEBUG_TIMESTAMP_FORMAT"] = noRoom;
  Init();
  EXPECT_EQ(ncclDebugTimestampSubsecondsStart, -1);
  EXPECT_EQ(std::string(ncclDebugTimestampFormat), noRoom);
  g_env["NCCL_DEBUG_TIMESTAMP_FORMAT"] = std::string(255, 'x');
  Init();
  EXPECT_EQ(std::string(ncclDebugTimestampFormat), std::string(255, 'x'));
  g_env["NCCL_DEBUG_TIMESTAMP_FORMAT"] = std::string(256, 'x');
  Init();
  EXPECT_STREQ(ncclDebugTimestampFormat, "[%F %T] ");
}

TEST_F(DebugMicrotest, TimestampFormat_ExactFitKeepsSubseconds) {
  // 9 digits + 249 - 3 = 255, exactly the space available.
  g_env["NCCL_DEBUG_TIMESTAMP_FORMAT"] = std::string(246, 'x') + "%9f";
  Init();
  EXPECT_EQ(ncclDebugTimestampSubsecondsStart, 246);
  g_env["NCCL_DEBUG_TIMESTAMP_FORMAT"] = std::string(247, 'x') + "%9f";
  Init();
  EXPECT_EQ(ncclDebugTimestampSubsecondsStart, -1);
}

TEST_F(DebugMicrotest, HostName_PercentAndSlashBecomeDashes) {
  g_hostName = "a%b/c";
  g_env["NCCL_DEBUG"] = "INFO";
  Init();
  EXPECT_STREQ(hostname, "a-b-c");
  ncclDebugLog(NCCL_LOG_INFO, NCCL_INIT, "f", 1, "hi");
  EXPECT_EQ(Output(), "a-b-c:4242:77 [3] NCCL INFO hi\n");
}

TEST_F(DebugMicrotest, HostName_LookupFailureKeepsThePreviousName) {
  Init();
  g_hostName = "other";
  g_hostNameResult = ncclSystemError;
  Init();
  EXPECT_STREQ(hostname, "node1");
}

TEST_F(DebugMicrotest, Tid_IsLookedUpOncePerThread) {
  g_env["NCCL_DEBUG"] = "INFO";
  Init();
  ncclDebugLog(NCCL_LOG_INFO, NCCL_INIT, "f", 1, "a");
  g_tid = 78;
  ncclDebugLog(NCCL_LOG_INFO, NCCL_INIT, "f", 1, "b");
  std::thread([] { ncclDebugLog(NCCL_LOG_INFO, NCCL_INIT, "f", 1, "c"); }).join();
  EXPECT_EQ(Output(),
            "node1:4242:77 [3] NCCL INFO a\nnode1:4242:77 [3] NCCL INFO b\nnode1:4242:78 [3] NCCL INFO c\n");
}

TEST_F(DebugMicrotest, DebugFile_ExpandsHostPidAndPercent) {
  g_env["NCCL_DEBUG"] = "INFO";
  g_env["NCCL_DEBUG_FILE"] = Dir() + "/log.%h.%p.%%.%x";
  Init();
  ASSERT_NE(ncclDebugFile, stdout);
  ncclDebugLog(NCCL_LOG_INFO, NCCL_INIT, "f", 1, "to file");
  EXPECT_EQ(ReadFile(Dir() + "/log.node1.4242.%.%x"), "node1:4242:77 [3] NCCL INFO to file\n");
}

TEST_F(DebugMicrotest, DebugFile_IsLineBuffered) {
  g_env["NCCL_DEBUG"] = "WARN";
  g_env["NCCL_DEBUG_FILE"] = Dir() + "/log";
  g_env["NCCL_DEBUG_TIMESTAMP_LEVELS"] = "";
  Init();
  ncclDebugLog(NCCL_LOG_WARN, NCCL_INIT, "f", 1, "w");
  EXPECT_EQ(ReadFile(Dir() + "/log"), "\nnode1:4242:77 [3] f:1 NCCL WARN w\n");
}

// A trailing '%' makes the loop read past the NUL (a second NUL keeps this in bounds). Pins AICOMRCCL-2953; a fix
// flips this pin.
TEST_F(DebugMicrotest, DebugFile_KnownBug_TrailingPercentReadsPastTheName) {
  g_env["NCCL_DEBUG"] = "INFO";
  g_env["NCCL_DEBUG_FILE"] = Dir() + "/log%";
  g_env["NCCL_DEBUG_FILE"].push_back('\0');
  Init();
  ncclDebugLog(NCCL_LOG_INFO, NCCL_INIT, "f", 1, "x");
  EXPECT_EQ(ReadFile(Dir() + "/log%"), "node1:4242:77 [3] NCCL INFO x\n");
}

TEST_F(DebugMicrotest, DebugFile_NotOpenedForErrorAndVersionOnly) {
  g_env["NCCL_DEBUG_FILE"] = Dir() + "/log";
  for (const char* level : {"VERSION", "NONE"}) {
    g_env["NCCL_DEBUG"] = level;
    Init();
    EXPECT_EQ(ncclDebugFile, capture_) << level;
  }
  g_env.erase("NCCL_DEBUG");
  Init();
  EXPECT_EQ(ncclDebugFile, capture_);
  // Any level above VERSION, even one added by NCCL_DEBUG_LEVELS, opens it.
  g_env["NCCL_DEBUG_LEVELS"] = "ATTN";
  Init();
  EXPECT_NE(ncclDebugFile, capture_);
  struct stat st;
  EXPECT_EQ(stat((Dir() + "/log").c_str(), &st), 0);
}

TEST_F(DebugMicrotest, DebugFile_EmptyOrUnopenableStaysOnStdout) {
  g_env["NCCL_DEBUG"] = "INFO";
  g_env["NCCL_DEBUG_FILE"] = "";
  ncclDebugLevelMask = NCCL_DEBUG_LEVEL_MASK_RESET_TRIGGERED;
  ncclDebugInit();
  EXPECT_EQ(ncclDebugFile, stdout);
  g_env["NCCL_DEBUG_FILE"] = Dir() + "/missing/log";
  ncclDebugInit();
  EXPECT_EQ(ncclDebugFile, stdout);
}

TEST_F(DebugMicrotest, DebugFile_LongNamesAreClampedToPathMax) {
  g_env["NCCL_DEBUG"] = "INFO";
  g_hostName = std::string(100, 'h');
  for (const char* tail : {"%h", "%p", "%x", "%%", "zz"}) {
    g_env["NCCL_DEBUG_FILE"] = std::string(PATH_MAX - 1, 'a') + tail;
    ncclDebugLevelMask = NCCL_DEBUG_LEVEL_MASK_RESET_TRIGGERED;
    ncclDebugInit();
    EXPECT_EQ(ncclDebugFile, stdout) << tail;  // ENAMETOOLONG
  }
}

// "./" segments keep a path this long valid.
TEST_F(DebugMicrotest, DebugFile_PathUpToPathMaxIsUsed) {
  g_env["NCCL_DEBUG"] = "INFO";
  std::string head = Dir() + "/";
  while (head.size() < PATH_MAX - 200) head += "./";
  // The expanded name (".%p" becomes ".4242") is PATH_MAX - 1 chars.
  std::string expanded = head + std::string(PATH_MAX - 1 - head.size() - 5, 'L') + ".4242";
  g_env["NCCL_DEBUG_FILE"] = expanded.substr(0, expanded.size() - 4) + "%p";
  Init();
  ASSERT_NE(ncclDebugFile, capture_);
  ncclDebugLog(NCCL_LOG_INFO, NCCL_INIT, "f", 1, "deep");
  EXPECT_EQ(ReadFile(expanded), "node1:4242:77 [3] NCCL INFO deep\n");
}

TEST_F(DebugMicrotest, Log_InfoLine) {
  g_env["NCCL_DEBUG"] = "INFO";
  Init();
  ncclDebugLog(NCCL_LOG_INFO, NCCL_INIT, "fn", 12, "value=%d", 5);
  EXPECT_EQ(Output(), "node1:4242:77 [3] NCCL INFO value=5\n");
}

TEST_F(DebugMicrotest, Log_InfoIsFilteredBySubsystem) {
  g_env["NCCL_DEBUG"] = "INFO";
  Init();
  ncclDebugLog(NCCL_LOG_INFO, NCCL_NET, "fn", 12, "net");
  EXPECT_EQ(Output(), "");
}

TEST_F(DebugMicrotest, Log_LevelsOutsideTheMaskAreDropped) {
  g_env["NCCL_DEBUG"] = "WARN";
  Init();
  ncclDebugLog(NCCL_LOG_INFO, NCCL_INIT, "fn", 1, "info");
  ncclDebugLog(NCCL_LOG_ATTN, NCCL_INIT, "fn", 1, "attn");
  EXPECT_EQ(Output(), "");
}

TEST_F(DebugMicrotest, Log_WarnLineHasALeadingNewlineAndTimestamp) {
  g_env["NCCL_DEBUG"] = "WARN";
  Init();
  ncclDebugLog(NCCL_LOG_WARN, NCCL_ALL, "file.cc", 34, "bad %s", "thing");
  EXPECT_EQ(Output(), "\n[1970-01-01 00:00:00] node1:4242:77 [3] file.cc:34 NCCL WARN bad thing\n");
  EXPECT_STREQ(ncclLastError, "bad thing");
}

TEST_F(DebugMicrotest, Log_WarnSetsLastErrorEvenWhenNotPrinted) {
  g_env["NCCL_DEBUG"] = "NONE";
  Init();
  ncclDebugLog(NCCL_LOG_WARN, NCCL_ALL, "f", 1, "silent %d", 7);
  EXPECT_STREQ(ncclLastError, "silent 7");
  EXPECT_EQ(Output(), "");
}

TEST_F(DebugMicrotest, Log_AttnLineDoesNotSetLastError) {
  g_env["NCCL_DEBUG"] = "ATTN";
  Init();
  ncclDebugLog(NCCL_LOG_ATTN, NCCL_ALL, "a.cc", 2, "notice");
  EXPECT_EQ(Output(), "\nnode1:4242:77 [3] a.cc:2 NCCL ATTN notice\n");
  EXPECT_STREQ(ncclLastError, "");
}

TEST_F(DebugMicrotest, Log_VersionLineHasNoHostPrefix) {
  g_env["NCCL_DEBUG"] = "VERSION";
  Init();
  ncclDebugLog(NCCL_LOG_VERSION, NCCL_ALL, "f", 1, "RCCL 2.x");
  EXPECT_EQ(Output(), "RCCL 2.x\n");
}

TEST_F(DebugMicrotest, Log_ErrorLineIsFatal) {
  Init();
  ncclDebugLog(NCCL_LOG_ERROR, NCCL_ALL, "f", 1, "boom");
  EXPECT_EQ(Output(), "node1:4242:77 [3] [FATAL ERROR]: boom\n");
}

TEST_F(DebugMicrotest, Log_AbortLineHasNoLevelTag) {
  g_env["NCCL_DEBUG"] = "ABORT";
  Init();
  ncclDebugLog(NCCL_LOG_ABORT, NCCL_INIT, "f", 1, "abort");
  EXPECT_EQ(Output(), "node1:4242:77 abort\n");
  EXPECT_EQ(g_getDeviceCalls, 1);
}

TEST_F(DebugMicrotest, Log_TraceCallSkipsTheDeviceQuery) {
  g_env["NCCL_DEBUG"] = "TRACE";
  g_env["NCCL_DEBUG_SUBSYS"] = "CALL";
  Init();
  ncclDebugLog(NCCL_LOG_TRACE, NCCL_CALL, "f", 1, "ncclAllReduce()");
  EXPECT_EQ(Output(), "node1:4242:77 NCCL CALL ncclAllReduce()\n");
  EXPECT_EQ(g_getDeviceCalls, 0);
}

TEST_F(DebugMicrotest, Log_TraceLineHasMillisecondsSinceInit) {
  g_env["NCCL_DEBUG"] = "TRACE";
  Init();
  ncclDebugLog(NCCL_LOG_TRACE, NCCL_INIT, "t.cc", 9, "step");
  std::string out = Output();
  double ms = -1;
  char tail[64] = {};
  ASSERT_EQ(std::sscanf(out.c_str(), "node1:4242:77 [3] %lf t.cc:9 NCCL TRACE %63[^\n]", &ms, tail), 2) << out;
  EXPECT_GE(ms, 0.0);
  EXPECT_LT(ms, 60000.0);
  EXPECT_STREQ(tail, "step");
}

TEST_F(DebugMicrotest, Log_NoWarnDowngradesWarnToInfoOnItsSubsystem) {
  g_env["NCCL_DEBUG"] = "INFO";
  g_env["NCCL_DEBUG_SUBSYS"] = "NET";
  Init();
  ncclDebugNoWarn = NCCL_NET;
  ncclDebugLog(NCCL_LOG_WARN, NCCL_ALL, "f", 1, "quiet");
  ncclDebugNoWarn = NCCL_GRAPH;
  ncclDebugLog(NCCL_LOG_WARN, NCCL_ALL, "f", 1, "dropped");
  ncclDebugLog(NCCL_LOG_INFO, NCCL_NET, "f", 1, "info keeps its flags");
  ncclDebugNoWarn = 0;
  EXPECT_EQ(Output(), "node1:4242:77 [3] NCCL INFO quiet\nnode1:4242:77 [3] NCCL INFO info keeps its flags\n");
  EXPECT_STREQ(ncclLastError, "");
}

TEST_F(DebugMicrotest, Log_LongMessagesAreTruncatedToTheBuffer) {
  g_env["NCCL_DEBUG"] = "INFO";
  Init();
  std::string big(2000, 'm');
  ncclDebugLog(NCCL_LOG_INFO, NCCL_INIT, "f", 1, "%s", big.c_str());
  std::string out = Output();
  ASSERT_EQ(out.size(), 1024u);
  EXPECT_EQ(out.back(), '\n');
  EXPECT_EQ(out.substr(0, 28), "node1:4242:77 [3] NCCL INFO ");
  EXPECT_EQ(out.find_first_not_of('m', 28), 1023u);
}

TEST_F(DebugMicrotest, Log_LongHostPrefixIsTruncatedToo) {
  g_hostName = std::string(1100, 'h');  // cut to the 1023 chars hostname[] holds
  g_env["NCCL_DEBUG"] = "INFO";
  Init();
  ncclDebugLog(NCCL_LOG_INFO, NCCL_INIT, "f", 1, "lost");
  std::string out = Output();
  ASSERT_EQ(out.size(), 1024u);
  EXPECT_EQ(out, std::string(1023, 'h') + "\n");
}

TEST_F(DebugMicrotest, Timestamp_SubsecondsAreFilledInPerLine) {
  g_env["NCCL_DEBUG"] = "INFO";
  g_env["NCCL_DEBUG_TIMESTAMP_LEVELS"] = "INFO";
  g_env["NCCL_DEBUG_TIMESTAMP_FORMAT"] = "%T.%3f_";
  Init();
  ncclDebugLog(NCCL_LOG_INFO, NCCL_INIT, "f", 1, "a");
  g_now = {61, 9000000};
  ncclDebugLog(NCCL_LOG_INFO, NCCL_INIT, "f", 1, "b");
  EXPECT_EQ(Output(), "00:00:00.123 node1:4242:77 [3] NCCL INFO a\n00:01:01.009 node1:4242:77 [3] NCCL INFO b\n");
}

TEST_F(DebugMicrotest, Timestamp_NineDigitsAreNanoseconds) {
  g_env["NCCL_DEBUG_TIMESTAMP_LEVELS"] = "VERSION";
  g_env["NCCL_DEBUG_TIMESTAMP_FORMAT"] = "%9f ";
  g_env["NCCL_DEBUG"] = "VERSION";
  Init();
  ncclDebugLog(NCCL_LOG_VERSION, NCCL_ALL, "f", 1, "v");
  EXPECT_EQ(Output(), "123456789 v\n");
}

TEST_F(DebugMicrotest, Timestamp_EmptyFormatPrintsNothing) {
  g_env["NCCL_DEBUG"] = "WARN";
  g_env["NCCL_DEBUG_TIMESTAMP_FORMAT"] = "";
  Init();
  ncclDebugLog(NCCL_LOG_WARN, NCCL_ALL, "f", 1, "w");
  EXPECT_EQ(Output(), "\nnode1:4242:77 [3] f:1 NCCL WARN w\n");
}

TEST_F(DebugMicrotest, Timestamp_OverflowingFormatFallsBackToTheDefault) {
  g_env["NCCL_DEBUG"] = "WARN";
  std::string many;
  for (int i = 0; i < 100; ++i) many += "%c";  // ~2400 chars expanded, more than the line buffer
  g_env["NCCL_DEBUG_TIMESTAMP_FORMAT"] = many;
  Init();
  ncclDebugLog(NCCL_LOG_WARN, NCCL_ALL, "f", 1, "w");
  EXPECT_EQ(Output(), "\n[1970-01-01 00:00:00] node1:4242:77 [3] f:1 NCCL WARN w\n");
}

TEST_F(DebugMicrotest, Timestamp_OnlyOnSelectedLevels) {
  g_env["NCCL_DEBUG"] = "INFO";
  g_env["NCCL_DEBUG_TIMESTAMP_LEVELS"] = "^WARN";
  g_env["NCCL_DEBUG_TIMESTAMP_FORMAT"] = "<%S>_";
  Init();
  ncclDebugLog(NCCL_LOG_WARN, NCCL_ALL, "f", 1, "w");
  ncclDebugLog(NCCL_LOG_INFO, NCCL_INIT, "f", 1, "i");
  EXPECT_EQ(Output(), "\nnode1:4242:77 [3] f:1 NCCL WARN w\n<00> node1:4242:77 [3] NCCL INFO i\n");
}

TEST_F(DebugMicrotest, Log_WarnEnablesInfoWhenRequested) {
  g_env["NCCL_DEBUG"] = "WARN";
  g_env["NCCL_WARN_ENABLE_DEBUG_INFO"] = "1";
  g_env["NCCL_DEBUG_TIMESTAMP_LEVELS"] = "";
  Init();
  ncclDebugLog(NCCL_LOG_INFO, NCCL_INIT, "f", 1, "before");
  ncclDebugLog(NCCL_LOG_WARN, NCCL_ALL, "f", 1, "w");
  ncclDebugLog(NCCL_LOG_INFO, NCCL_INIT, "f", 1, "after");
  EXPECT_EQ(Output(), "\nnode1:4242:77 [3] f:1 NCCL WARN w\nnode1:4242:77 [3] NCCL INFO after\n");
  EXPECT_EQ(ncclDebugLevelMask, kError | kVersion | kWarn | kAttn | kInfo);
}

TEST_F(DebugMicrotest, LazyInit_FirstLogReadsTheEnvironment) {
  g_env["NCCL_DEBUG"] = "INFO";
  g_env["NCCL_DEBUG_FILE"] = Dir() + "/lazy";
  ncclDebugLevelMask = NCCL_DEBUG_LEVEL_MASK_UNINITIALIZED;
  ncclDebugMask = 0;
  ncclDebugLog(NCCL_LOG_INFO, NCCL_INIT, "f", 1, "first");
  EXPECT_EQ(ncclDebugMask, kDefaultSubsys);
  EXPECT_EQ(ReadFile(Dir() + "/lazy"), "node1:4242:77 [3] NCCL INFO first\n");
}

TEST_F(DebugMicrotest, LazyInit_RechecksAfterInit) {
  g_env["NCCL_DEBUG"] = "INFO";
  g_env["NCCL_DEBUG_FILE"] = Dir() + "/lazy";
  ncclDebugLevelMask = NCCL_DEBUG_LEVEL_MASK_UNINITIALIZED;
  ncclDebugLog(NCCL_LOG_INFO, NCCL_NET, "f", 1, "net");
  EXPECT_EQ(ncclDebugLevelMask, kError | kVersion | kWarn | kAttn | kInfo);
  EXPECT_EQ(ReadFile(Dir() + "/lazy"), "");
}

TEST_F(DebugMicrotest, Reset_ClosesTheOldFileAndRereadsTheEnvironment) {
  g_env["NCCL_DEBUG"] = "INFO";
  g_env["NCCL_DEBUG_FILE"] = Dir() + "/one";
  Init();
  ncclDebugLog(NCCL_LOG_INFO, NCCL_INIT, "f", 1, "1");
  ncclResetDebugInitInternal();
  EXPECT_EQ(ncclDebugLevelMask, NCCL_DEBUG_LEVEL_MASK_RESET_TRIGGERED);
  g_env["NCCL_DEBUG_FILE"] = Dir() + "/two";
  g_env["NCCL_DEBUG_SUBSYS"] = "NET";
  ncclDebugLog(NCCL_LOG_INFO, NCCL_NET, "f", 1, "2");
  EXPECT_EQ(ncclDebugMask, (uint64_t)NCCL_NET);
  EXPECT_EQ(ReadFile(Dir() + "/one"), "node1:4242:77 [3] NCCL INFO 1\n");
  EXPECT_EQ(ReadFile(Dir() + "/two"), "node1:4242:77 [3] NCCL INFO 2\n");
}

TEST_F(DebugMicrotest, Reset_InitWithoutResetKeepsTheFileOpen) {
  ncclDebugFile = capture_;
  ncclDebugLevelMask = NCCL_DEBUG_LEVEL_MASK_UNINITIALIZED;
  ncclDebugInit();
  EXPECT_EQ(ncclDebugFile, capture_);
}

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
// Not pncclResetDebugInit: the in-build clang does not emit its alias, so a call there fails to link.
TEST_F(DebugMicrotest, Reset_DeprecatedPublicEntryPointStillResets) {
  Init();
  ncclResetDebugInit();
  EXPECT_EQ(ncclDebugLevelMask, NCCL_DEBUG_LEVEL_MASK_RESET_TRIGGERED);
}
#pragma clang diagnostic pop

TEST_F(DebugMicrotest, Env_WithoutTheEnvPluginReadsTheProcessEnvironment) {
  using RcclUnitTesting::ScopedEnv;
  ScopedEnv debug("NCCL_DEBUG", "WARN"), levels("NCCL_DEBUG_LEVELS", nullptr), subsys("NCCL_DEBUG_SUBSYS", nullptr),
    warnInfo("NCCL_WARN_ENABLE_DEBUG_INFO", nullptr), tsLevels("NCCL_DEBUG_TIMESTAMP_LEVELS", nullptr),
    tsFormat("NCCL_DEBUG_TIMESTAMP_FORMAT", nullptr), file("NCCL_DEBUG_FILE", nullptr);
  g_envPluginReady = false;
  g_env["NCCL_DEBUG"] = "TRACE";  // ignored
  g_env["NCCL_DEBUG_SUBSYS"] = "NET";  // ignored

  Init();

  EXPECT_EQ(ncclDebugLevelMask, kError | kVersion | kWarn);
  EXPECT_EQ(ncclDebugMask, kDefaultSubsys);
}

TEST_F(DebugMicrotest, Log_ConcurrentLinesAreNotInterleaved) {
  g_env["NCCL_DEBUG"] = "INFO";
  Init();
  std::vector<std::thread> threads;
  for (int t = 0; t < 4; ++t) {
    threads.emplace_back([t] {
      for (int i = 0; i < 200; ++i) ncclDebugLog(NCCL_LOG_INFO, NCCL_INIT, "f", 1, "t%d-%s", t, "payload-payload");
    });
  }
  for (auto& th : threads) th.join();
  std::istringstream lines(Output());
  std::string line;
  int n = 0;
  while (std::getline(lines, line)) {
    ++n;
    ASSERT_EQ(line.rfind("node1:4242:77 [3] NCCL INFO t", 0), 0u) << line;
    ASSERT_EQ(line.substr(line.size() - 16), "-payload-payload") << line;
  }
  EXPECT_EQ(n, 800);
}

// A writer waits for ncclDebugLock even when it did not take it for ncclLastError.
TEST_F(DebugMicrotest, Log_WritesUnderTheDebugLock) {
  g_env["NCCL_DEBUG"] = "INFO";
  Init();
  pthread_mutex_lock(&ncclDebugLock);
  std::thread writer([] { ncclDebugLog(NCCL_LOG_INFO, NCCL_INIT, "f", 1, "late"); });
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  EXPECT_EQ(Output(), "");
  pthread_mutex_unlock(&ncclDebugLock);
  writer.join();
  EXPECT_EQ(Output(), "node1:4242:77 [3] NCCL INFO late\n");
}

TEST_F(DebugMicrotest, Params_DebugSettingsAreRegistered) {
  std::set<std::string> keys(g_registeredKeys.begin(), g_registeredKeys.end());
  for (const char* key : {"NCCL_DEBUG", "NCCL_DEBUG_LEVELS", "NCCL_DEBUG_SUBSYS", "NCCL_WARN_ENABLE_DEBUG_INFO",
                          "NCCL_DEBUG_TIMESTAMP_LEVELS", "NCCL_DEBUG_TIMESTAMP_FORMAT", "NCCL_DEBUG_FILE"}) {
    EXPECT_EQ(keys.count(key), 1u) << key;
  }
}

std::string ThreadName(pthread_t t) {
  char name[NCCL_THREAD_NAMELEN] = {};
  pthread_getname_np(t, name, sizeof(name));
  return name;
}

class DebugThreadNameMicrotest : public DebugMicrotest {
 protected:
  // Runs a thread that waits until it has been named, and returns its name.
  template <typename Namer>
  std::string NameOf(Namer namer) {
    std::mutex m;
    std::condition_variable cv;
    bool named = false;
    std::string result;
    std::thread th([&] {
      std::unique_lock<std::mutex> lock(m);
      cv.wait(lock, [&] { return named; });
      result = ThreadName(pthread_self());
    });
    namer(th);
    {
      std::lock_guard<std::mutex> lock(m);
      named = true;
    }
    cv.notify_one();
    th.join();
    return result;
  }
  std::string initialName_ = ThreadName(pthread_self());
};

TEST_F(DebugThreadNameMicrotest, StdThread_NamedOnlyWhenEnabled) {
  g_setThreadName = 1;
  EXPECT_EQ(NameOf([](std::thread& t) { ncclSetThreadName(t, "NCCL Svc %d", 7); }), "NCCL Svc 7");
  g_setThreadName = 0;
  EXPECT_EQ(NameOf([](std::thread& t) { ncclSetThreadName(t, "NCCL Svc %d", 7); }), initialName_);
}

TEST_F(DebugThreadNameMicrotest, StdThread_NameIsCutToFifteenChars) {
  g_setThreadName = 1;
  EXPECT_EQ(NameOf([](std::thread& t) { ncclSetThreadName(t, "%s", "0123456789abcdefXYZ"); }), "0123456789abcde");
}

TEST_F(DebugThreadNameMicrotest, Pthread_NamedOnlyWhenEnabled) {
  g_setThreadName = 1;
  EXPECT_EQ(NameOf([](std::thread& t) { ncclSetThreadName(t.native_handle(), "IB %s", "async"); }), "IB async");
  g_setThreadName = 0;
  EXPECT_EQ(NameOf([](std::thread& t) { ncclSetThreadName(t.native_handle(), "IB %s", "async"); }), initialName_);
}

// For values other than 0 and 1, only the std::thread overload names the thread.
TEST_F(DebugThreadNameMicrotest, SetThreadName_NonBinarySettingNamesOnlyStdThreads) {
  g_setThreadName = 2;
  EXPECT_EQ(NameOf([](std::thread& t) { ncclSetThreadName(t, "std"); }), "std");
  EXPECT_EQ(NameOf([](std::thread& t) { ncclSetThreadName(t.native_handle(), "posix"); }), initialName_);
}

}  // namespace
