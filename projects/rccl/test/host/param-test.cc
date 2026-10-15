/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for the param runtime: ncclParam<T> (include/param/param.h), src/param/param.cc,
// src/param/c_api.cc and the legacy env loading in src/misc/param.cc. The env plugin, ncclOsSetEnv and
// logging are seamed in this TU; NCCL_NO_CACHE always lists kUncachedKey and kUncachedLegacyKey.

#include <gtest/gtest.h>

#include <unistd.h>

#include <cerrno>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "../common/LogCapture.hpp"
#include "../common/ProcessIsolatedTestRunner.hpp"
#include "nccl.h"
#include "debug.h"
#include "env.h"
#include "os.h"

uint32_t ncclDebugLevelMask = NCCL_DEBUG_LEVEL_MASK_UNINITIALIZED;  // log everything
uint64_t ncclDebugMask = ~uint64_t(0);
thread_local int ncclDebugNoWarn = 0;

namespace {
constexpr char kUncachedKey[] = "PARAM_TEST_UNCACHED";
constexpr char kUncachedLegacyKey[] = "PARAM_TEST_UNCACHED_LEGACY";

struct LoggedMessage {
  ncclDebugLogLevel level;
  std::string text;
};
std::vector<LoggedMessage> g_logged;

bool Logged(ncclDebugLogLevel level, const std::string& needle) {
  for (const auto& m : g_logged)
    if (m.level == level && m.text.find(needle) != std::string::npos) return true;
  return false;
}

std::map<std::string, std::string> g_pluginEnv;
std::vector<std::string> g_pluginLookups;
int Lookups(const std::string& key) {
  int n = 0;
  for (const auto& k : g_pluginLookups) n += k == key;
  return n;
}
bool g_pluginInitialized = false;
int g_initEnvCalls = 0;
std::vector<std::pair<std::string, std::string>> g_setEnv;
std::vector<std::string> g_opened;
std::map<std::string, std::string> g_fopenRedirect;

void ResetSeams() {
  // Kept across every reset: the first ncclParamIsCacheDisabled() call latches it for the process.
  g_pluginEnv = {{"NCCL_NO_CACHE", std::string(kUncachedKey) + ", " + kUncachedLegacyKey}};
  g_pluginLookups.clear();
  g_pluginInitialized = false;
  g_initEnvCalls = 0;
  g_setEnv.clear();
  g_opened.clear();
  g_fopenRedirect.clear();
  g_logged.clear();
}
}  // namespace

void ncclDebugLog(ncclDebugLogLevel level, unsigned long /*flags*/, const char* /*filefunc*/, int /*line*/,
                  const char* fmt, ...) {
  char buf[512];
  va_list ap;
  va_start(ap, fmt);
  std::vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  g_logged.push_back({level, buf});
}

ncclResult_t ncclInitEnv() {
  ++g_initEnvCalls;
  return ncclSuccess;
}

const char* ncclEnvPluginGetEnv(const char* name) {
  g_pluginLookups.push_back(name);
  auto it = g_pluginEnv.find(name);
  return it == g_pluginEnv.end() ? nullptr : it->second.c_str();
}

bool ncclEnvPluginInitialized() { return g_pluginInitialized; }

void ncclOsSetEnv(const char* name, const char* value) { g_setEnv.emplace_back(name, value); }

static FILE* ParamTestFopen(const char* path, const char* mode) {
  g_opened.push_back(path);
  auto it = g_fopenRedirect.find(path);
  return std::fopen(it == g_fopenRedirect.end() ? path : it->second.c_str(), mode);
}

#include PARAM_REGISTRY_CC_PATH
#include PARAM_CC_PATH
#include C_API_CC_PATH
#define fopen ParamTestFopen
#include MISC_PARAM_CC_PATH
#undef fopen

DEFINE_NCCL_PARAM(paramTestDefined, int32_t, PARAM_TEST_DEFINED, 11, NCCL_PARAM_FLAG_PUBLISHED, NCCL_PARAM_DEFAULT,
                  "Defined through the macro");

namespace {

using RcclUnitTesting::CaptureStdout;
using Pairs = std::vector<std::pair<std::string, std::string>>;

class ParamMicrotest : public ::testing::Test {
 protected:
  void SetUp() override { ResetSeams(); }

  void TearDown() override {
    for (const auto& key : keys_) {
      ncclParamRegistry::remove(*key);
      unsetenv(key->c_str());
    }
    for (const auto& path : files_) std::remove(path.c_str());
    ResetSeams();
  }

  // Unique on every call, so repeats never collide in the registry or in c_api's once-per-key notes.
  const char* Key() {
    static int serial = 0;
    keys_.push_back(std::make_unique<std::string>(
        std::string("PARAM_TEST_") + ::testing::UnitTest::GetInstance()->current_test_info()->name() + "_" +
        std::to_string(serial++)));
    return keys_.back()->c_str();
  }

  template <typename T>
  ncclParam<T>& Make(T def, uint64_t flags = NCCL_PARAM_FLAG_NONE, ncclParamParser<T> parser = {},
                     const char* key = nullptr) {
    if (key == nullptr) key = Key();
    else keys_.push_back(std::make_unique<std::string>(key));
    auto param = std::make_shared<ncclParam<T>>(key, def, std::move(parser), "T", flags, "test param");
    params_.push_back(param);
    return *param;
  }

  std::string WriteFile(const std::string& contents) {
    std::string path = ::testing::TempDir() + "param_test_" + std::to_string(getpid()) + "_" +
                       std::to_string(files_.size()) + ".conf";
    FILE* f = std::fopen(path.c_str(), "w");
    std::fputs(contents.c_str(), f);
    std::fclose(f);
    files_.push_back(path);
    return path;
  }

  std::vector<std::unique_ptr<std::string>> keys_;
  std::vector<std::shared_ptr<void>> params_;
  std::vector<std::string> files_;
};

// ncclParam<T> (include/param/param.h)

TEST_F(ParamMicrotest, Ctor_FillsInfoAndRegistersItself) {
  auto& p = Make<int32_t>(7, NCCL_PARAM_FLAG_PUBLISHED | NCCL_PARAM_FLAG_CACHED);
  EXPECT_EQ(NCCL_PARAM_TYPE_I32, p.info.typeId);
  EXPECT_EQ(static_cast<uint64_t>(NCCL_PARAM_FLAG_PUBLISHED | NCCL_PARAM_FLAG_CACHED), p.info.flags);
  EXPECT_STREQ("T", p.info.typeStr);
  EXPECT_STREQ("test param", p.info.desc);
  EXPECT_EQ(7, p.defaultValue);
  EXPECT_EQ(7, p.value);
  EXPECT_FALSE(p.loaded.load());

  auto* entry = ncclParamRegistry::find(p.info.key);
  ASSERT_NE(nullptr, entry);
  EXPECT_EQ(&p, entry->param);
  EXPECT_EQ(p.info.flags, entry->info.flags);
}

TEST_F(ParamMicrotest, Ctor_DefaultsTheOptionalArgumentsAndFillsANullParser) {
  const char* key = Key();
  ncclParam<int32_t> p(key, 3);
  EXPECT_STREQ("", p.info.typeStr);
  EXPECT_STREQ("", p.info.desc);
  EXPECT_EQ(static_cast<uint64_t>(NCCL_PARAM_FLAG_NONE), p.info.flags);
  ASSERT_TRUE(static_cast<bool>(p.parser));
  EXPECT_EQ("Integer", p.parser.desc);
}

TEST_F(ParamMicrotest, Ctor_KeepsACustomParser) {
  auto& p = Make<int32_t>(5, NCCL_PARAM_FLAG_NONE, ncclParamBounded<int32_t>(0, 10));
  EXPECT_EQ("Integer in range [0, 10]", p.parser.desc);
}

TEST_F(ParamMicrotest, DefineMacro_DefinesTheKeyAndARegisteredParam) {
  EXPECT_STREQ("PARAM_TEST_DEFINED", paramTestDefinedKey);
  EXPECT_STREQ("int32_t", paramTestDefined.info.typeStr);
  EXPECT_STREQ("Defined through the macro", paramTestDefined.info.desc);
  EXPECT_EQ(11, paramTestDefined.defaultValue);
  auto* entry = ncclParamRegistry::find("PARAM_TEST_DEFINED");
  ASSERT_NE(nullptr, entry);
  EXPECT_EQ(&paramTestDefined, entry->param);
}

TEST_F(ParamMicrotest, Call_ReadsTheEnvPluginAfterInitializingIt) {
  auto& p = Make<int32_t>(7);
  g_pluginEnv[p.info.key] = "42";
  EXPECT_EQ(42, p());
  EXPECT_EQ(1, g_initEnvCalls);
  EXPECT_STREQ("EnvPlugin", p.srcStr);
  EXPECT_TRUE(p.loaded.load());
}

TEST_F(ParamMicrotest, Call_UnsetUnparsableOrInvalidValuesFallBackToTheDefault) {
  auto& p = Make<int32_t>(7, NCCL_PARAM_FLAG_NONE, ncclParamBounded<int32_t>(0, 10));
  for (const char* bad : {"abc", "11"}) {
    g_pluginEnv[p.info.key] = "4";
    EXPECT_EQ(4, p());
    g_pluginEnv[p.info.key] = bad;
    EXPECT_EQ(7, p()) << bad;
    EXPECT_STREQ("Default", p.srcStr) << bad;
  }
  g_pluginEnv[p.info.key] = "4";
  EXPECT_EQ(4, p());
  g_pluginEnv.erase(p.info.key);
  EXPECT_EQ(7, p());
  EXPECT_STREQ("Default", p.srcStr);
}

TEST_F(ParamMicrotest, Call_UncachedParamRereadsEveryCall) {
  auto& p = Make<int32_t>(7);
  g_pluginEnv[p.info.key] = "1";
  EXPECT_EQ(1, p());
  g_pluginEnv[p.info.key] = "2";
  EXPECT_EQ(2, p());
  EXPECT_EQ(2, g_initEnvCalls);
}

TEST_F(ParamMicrotest, Call_CachedParamLoadsOnce) {
  auto& p = Make<int32_t>(7, NCCL_PARAM_FLAG_CACHED);
  g_pluginEnv[p.info.key] = "1";
  EXPECT_EQ(1, p());
  g_pluginEnv[p.info.key] = "2";
  EXPECT_EQ(1, p());
  // Not g_initEnvCalls: the process's first NCCL_NO_CACHE load adds one.
  EXPECT_EQ(1, Lookups(p.info.key));
}

TEST_F(ParamMicrotest, Call_CachedParamListedInNoCacheRereadsEveryCall) {
  auto& p = Make<int32_t>(7, NCCL_PARAM_FLAG_CACHED, {}, kUncachedKey);
  g_pluginEnv[kUncachedKey] = "1";
  EXPECT_EQ(1, p());
  g_pluginEnv[kUncachedKey] = "2";
  EXPECT_EQ(2, p());
}

TEST_F(ParamMicrotest, Call_NoEnvPluginInitReadsGetenvUntilThePluginIsUp) {
  auto& p = Make<int32_t>(7, NCCL_PARAM_FLAG_NO_ENVPLUGIN_INIT);
  setenv(p.info.key, "3", 1);
  g_pluginEnv[p.info.key] = "4";
  EXPECT_EQ(3, p());
  g_pluginInitialized = true;
  EXPECT_EQ(4, p());
  EXPECT_EQ(0, g_initEnvCalls);
}

TEST_F(ParamMicrotest, GetRawData_RejectsBadArgumentsAndShortBuffers) {
  auto& p = Make<int32_t>(7);
  int32_t out = 0;
  int len = 99;
  EXPECT_EQ(ncclInvalidArgument, p.getRawData(nullptr, sizeof(out), &len));
  EXPECT_EQ(ncclInvalidArgument, p.getRawData(&out, sizeof(out), nullptr));
  EXPECT_EQ(ncclInvalidArgument, p.getRawData(&out, 0, &len));
  EXPECT_EQ(99, len);
  EXPECT_EQ(ncclInvalidArgument, p.getRawData(&out, sizeof(out) - 1, &len));
  EXPECT_EQ(0, len);
  EXPECT_EQ(0, out);
}

TEST_F(ParamMicrotest, GetRawData_CopiesTheLoadedValue) {
  auto& p = Make<int64_t>(7);
  g_pluginEnv[p.info.key] = "-5000000000";
  int64_t out = 0;
  int len = 0;
  EXPECT_EQ(ncclSuccess, p.getRawData(&out, sizeof(out), &len));
  EXPECT_EQ(-5000000000LL, out);
  EXPECT_EQ(static_cast<int>(sizeof(out)), len);
}

TEST_F(ParamMicrotest, ToString_UsesTheParser) {
  auto& p = Make<bool>(false);
  g_pluginEnv[p.info.key] = "t";
  EXPECT_EQ("TRUE", p.toString());
}

TEST_F(ParamMicrotest, Dump_ListsKeyFlagsValueSourceDefaultAndAcceptedValues) {
  auto& p = Make<int32_t>(7, NCCL_PARAM_FLAG_PUBLISHED | NCCL_PARAM_FLAG_CACHED);
  g_pluginEnv[p.info.key] = "5";
  EXPECT_EQ(std::string(p.info.key) +
                " (T) [Cached] test param\n"
                "    Current value=5 set_by=EnvPlugin default=7\n"
                "    Accepted value: Integer\n",
            p.dump());
}

TEST_F(ParamMicrotest, Dump_RedactsSensitiveValuesButStillShowsUnset) {
  auto& secret = Make<int32_t>(7, NCCL_PARAM_FLAG_PUBLISHED | NCCL_PARAM_FLAG_SENSITIVE);
  g_pluginEnv[secret.info.key] = "5";
  EXPECT_EQ(std::string(secret.info.key) +
                " (T) [Sensitive] test param\n"
                "    Current value=<redacted> set_by=EnvPlugin default=<redacted>\n"
                "    Accepted value: Integer\n",
            secret.dump());

  auto& unset = Make<const char*>(nullptr, NCCL_PARAM_FLAG_SENSITIVE);
  EXPECT_EQ(std::string(unset.info.key) +
                " (T) [Private, Sensitive] test param\n"
                "    Current value=<unset> set_by=Default default=<unset>\n"
                "    Accepted value: String\n",
            unset.dump());
}

// ncclParam<const char*> specializations

TEST_F(ParamMicrotest, Cstr_CallReturnsNullOrACopyOfTheDefault) {
  auto& none = Make<const char*>(nullptr);
  EXPECT_EQ(nullptr, none());
  static const char kDefault[] = "dflt";
  auto& dflt = Make<const char*>(kDefault);
  const char* got = dflt();
  EXPECT_STREQ("dflt", got);
  EXPECT_NE(kDefault, got);
  EXPECT_STREQ("Default", dflt.srcStr);
}

TEST_F(ParamMicrotest, Cstr_CopiesThePluginValue) {
  auto& p = Make<const char*>("dflt", NCCL_PARAM_FLAG_CACHED);
  g_pluginEnv[p.info.key] = "first";
  EXPECT_STREQ("first", p());
  EXPECT_STREQ("EnvPlugin", p.srcStr);
  // Rewrites the plugin's buffer in place; the cached value must not alias it.
  g_pluginEnv[p.info.key] = "other";
  EXPECT_STREQ("first", p());
  EXPECT_EQ("first", p.toString());
}

TEST_F(ParamMicrotest, Cstr_UnsettingClearsTheStoredString) {
  auto& p = Make<const char*>(nullptr);
  g_pluginEnv[p.info.key] = "abc";
  EXPECT_STREQ("abc", p());
  g_pluginEnv.erase(p.info.key);
  EXPECT_EQ(nullptr, p());
  char buf[8] = {'x'};
  int len = 0;
  EXPECT_EQ(ncclSuccess, p.getRawData(buf, sizeof(buf), &len));
  EXPECT_EQ(1, len);
  EXPECT_EQ('\0', buf[0]);
}

TEST_F(ParamMicrotest, Cstr_GetRawDataCopiesTheTerminatorOrRejects) {
  auto& p = Make<const char*>("abc");
  char buf[8] = {};
  int len = 99;
  EXPECT_EQ(ncclInvalidArgument, p.getRawData(nullptr, 4, &len));
  EXPECT_EQ(ncclInvalidArgument, p.getRawData(buf, 4, nullptr));
  EXPECT_EQ(ncclInvalidArgument, p.getRawData(buf, 0, &len));
  EXPECT_EQ(99, len);
  EXPECT_EQ(ncclInvalidArgument, p.getRawData(buf, 3, &len));
  EXPECT_EQ(0, len);
  std::memset(buf, 'x', sizeof(buf));
  EXPECT_EQ(ncclSuccess, p.getRawData(buf, 4, &len));
  EXPECT_EQ(4, len);
  EXPECT_STREQ("abc", buf);
}

// src/param/param.cc

TEST_F(ParamMicrotest, Registers_TheParamSystemsOwnParams) {
  auto* dumpAll = ncclParamRegistry::find("NCCL_PARAM_DUMP_ALL");
  ASSERT_NE(nullptr, dumpAll);
  EXPECT_EQ(NCCL_PARAM_TYPE_BOOL, dumpAll->info.typeId);
  EXPECT_EQ(static_cast<uint64_t>(NCCL_PARAM_FLAG_NONE), dumpAll->info.flags);
  EXPECT_EQ("FALSE", dumpAll->param->toString());

  auto* noCache = ncclParamRegistry::find("NCCL_NO_CACHE");
  ASSERT_NE(nullptr, noCache);
  EXPECT_EQ(NCCL_PARAM_TYPE_CSTR, noCache->info.typeId);
  EXPECT_EQ(static_cast<uint64_t>(NCCL_PARAM_FLAG_CACHED), noCache->info.flags);
}

class ParamNoCacheMicrotest : public ParamMicrotest {};

TEST_F(ParamNoCacheMicrotest, ListedKeysAreDisabledWithAnInfo) {
  EXPECT_TRUE(ncclParamIsCacheDisabled(kUncachedKey));
  EXPECT_TRUE(Logged(NCCL_LOG_INFO, std::string("Disabling caching for environment variable ") + kUncachedKey));
  // Listed after ", ", so the entries are trimmed.
  EXPECT_TRUE(ncclParamIsCacheDisabled(kUncachedLegacyKey));
  g_logged.clear();
  EXPECT_FALSE(ncclParamIsCacheDisabled("PARAM_TEST_NOT_LISTED"));
  EXPECT_FALSE(ncclParamIsCacheDisabled("PARAM_TEST"));
  EXPECT_TRUE(g_logged.empty());
}

TEST_F(ParamNoCacheMicrotest, NoCacheItselfIsNeverDisabled) {
  EXPECT_FALSE(ncclParamIsCacheDisabled("NCCL_NO_CACHE"));
}

class ParamNoCacheMicrotestIsolated : public ParamMicrotest {};

TEST_F(ParamNoCacheMicrotestIsolated, AllDisablesEveryKeyButNoCache) {
  RUN_ISOLATED_TEST("Param_NoCache_All", []() {
    g_pluginEnv["NCCL_NO_CACHE"] = "FOO,ALL";
    EXPECT_TRUE(ncclParamIsCacheDisabled("PARAM_TEST_ANY"));
    EXPECT_TRUE(ncclParamIsCacheDisabled("FOO"));
    EXPECT_FALSE(ncclParamIsCacheDisabled("NCCL_NO_CACHE"));
  });
}

TEST_F(ParamNoCacheMicrotestIsolated, UnsetDisablesNothing) {
  RUN_ISOLATED_TEST("Param_NoCache_Unset", []() {
    g_pluginEnv.erase("NCCL_NO_CACHE");
    EXPECT_FALSE(ncclParamIsCacheDisabled(kUncachedKey));
    EXPECT_FALSE(ncclParamIsCacheDisabled("ALL"));
  });
}

class ParamEnvPluginGetMicrotest : public ParamMicrotest {};

TEST_F(ParamEnvPluginGetMicrotest, InitInitializesThePluginFirst) {
  g_pluginEnv["PARAM_TEST_GET"] = "v";
  EXPECT_STREQ("v", ncclParamEnvPluginGet("PARAM_TEST_GET", true));
  EXPECT_EQ(1, g_initEnvCalls);
  EXPECT_EQ(nullptr, ncclParamEnvPluginGet("PARAM_TEST_GET_UNSET", true));
}

TEST_F(ParamEnvPluginGetMicrotest, NoInitUsesGetenvUntilThePluginIsUp) {
  const char* key = Key();
  setenv(key, "env", 1);
  g_pluginEnv[key] = "plugin";
  EXPECT_STREQ("env", ncclParamEnvPluginGet(key, false));
  EXPECT_TRUE(g_pluginLookups.empty());
  g_pluginInitialized = true;
  EXPECT_STREQ("plugin", ncclParamEnvPluginGet(key, false));
  EXPECT_EQ(0, g_initEnvCalls);
}

// src/param/c_api.cc

class ParamCApiMicrotest : public ParamMicrotest {
 protected:
  ncclParamHandle_t Bind(const char* key) {
    ncclParamHandle_t h = nullptr;
    EXPECT_EQ(ncclSuccess, ncclParamBind(&h, key));
    return h;
  }

  // The getter for T returns the default of a T param, and rejects a param of any other type.
  template <typename T>
  void ExpectGetter(ncclResult_t (*get)(ncclParamHandle_t, T*), T def) {
    auto& p = Make<T>(def);
    T out{};
    EXPECT_EQ(ncclSuccess, get(Bind(p.info.key), &out));
    EXPECT_EQ(def, out);
    EXPECT_EQ(ncclInvalidArgument, get(nullptr, &out));
    EXPECT_EQ(ncclInvalidArgument, get(Bind(p.info.key), nullptr));

    auto& other = Make<const char*>("x");
    out = T{};
    g_logged.clear();
    EXPECT_EQ(ncclInvalidArgument, get(Bind(other.info.key), &out));
    EXPECT_EQ(T{}, out);
    EXPECT_TRUE(Logged(NCCL_LOG_INFO, std::string("type mismatch for key \"") + other.info.key + "\""));
  }

  std::vector<std::string> AllKeys() {
    const char** table = nullptr;
    int len = -1;
    EXPECT_EQ(ncclSuccess, ncclParamGetAllParameterKeys(&table, &len));
    return std::vector<std::string>(table, table + len);
  }

  static bool Has(const std::vector<std::string>& keys, const char* key) {
    for (const auto& k : keys)
      if (k == key) return true;
    return false;
  }
};

TEST_F(ParamCApiMicrotest, Bind_ReturnsTheRegistryEntryAsTheHandle) {
  auto& p = Make<int32_t>(1, NCCL_PARAM_FLAG_PUBLISHED);
  ncclParamHandle_t h = nullptr;
  EXPECT_EQ(ncclSuccess, ncclParamBind(&h, p.info.key));
  EXPECT_EQ(reinterpret_cast<ncclParamHandle_t>(ncclParamRegistry::find(p.info.key)), h);
}

TEST_F(ParamCApiMicrotest, Bind_RejectsNullArgumentsAndUnknownKeys) {
  auto& p = Make<int32_t>(1);
  ncclParamHandle_t h = nullptr;
  EXPECT_EQ(ncclInvalidArgument, ncclParamBind(nullptr, p.info.key));
  EXPECT_EQ(ncclInvalidArgument, ncclParamBind(&h, nullptr));
  EXPECT_EQ(ncclInvalidArgument, ncclParamBind(&h, "PARAM_TEST_NEVER_REGISTERED"));
  EXPECT_EQ(nullptr, h);
  EXPECT_TRUE(Logged(NCCL_LOG_INFO, "PARAM: key \"PARAM_TEST_NEVER_REGISTERED\" not found in registry"));
}

TEST_F(ParamCApiMicrotest, Bind_NotesUnusedDeprecatedAndPrivateParamsOncePerKey) {
  auto& unused = Make<int32_t>(1, NCCL_PARAM_FLAG_PUBLISHED | NCCL_PARAM_FLAG_UNUSED | NCCL_PARAM_FLAG_DEPRECATED);
  Bind(unused.info.key);
  EXPECT_TRUE(Logged(NCCL_LOG_INFO, std::string("ENV ") + unused.info.key + " is unused"));
  EXPECT_FALSE(Logged(NCCL_LOG_INFO, "deprecated"));
  EXPECT_FALSE(Logged(NCCL_LOG_INFO, "private"));

  auto& deprecated = Make<int32_t>(1, NCCL_PARAM_FLAG_PUBLISHED | NCCL_PARAM_FLAG_DEPRECATED);
  Bind(deprecated.info.key);
  EXPECT_TRUE(Logged(NCCL_LOG_INFO, std::string("ENV ") + deprecated.info.key + " is deprecated"));

  auto& priv = Make<int32_t>(1);
  g_logged.clear();
  Bind(priv.info.key);
  EXPECT_TRUE(Logged(NCCL_LOG_INFO, std::string("ENV: ") + priv.info.key + " is private"));
  EXPECT_EQ(1u, g_logged.size());

  g_logged.clear();
  Bind(unused.info.key);
  Bind(priv.info.key);
  EXPECT_TRUE(g_logged.empty());
}

TEST_F(ParamCApiMicrotest, TypedGetters_ReadTheirOwnTypeOnly) {
  ExpectGetter<int8_t>(ncclParamGetI8, -5);
  ExpectGetter<int16_t>(ncclParamGetI16, -300);
  ExpectGetter<int32_t>(ncclParamGetI32, -70000);
  ExpectGetter<int64_t>(ncclParamGetI64, -5000000000LL);
  ExpectGetter<uint8_t>(ncclParamGetU8, 200);
  ExpectGetter<uint16_t>(ncclParamGetU16, 60000);
  ExpectGetter<uint32_t>(ncclParamGetU32, 4000000000u);
  ExpectGetter<uint64_t>(ncclParamGetU64, (1ull << 40) + 3);
}

TEST_F(ParamCApiMicrotest, GetStr_ReturnsTheStringValue) {
  auto& p = Make<const char*>(nullptr);
  g_pluginEnv[p.info.key] = "hello";
  const char* out = nullptr;
  EXPECT_EQ(ncclSuccess, ncclParamGetStr(Bind(p.info.key), &out));
  EXPECT_STREQ("hello", out);
  EXPECT_EQ(ncclInvalidArgument, ncclParamGetStr(nullptr, &out));
  EXPECT_EQ(ncclInvalidArgument, ncclParamGetStr(Bind(p.info.key), nullptr));
}

TEST_F(ParamCApiMicrotest, GetStr_RejectsANonStringParam) {
  auto& p = Make<int32_t>(1);
  const char* out = nullptr;
  EXPECT_EQ(ncclInvalidArgument, ncclParamGetStr(Bind(p.info.key), &out));
  EXPECT_EQ(nullptr, out);
  EXPECT_TRUE(Logged(NCCL_LOG_INFO, std::string("type mismatch for key \"") + p.info.key + "\""));
}

TEST_F(ParamCApiMicrotest, Get_ForwardsToGetRawData) {
  auto& p = Make<const char*>("abc");
  ncclParamHandle_t h = Bind(p.info.key);
  char buf[8] = {};
  int len = 99;
  EXPECT_EQ(ncclInvalidArgument, ncclParamGet(nullptr, buf, sizeof(buf), &len));
  EXPECT_EQ(ncclInvalidArgument, ncclParamGet(h, nullptr, sizeof(buf), &len));
  EXPECT_EQ(ncclInvalidArgument, ncclParamGet(h, buf, 3, &len));
  EXPECT_EQ(0, len);
  EXPECT_EQ(ncclSuccess, ncclParamGet(h, buf, sizeof(buf), &len));
  EXPECT_EQ(4, len);
  EXPECT_STREQ("abc", buf);
}

TEST_F(ParamCApiMicrotest, GetAllParameterKeys_ListsPublishedKeysUnlessDumpAll) {
  auto& pub = Make<int32_t>(1, NCCL_PARAM_FLAG_PUBLISHED);
  auto& priv = Make<int32_t>(1);
  auto keys = AllKeys();
  EXPECT_TRUE(Has(keys, pub.info.key));
  EXPECT_FALSE(Has(keys, priv.info.key));
  EXPECT_FALSE(Has(keys, "NCCL_NO_CACHE"));

  g_pluginEnv["NCCL_PARAM_DUMP_ALL"] = "1";
  keys = AllKeys();
  EXPECT_TRUE(Has(keys, pub.info.key));
  EXPECT_TRUE(Has(keys, priv.info.key));
  EXPECT_EQ(ncclParamRegistry::instance().size(), keys.size());
}

TEST_F(ParamCApiMicrotest, GetAllParameterKeys_RejectsNullArguments) {
  const char** table = nullptr;
  int len = 0;
  EXPECT_EQ(ncclInvalidArgument, ncclParamGetAllParameterKeys(nullptr, &len));
  EXPECT_EQ(ncclInvalidArgument, ncclParamGetAllParameterKeys(&table, nullptr));
}

TEST_F(ParamCApiMicrotest, GetParameter_ReturnsTheValueAsAString) {
  auto& p = Make<int32_t>(1);
  g_pluginEnv[p.info.key] = "-12";
  const char* value = nullptr;
  int len = 0;
  EXPECT_EQ(ncclSuccess, ncclParamGetParameter(p.info.key, &value, &len));
  EXPECT_STREQ("-12", value);
  EXPECT_EQ(3, len);
  // Goes through the same once-per-key notes as ncclParamBind.
  EXPECT_TRUE(Logged(NCCL_LOG_INFO, std::string("ENV: ") + p.info.key + " is private"));
}

TEST_F(ParamCApiMicrotest, GetParameter_RejectsNullArgumentsAndUnknownKeys) {
  const char* value = "stale";
  int len = 7;
  EXPECT_EQ(ncclInvalidArgument, ncclParamGetParameter(nullptr, &value, &len));
  EXPECT_EQ(ncclInvalidArgument, ncclParamGetParameter("K", nullptr, &len));
  EXPECT_EQ(ncclInvalidArgument, ncclParamGetParameter("K", &value, nullptr));
  EXPECT_EQ(ncclInvalidArgument, ncclParamGetParameter("PARAM_TEST_NEVER_REGISTERED", &value, &len));
  EXPECT_EQ(nullptr, value);
  EXPECT_EQ(0, len);
  EXPECT_TRUE(Logged(NCCL_LOG_INFO, "PARAM: key \"PARAM_TEST_NEVER_REGISTERED\" not found in registry"));
}

TEST_F(ParamCApiMicrotest, DumpAll_PrintsPublishedParamsUnlessDumpAll) {
  auto& pub = Make<int32_t>(1, NCCL_PARAM_FLAG_PUBLISHED);
  auto& priv = Make<int32_t>(2);
  std::string out = CaptureStdout([] { ncclParamDumpAll(); });
  EXPECT_EQ(0u, out.find("=== ncclParam Registry Dump ===\n"));
  EXPECT_NE(std::string::npos, out.find(pub.dump() + "\n"));
  EXPECT_EQ(std::string::npos, out.find(priv.info.key));

  g_pluginEnv["NCCL_PARAM_DUMP_ALL"] = "1";
  out = CaptureStdout([] { ncclParamDumpAll(); });
  EXPECT_NE(std::string::npos, out.find(priv.dump() + "\n"));
}

// src/misc/param.cc

class ParamLoadMicrotest : public ParamMicrotest {
 protected:
  static constexpr int64_t kUninit = INT64_MIN;
  int64_t cache = kUninit;
  int8_t noCache = -1;

  int64_t Load(const char* key, const char* value, int64_t def = 5) {
    if (value) g_pluginEnv[key] = value;
    return ncclLoadParam(key, def, kUninit, &cache, &noCache);
  }
};

TEST_F(ParamLoadMicrotest, Unset_ReturnsAndCachesTheDefault) {
  EXPECT_EQ(5, Load("PARAM_TEST_LOAD", nullptr));
  EXPECT_EQ(5, cache);
  EXPECT_EQ(0, noCache);
  EXPECT_TRUE(g_logged.empty());
  cache = kUninit;
  EXPECT_EQ(5, Load("PARAM_TEST_LOAD_EMPTY", ""));
  EXPECT_TRUE(g_logged.empty());
}

TEST_F(ParamLoadMicrotest, Numeric_ParsesAnyBaseAndReportsIt) {
  EXPECT_EQ(42, Load("PARAM_TEST_LOAD", "42"));
  EXPECT_EQ(42, cache);
  EXPECT_TRUE(Logged(NCCL_LOG_INFO, "PARAM_TEST_LOAD set by environment to 42."));
  struct Case {
    const char* in;
    int64_t want;
  };
  for (const auto& c : {Case{"0x10", 16}, Case{"010", 8}, Case{"-3", -3}, Case{"12abc", 12}}) {
    cache = kUninit;
    EXPECT_EQ(c.want, Load("PARAM_TEST_LOAD", c.in)) << c.in;
  }
}

TEST_F(ParamLoadMicrotest, Numeric_IgnoresAStaleErrno) {
  errno = ERANGE;
  EXPECT_EQ(42, Load("PARAM_TEST_LOAD", "42"));
}

TEST_F(ParamLoadMicrotest, Invalid_FallsBackToTheDefaultWithAnAttn) {
  EXPECT_EQ(5, Load("PARAM_TEST_LOAD", "abc"));
  EXPECT_EQ(5, cache);
  EXPECT_TRUE(Logged(NCCL_LOG_ATTN, "Invalid value abc for PARAM_TEST_LOAD, using default 5."));
  cache = kUninit;
  EXPECT_EQ(5, Load("PARAM_TEST_LOAD", "99999999999999999999"));
  EXPECT_TRUE(Logged(NCCL_LOG_ATTN, "Invalid value 99999999999999999999"));
}

TEST_F(ParamLoadMicrotest, Cached_ReturnsTheCacheWithoutReadingTheEnv) {
  cache = 9;
  EXPECT_EQ(9, Load("PARAM_TEST_LOAD", "42"));
  EXPECT_EQ(0, Lookups("PARAM_TEST_LOAD"));
  EXPECT_EQ(0, noCache);
}

TEST_F(ParamLoadMicrotest, NoCache_ListedKeyIsNeverStored) {
  EXPECT_EQ(42, Load(kUncachedLegacyKey, "42"));
  EXPECT_EQ(1, noCache);
  EXPECT_EQ(kUninit, cache);
}

TEST_F(ParamLoadMicrotest, NoCache_ResolvedPolicyIsNotRecomputed) {
  noCache = 1;
  EXPECT_EQ(42, Load("PARAM_TEST_LOAD", "42"));
  EXPECT_EQ(kUninit, cache);
  EXPECT_EQ(1, noCache);

  noCache = 0;
  EXPECT_EQ(42, Load(kUncachedLegacyKey, "42"));
  EXPECT_EQ(42, cache);
  EXPECT_FALSE(Logged(NCCL_LOG_INFO, "Disabling caching"));
}

TEST_F(ParamLoadMicrotest, GetEnv_InitializesThePluginAndReadsIt) {
  g_pluginEnv["PARAM_TEST_GETENV"] = "v";
  EXPECT_STREQ("v", ncclGetEnv("PARAM_TEST_GETENV"));
  EXPECT_EQ(nullptr, ncclGetEnv("PARAM_TEST_GETENV_UNSET"));
  EXPECT_EQ(2, g_initEnvCalls);
}

class ParamEnvFileMicrotest : public ParamMicrotest {};

TEST_F(ParamEnvFileMicrotest, UserHomeDir_IsHome) {
  const char* home = std::getenv("HOME");
  std::string saved = home ? home : "";
  setenv("HOME", "/param/test/home", 1);
  EXPECT_STREQ("/param/test/home", userHomeDir());
  unsetenv("HOME");
  EXPECT_EQ(nullptr, userHomeDir());
  if (home) setenv("HOME", saved.c_str(), 1);
}

TEST_F(ParamEnvFileMicrotest, SetEnvFile_MissingFileIsIgnored) {
  setEnvFile("/nonexistent/param_test.conf");
  EXPECT_TRUE(g_setEnv.empty());
}

TEST_F(ParamEnvFileMicrotest, SetEnvFile_SetsEachAssignmentLine) {
  setEnvFile(WriteFile("# COMMENT=1\nA=1\nB=x=y\r\n\nNO_EQUALS\n=empty\nC=\nD=last\r").c_str());
  EXPECT_EQ((Pairs{{"A", "1"}, {"B", "x=y"}, {"", "empty"}, {"C", ""}, {"D", "last"}}), g_setEnv);
}

TEST_F(ParamEnvFileMicrotest, SetEnvFile_TruncatesNamesAndValuesTo1023) {
  setEnvFile(WriteFile(std::string(1100, 'K') + "=" + std::string(1100, 'v') + "\n").c_str());
  EXPECT_EQ((Pairs{{std::string(1023, 'K'), std::string(1023, 'v')}}), g_setEnv);
}

class ParamEnvFileMicrotestIsolated : public ParamEnvFileMicrotest {};

// initEnv() latches behind std::call_once, so each source order needs a fresh process.
TEST_F(ParamEnvFileMicrotestIsolated, InitEnv_ReadsTheUserFileThenEtcOnce) {
  RUN_ISOLATED_TEST("Param_InitEnv_UserFile", [this]() {
    std::string user = WriteFile("U=1\n");
    g_fopenRedirect["/etc/rccl.conf"] = WriteFile("E=2\n");
    setenv("NCCL_CONF_FILE", user.c_str(), 1);
    initEnv();
    initEnv();
    EXPECT_EQ((std::vector<std::string>{user, "/etc/rccl.conf"}), g_opened);
    EXPECT_EQ((Pairs{{"U", "1"}, {"E", "2"}}), g_setEnv);
    for (const auto& path : files_) std::remove(path.c_str());  // TearDown does not run in the child
  });
}

TEST_F(ParamEnvFileMicrotestIsolated, InitEnv_EmptyConfFileFallsBackToHome) {
  RUN_ISOLATED_TEST("Param_InitEnv_Home", []() {
    g_fopenRedirect["/etc/rccl.conf"] = "/nonexistent/param_test.conf";
    setenv("NCCL_CONF_FILE", "", 1);
    setenv("HOME", "/param/test/home", 1);
    initEnv();
    EXPECT_EQ((std::vector<std::string>{"/param/test/home/.rccl.conf", "/etc/rccl.conf"}), g_opened);
  });
}

TEST_F(ParamEnvFileMicrotestIsolated, InitEnv_WithoutConfFileOrHomeReadsOnlyEtc) {
  RUN_ISOLATED_TEST("Param_InitEnv_EtcOnly", []() {
    g_fopenRedirect["/etc/rccl.conf"] = "/nonexistent/param_test.conf";
    unsetenv("NCCL_CONF_FILE");
    unsetenv("HOME");
    initEnv();
    EXPECT_EQ((std::vector<std::string>{"/etc/rccl.conf"}), g_opened);
  });
}

}  // namespace
