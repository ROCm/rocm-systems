/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/plugin/env/env_v1.cc, the adapter that presents a v1 env plugin as the current
// ncclEnv_t: init drops the logger argument, finalize and getEnv are used as exported.

#include <gtest/gtest.h>

#include <cstring>
#include <string>
#include <vector>

#include "nccl_env.h"
#include "os.h"

namespace {

void* const kLib = reinterpret_cast<void*>(0x21);
ncclEnv_v1_t* g_export = nullptr;
std::vector<std::string> g_dlsyms;

void* FakeDlsym(void* lib, const char* symbol) {
  g_dlsyms.push_back(symbol);
  if (lib != kLib) return nullptr;
  if (!std::strcmp(symbol, "ncclEnvPlugin_v1")) return g_export;
  return nullptr;
}

}  // namespace

#define ncclOsDlsym FakeDlsym
#include ENV_V1_CC_PATH
#undef ncclOsDlsym

namespace {

struct InitCall {
  int calls = 0;
  uint8_t major = 0, minor = 0, patch = 0;
  std::string suffix;
};
InitCall g_init;
ncclResult_t g_initResult = ncclSuccess;

ncclResult_t V1Init(uint8_t major, uint8_t minor, uint8_t patch, const char* suffix) {
  g_init.calls++;
  g_init.major = major;
  g_init.minor = minor;
  g_init.patch = patch;
  g_init.suffix = suffix;
  return g_initResult;
}
ncclResult_t V1Finalize() { return ncclSuccess; }
const char* V1GetEnv(const char*) { return "value"; }
void Logger(ncclDebugLogLevel, unsigned long, const char*, int, const char*, ...) {}

class EnvV1Microtest : public ::testing::Test {
 protected:
  void SetUp() override {
    plugin_ = {"fake-v1", V1Init, V1Finalize, V1GetEnv};
    g_export = &plugin_;
    g_dlsyms.clear();
    g_init = {};
    g_initResult = ncclSuccess;
  }
  void TearDown() override { g_export = nullptr; }

  ncclEnv_v1_t plugin_;
};

TEST_F(EnvV1Microtest, MissingSymbolReturnsNull) {
  g_export = nullptr;
  EXPECT_EQ(nullptr, getNcclEnv_v1(kLib));
  EXPECT_EQ(g_dlsyms, std::vector<std::string>{"ncclEnvPlugin_v1"});
}

TEST_F(EnvV1Microtest, AdapterUsesFinalizeAndGetEnvAsExported) {
  ncclEnv_t* env = getNcclEnv_v1(kLib);
  ASSERT_NE(nullptr, env);
  EXPECT_EQ(V1Finalize, env->finalize);
  EXPECT_EQ(V1GetEnv, env->getEnv);
}

TEST_F(EnvV1Microtest, InitForwardsTheVersionAndDropsTheLogger) {
  ncclEnv_t* env = getNcclEnv_v1(kLib);
  ASSERT_NE(nullptr, env);
  EXPECT_EQ(ncclSuccess, env->init(2, 31, 7, "rc1", Logger));
  EXPECT_EQ(1, g_init.calls);
  EXPECT_EQ(2, g_init.major);
  EXPECT_EQ(31, g_init.minor);
  EXPECT_EQ(7, g_init.patch);
  EXPECT_EQ("rc1", g_init.suffix);
}

TEST_F(EnvV1Microtest, InitPropagatesThePluginError) {
  ncclEnv_t* env = getNcclEnv_v1(kLib);
  ASSERT_NE(nullptr, env);
  g_initResult = ncclInvalidUsage;
  EXPECT_EQ(ncclInvalidUsage, env->init(2, 31, 7, "", Logger));
}

}  // namespace
