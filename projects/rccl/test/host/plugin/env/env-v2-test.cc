/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/plugin/env/env_v2.cc: a v2 env plugin is the current ABI and is used as exported.

#include <gtest/gtest.h>

#include <cstring>
#include <string>
#include <vector>

#include "nccl_env.h"
#include "os.h"

namespace {

void* const kLib = reinterpret_cast<void*>(0x22);
ncclEnv_v2_t g_plugin{"fake-v2", nullptr, nullptr, nullptr};
std::vector<std::string> g_dlsyms;

void* FakeDlsym(void* lib, const char* symbol) {
  g_dlsyms.push_back(symbol);
  if (lib != kLib) return nullptr;
  if (!std::strcmp(symbol, "ncclEnvPlugin_v2")) return &g_plugin;
  return nullptr;
}

}  // namespace

#define ncclOsDlsym FakeDlsym
#include ENV_V2_CC_PATH
#undef ncclOsDlsym

namespace {

TEST(EnvV2Microtest, GetterReturnsTheExportedTableOrNull) {
  g_dlsyms.clear();
  EXPECT_EQ(nullptr, getNcclEnv_v2(reinterpret_cast<void*>(0x23)));
  EXPECT_EQ(&g_plugin, getNcclEnv_v2(kLib));
  EXPECT_EQ(g_dlsyms, (std::vector<std::string>{"ncclEnvPlugin_v2", "ncclEnvPlugin_v2"}));
}

}  // namespace
