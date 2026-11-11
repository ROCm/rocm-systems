/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/plugin/net/net_v12.cc: a v12 plugin is the current ABI and is used as exported.

#include <gtest/gtest.h>

#include <dlfcn.h>

#include <cstring>
#include <string>
#include <vector>

#include "nccl_net.h"
#include "proxy.h"

namespace {

void* const kLib = reinterpret_cast<void*>(0x12);
ncclNet_v12_t g_net{};
ncclCollNet_v12_t g_collNet{};
std::vector<std::string> g_dlsyms;

void* FakeDlsym(void* lib, const char* symbol) {
  g_dlsyms.push_back(symbol);
  if (lib != kLib) return nullptr;
  if (!std::strcmp(symbol, "ncclNetPlugin_v12")) return &g_net;
  if (!std::strcmp(symbol, "ncclCollNetPlugin_v12")) return &g_collNet;
  return nullptr;
}

}  // namespace

// net_v12.cc calls dlsym directly rather than ncclOsDlsym. Its headers are included above so the rename
// only reaches net_v12.cc itself.
#define dlsym FakeDlsym
#include NET_V12_CC_PATH
#undef dlsym

namespace {

TEST(PluginNetV12Microtest, GettersReturnTheExportedTablesUnchanged) {
  g_dlsyms.clear();
  EXPECT_EQ(getNcclNet_v12(nullptr), nullptr);
  EXPECT_EQ(getNcclCollNet_v12(nullptr), nullptr);
  EXPECT_EQ(getNcclNet_v12(kLib), &g_net);
  EXPECT_EQ(getNcclCollNet_v12(kLib), &g_collNet);
  EXPECT_EQ(g_dlsyms, (std::vector<std::string>{"ncclNetPlugin_v12", "ncclCollNetPlugin_v12", "ncclNetPlugin_v12",
                                                "ncclCollNetPlugin_v12"}));
}

}  // namespace
