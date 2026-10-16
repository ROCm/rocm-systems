/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/plugin/gin.cc's ncclGinPluginInit().
//
// NCCL 2.30.7 / NVIDIA/nccl#2179: init() allocates a per-comm GIN context
// before devices() is probed. If devices() then fails or reports ndev <= 0,
// the plugin is disabled and that context must be finalize()'d. Before the
// fix the pointer was dropped, leaking 8 bytes/rank (Valgrind: ncclCalloc
// in ncclGinIbInitType).
//
// ncclGinPluginInit is file-static, so this TU #includes the hipified
// gin.cc (GIN_CC_PATH), the standard rccl-UnitTestsMicro pattern. A
// scriptable ncclGin_t vtable stands in for the plugin; no GPU, no
// librccl.so, no network.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "nccl.h"
#include "comm.h"
#include "nccl_gin.h"
#include "plugin.h"

#include "fakes/nccl_fakes.h"

#ifdef ENABLE_ROCSHMEM_GIN
// gin.cc compares the plugin pointer against the built-in rocSHMEM/Anvil
// vtables after a successful init. Those objects live in TUs this binary
// does not compile; the addresses only need to exist so the comparisons
// are well-defined. Our fake vtable is a distinct object, so the branches
// are not taken.
ncclGin_t ncclGinRocshmemGdaPlugin{};
ncclGin_t ncclGinAnvilSdmaPlugin{};
void ncclGinAnvilSetInitContext(void*, struct ncclComm*) {}
#endif

#ifdef RCCL_NET_IB_CAST_ENABLE_GDAKI
ncclGin_t IbCastGinIbGdaki{};
#endif

// File-scope getNcclGin_v13/v14 pointers in gin.cc take these addresses
// even if the load path is --gc-sections'd away. Coverage instrumentation
// can also keep the rest of gin.cc live, so stub every remaining extern.
ncclGin_t* getNcclGin_v13(void*) { return nullptr; }
ncclGin_t* getNcclGin_v14(void*) { return nullptr; }

int64_t ncclParamGinEnable() { return 1; }
int64_t ncclParamGinType() { return -1; }

ncclGin_t ncclGinProxy{};
int ncclGinProxyVersion = 14;

char* ncclPluginLibPaths[6] = {};
void* ncclOpenGinPluginLib(const char*) { return nullptr; }
ncclResult_t ncclClosePluginLib(void*, ncclPluginType) { return ncclSuccess; }
void* ncclGetNetPluginLib(ncclPluginType) { return nullptr; }
const char* ncclGetPluginLibName(ncclPluginType) { return ""; }

#include "fakes/param_redirect.h"

#include GIN_CC_PATH

namespace {

struct FakeGin {
  int initCalls = 0;
  int devicesCalls = 0;
  int finalizeCalls = 0;
  ncclResult_t initResult = ncclSuccess;
  ncclResult_t devicesResult = ncclSuccess;
  int ndev = 0;
  void* lastCtx = nullptr;
  bool initNullFn = false;

  static FakeGin*& currentPtr() {
    static FakeGin* p = nullptr;
    return p;
  }
  static FakeGin& current() { return *currentPtr(); }
  static void setCurrent(FakeGin* p) { currentPtr() = p; }

  static ncclResult_t Init(void** ctx, uint64_t /*commId*/, ncclDebugLogger_t) {
    FakeGin& self = current();
    ++self.initCalls;
    if (self.initResult != ncclSuccess) return self.initResult;
    // Match the NVIDIA leak: ncclGinIbInitType calloc'd an 8-byte config.
    self.lastCtx = std::malloc(8);
    if (ctx) *ctx = self.lastCtx;
    return ncclSuccess;
  }

  static ncclResult_t Devices(int* ndev) {
    FakeGin& self = current();
    ++self.devicesCalls;
    if (self.devicesResult != ncclSuccess) return self.devicesResult;
    if (ndev) *ndev = self.ndev;
    return ncclSuccess;
  }

  static ncclResult_t Finalize(void* ctx) {
    FakeGin& self = current();
    ++self.finalizeCalls;
    std::free(ctx);
    if (self.lastCtx == ctx) self.lastCtx = nullptr;
    return ncclSuccess;
  }

  ncclGin_t vtable() {
    ncclGin_t gin{};
    gin.name = "GinInitLeakStub";
    gin.init = initNullFn ? nullptr : &Init;
    gin.devices = &Devices;
    gin.finalize = &Finalize;
    return gin;
  }
};

class GinPluginInitTest : public ::testing::Test {
 protected:
  FakeGin fake_;
  ncclGin_t gin_{};
  ginPluginLib_t pluginLib_{};
  ncclComm comm_{};
  void* ctx_ = nullptr;

  void SetUp() override {
    FakeGin::setCurrent(&fake_);
    gin_ = fake_.vtable();
    pluginLib_.ncclGin = &gin_;
    pluginLib_.state = ncclGinPluginStateInitReady;
    comm_.commHash = 0xabcdu;
    ctx_ = nullptr;
  }

  void TearDown() override {
    // Happy-path tests leave the context with the caller; free any leftover
    // so a missed finalize cannot leak into later cases.
    if (fake_.lastCtx) {
      std::free(fake_.lastCtx);
      fake_.lastCtx = nullptr;
    }
    FakeGin::setCurrent(nullptr);
  }

  ncclResult_t runInit() { return ncclGinPluginInit(&comm_, &pluginLib_, &ctx_); }
};

// NVIDIA/nccl#2179: devices() error after a successful init() must finalize.
TEST_F(GinPluginInitTest, FinalizesWhenDevicesFailsAfterInit) {
  fake_.devicesResult = ncclSystemError;

  ASSERT_EQ(runInit(), ncclSuccess);
  EXPECT_EQ(pluginLib_.state, ncclGinPluginStateDisabled);
  EXPECT_EQ(fake_.initCalls, 1);
  EXPECT_EQ(fake_.devicesCalls, 1);
  EXPECT_EQ(fake_.finalizeCalls, 1);
  EXPECT_EQ(ctx_, nullptr);
  EXPECT_EQ(fake_.lastCtx, nullptr);
}

// Same cleanup when devices() succeeds but reports no adapters.
TEST_F(GinPluginInitTest, FinalizesWhenDevicesReportsZero) {
  fake_.ndev = 0;
  fake_.devicesResult = ncclSuccess;

  ASSERT_EQ(runInit(), ncclSuccess);
  EXPECT_EQ(pluginLib_.state, ncclGinPluginStateDisabled);
  EXPECT_EQ(fake_.initCalls, 1);
  EXPECT_EQ(fake_.devicesCalls, 1);
  EXPECT_EQ(fake_.finalizeCalls, 1);
  EXPECT_EQ(ctx_, nullptr);
  EXPECT_EQ(fake_.lastCtx, nullptr);
}

// A failed init() never produced a context, so finalize() must not run.
TEST_F(GinPluginInitTest, DoesNotFinalizeAfterFailedInit) {
  fake_.initResult = ncclSystemError;

  ASSERT_EQ(runInit(), ncclSuccess);
  EXPECT_EQ(pluginLib_.state, ncclGinPluginStateDisabled);
  EXPECT_EQ(fake_.initCalls, 1);
  EXPECT_EQ(fake_.devicesCalls, 0);
  EXPECT_EQ(fake_.finalizeCalls, 0);
  EXPECT_EQ(ctx_, nullptr);
}

TEST_F(GinPluginInitTest, DoesNotFinalizeWhenInitPointerIsNull) {
  fake_.initNullFn = true;
  gin_ = fake_.vtable();

  ASSERT_EQ(runInit(), ncclSuccess);
  EXPECT_EQ(pluginLib_.state, ncclGinPluginStateDisabled);
  EXPECT_EQ(fake_.initCalls, 0);
  EXPECT_EQ(fake_.finalizeCalls, 0);
}

// Success path: keep the context, mark the plugin enabled, do not finalize.
TEST_F(GinPluginInitTest, KeepsContextWhenDevicesSucceed) {
  fake_.ndev = 2;

  ASSERT_EQ(runInit(), ncclSuccess);
  EXPECT_EQ(pluginLib_.state, ncclGinPluginStateEnabled);
  EXPECT_EQ(pluginLib_.physDevs, 2);
  EXPECT_EQ(fake_.initCalls, 1);
  EXPECT_EQ(fake_.devicesCalls, 1);
  EXPECT_EQ(fake_.finalizeCalls, 0);
  ASSERT_NE(ctx_, nullptr);
  EXPECT_EQ(ctx_, fake_.lastCtx);
}

}  // namespace
