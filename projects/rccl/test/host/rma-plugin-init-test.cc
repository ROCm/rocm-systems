/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/plugin/rma.cc's ncclRmaPluginInit()
// (AICOMRCCL-3322, ported from upstream NVIDIA/nccl commit 8136781a37).
//
// init() allocates a per-comm RMA context before devices() is probed. If
// devices() then fails or reports ndev <= 0, the plugin is disabled and that
// context must be finalize()'d and comm->rmaContext cleared. Before the fix the
// pointer was dropped (the context leaked). The fix adds an initCompleted flag
// and restructures the control flow onto a goto fail/exit pair, mirroring the
// GIN plugin path covered by gin-plugin-init-test.cc.
//
// ncclRmaPluginInit is file-static, so this TU #includes the hipified plugin
// rma.cc (RMA_PLUGIN_CC_PATH), the standard rccl-UnitTestsMicro pattern. A
// scriptable ncclRma_t vtable stands in for the plugin; no GPU, no librccl.so,
// no network.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>

#include "nccl.h"
#include "comm.h"
#include "nccl_rma.h"
#include "plugin.h"

#include "fakes/nccl_fakes.h"

// Externs the plugin rma.cc load path references. The tests call the file-static
// ncclRmaPluginInit directly and never exercise these, but ncclRmaInit (compiled
// into this TU) references them through initPluginLibsOnceFunc / ncclRmaPluginLoad,
// so they must resolve at link time. ncclPluginLibPaths, ncclClosePluginLib,
// ncclGetNetPluginLib and ncclGetPluginLibName are owned by gin-plugin-init-test.cc
// in the same binary, so they are deliberately not redefined here.
ncclRma_t* getNcclRma_v16(void*) { return nullptr; }
ncclRma_t* getNcclRma_v15(void*) { return nullptr; }
ncclRma_t* getNcclRma_v14(void*) { return nullptr; }
ncclRma_t* getNcclRma_v13(void*) { return nullptr; }
void* ncclOpenRmaPluginLib(const char*) { return nullptr; }
void* ncclGetGinPluginLib(ncclPluginType) { return nullptr; }

// The three built-in RMA proxies live in production TUs this binary does not
// compile; only their addresses need to exist for initPluginLibsOnceFunc.
ncclRma_t ncclRmaIbProxy{};
ncclRma_t IbCastRmaIbProxy{};
ncclRma_t ncclRmaSocketProxy{};

// ncclRmaInit tail-calls this once a plugin is assigned; never reached by the
// file-static entry point under test.
ncclResult_t ncclGinProxyInit(struct ncclComm*) { return ncclSuccess; }

#include "fakes/param_redirect.h"

#include RMA_PLUGIN_CC_PATH

namespace {

// One scriptable ncclRma_t. init() malloc's a context the way the IB plugin
// calloc'd one; finalize() frees whatever context it is handed. State is static
// so the vtable slots (plain C function pointers) can reach it; the fixture
// resets it on entry and exit.
struct FakeRma {
  static int initCalls;
  static int devicesCalls;
  static int finalizeCalls;
  static ncclResult_t initResult;
  static ncclResult_t devicesResult;
  static int ndev;
  static void* lastCtx;      // the context init() handed out, or null once freed
  static void* finalizedCtx; // the pointer finalize() got; kNotCalled if never
  static bool initNullFn;    // leave vtable.init null, as an unloaded slot would

  static void* kNotCalled() { return reinterpret_cast<void*>(~std::uintptr_t(0)); }

  static void reset() {
    initCalls = devicesCalls = finalizeCalls = 0;
    initResult = devicesResult = ncclSuccess;
    ndev = 1;
    lastCtx = nullptr;
    finalizedCtx = kNotCalled();
    initNullFn = false;
  }

  static ncclResult_t Init(void** ctx, uint64_t /*commId*/, ncclDebugLogger_t) {
    ++initCalls;
    if (initResult != ncclSuccess) return initResult;
    lastCtx = std::malloc(8);
    if (ctx) *ctx = lastCtx;
    return ncclSuccess;
  }

  static ncclResult_t Devices(int* n) {
    ++devicesCalls;
    if (devicesResult != ncclSuccess) return devicesResult;
    if (n) *n = ndev;
    return ncclSuccess;
  }

  static ncclResult_t Finalize(void* ctx) {
    ++finalizeCalls;
    finalizedCtx = ctx;
    std::free(ctx);
    if (lastCtx == ctx) lastCtx = nullptr;
    return ncclSuccess;
  }

  static ncclRma_t vtable() {
    ncclRma_t r{};
    r.name = "FakeRma";
    r.init = initNullFn ? nullptr : &Init;
    r.devices = &Devices;
    r.finalize = &Finalize;
    return r;
  }
};

int FakeRma::initCalls = 0;
int FakeRma::devicesCalls = 0;
int FakeRma::finalizeCalls = 0;
ncclResult_t FakeRma::initResult = ncclSuccess;
ncclResult_t FakeRma::devicesResult = ncclSuccess;
int FakeRma::ndev = 1;
void* FakeRma::lastCtx = nullptr;
void* FakeRma::finalizedCtx = nullptr;
bool FakeRma::initNullFn = false;

class RmaPluginInitTest : public ::testing::Test {
 protected:
  // ncclComm is ~3.8 MB, so it is heap-allocated, never a stack fixture member.
  std::unique_ptr<ncclComm> comm_ = std::make_unique<ncclComm>();
  rmaPluginLib_t pluginLib_{};
  ncclRma_t rma_{};

  void SetUp() override {
    FakeRma::reset();
    rma_ = FakeRma::vtable();
    pluginLib_.ncclRma = &rma_;
    pluginLib_.state = ncclRmaPluginStateInitReady;
    pluginLib_.physDevs = 0;
    std::strncpy(pluginLib_.name, "FakeRma", MAX_STR_LEN - 1);
    comm_->commHash = 0xabcdu;
    comm_->rmaContext = nullptr;
  }

  void TearDown() override {
    // Happy-path tests leave the context with the comm; free any leftover so a
    // missed finalize cannot leak into later cases.
    if (comm_->rmaContext) {
      std::free(comm_->rmaContext);
      comm_->rmaContext = nullptr;
    }
    FakeRma::reset();
  }

  ncclResult_t runInit() { return ncclRmaPluginInit(comm_.get(), &pluginLib_); }
};

// Happy path: init() and devices() both succeed, so the plugin is enabled, its
// device count recorded, the context kept, and finalize() never runs.
TEST_F(RmaPluginInitTest, KeepsContextWhenDevicesSucceed) {
  FakeRma::ndev = 2;

  ASSERT_EQ(runInit(), ncclSuccess);
  EXPECT_EQ(pluginLib_.state, ncclRmaPluginStateEnabled);
  EXPECT_EQ(pluginLib_.physDevs, 2);
  EXPECT_EQ(FakeRma::initCalls, 1);
  EXPECT_EQ(FakeRma::devicesCalls, 1);
  EXPECT_EQ(FakeRma::finalizeCalls, 0);
  ASSERT_NE(comm_->rmaContext, nullptr);
  EXPECT_EQ(comm_->rmaContext, FakeRma::lastCtx);
}

// The fix: devices() error after a successful init() must finalize the context
// init() allocated and clear comm->rmaContext, rather than leak it.
TEST_F(RmaPluginInitTest, FinalizesWhenDevicesFailsAfterInit) {
  FakeRma::devicesResult = ncclSystemError;

  // ncclRmaPluginInit funnels every failure through its exit label, so it
  // reports ncclSuccess even when the plugin ends up disabled.
  ASSERT_EQ(runInit(), ncclSuccess);
  EXPECT_EQ(pluginLib_.state, ncclRmaPluginStateDisabled);
  EXPECT_EQ(pluginLib_.physDevs, 0);
  EXPECT_EQ(FakeRma::initCalls, 1);
  EXPECT_EQ(FakeRma::devicesCalls, 1);
  EXPECT_EQ(FakeRma::finalizeCalls, 1);
  // finalize() was handed the context init() produced, and the comm no longer
  // holds a dangling pointer to it.
  EXPECT_NE(FakeRma::finalizedCtx, FakeRma::kNotCalled());
  EXPECT_EQ(comm_->rmaContext, nullptr);
  EXPECT_EQ(FakeRma::lastCtx, nullptr);
}

// Same cleanup when devices() succeeds but reports no adapters (ndev <= 0).
TEST_F(RmaPluginInitTest, FinalizesWhenDevicesReportsZero) {
  FakeRma::ndev = 0;
  FakeRma::devicesResult = ncclSuccess;

  ASSERT_EQ(runInit(), ncclSuccess);
  EXPECT_EQ(pluginLib_.state, ncclRmaPluginStateDisabled);
  EXPECT_EQ(pluginLib_.physDevs, 0);
  EXPECT_EQ(FakeRma::initCalls, 1);
  EXPECT_EQ(FakeRma::devicesCalls, 1);
  EXPECT_EQ(FakeRma::finalizeCalls, 1);
  EXPECT_EQ(comm_->rmaContext, nullptr);
  EXPECT_EQ(FakeRma::lastCtx, nullptr);
}

// A failed init() never produced a context (initCompleted stays false), so the
// fail label must not call finalize() on a pointer init() never set.
TEST_F(RmaPluginInitTest, DoesNotFinalizeAfterFailedInit) {
  FakeRma::initResult = ncclSystemError;

  ASSERT_EQ(runInit(), ncclSuccess);
  EXPECT_EQ(pluginLib_.state, ncclRmaPluginStateDisabled);
  EXPECT_EQ(pluginLib_.physDevs, 0);
  EXPECT_EQ(FakeRma::initCalls, 1);
  EXPECT_EQ(FakeRma::devicesCalls, 0);
  EXPECT_EQ(FakeRma::finalizeCalls, 0);
  EXPECT_EQ(FakeRma::finalizedCtx, FakeRma::kNotCalled());
  EXPECT_EQ(comm_->rmaContext, nullptr);
}

// A plugin already enabled by an earlier comm: init() still runs to set this
// comm's context, but devices() is probed only on the first init, so the
// recorded device count is left untouched and nothing is finalized.
TEST_F(RmaPluginInitTest, AlreadyEnabled_ReinitsContextButSkipsDevices) {
  pluginLib_.state = ncclRmaPluginStateEnabled;
  pluginLib_.physDevs = 5;  // discovered on the first comm; must survive re-init

  ASSERT_EQ(runInit(), ncclSuccess);
  EXPECT_EQ(pluginLib_.state, ncclRmaPluginStateEnabled);
  EXPECT_EQ(pluginLib_.physDevs, 5);
  EXPECT_EQ(FakeRma::initCalls, 1);
  EXPECT_EQ(FakeRma::devicesCalls, 0);
  EXPECT_EQ(FakeRma::finalizeCalls, 0);
  ASSERT_NE(comm_->rmaContext, nullptr);
  EXPECT_EQ(comm_->rmaContext, FakeRma::lastCtx);
}

// Re-init of an already-enabled plugin whose init() now fails: initCompleted is
// false on this path too, so the fail label must not finalize.
TEST_F(RmaPluginInitTest, AlreadyEnabled_InitFails_DoesNotFinalize) {
  pluginLib_.state = ncclRmaPluginStateEnabled;
  pluginLib_.physDevs = 5;
  FakeRma::initResult = ncclSystemError;

  ASSERT_EQ(runInit(), ncclSuccess);
  EXPECT_EQ(pluginLib_.state, ncclRmaPluginStateDisabled);
  EXPECT_EQ(pluginLib_.physDevs, 0);
  EXPECT_EQ(FakeRma::initCalls, 1);
  EXPECT_EQ(FakeRma::finalizeCalls, 0);
  EXPECT_EQ(comm_->rmaContext, nullptr);
}

// Both guards carry a `&& pluginLib->ncclRma` term: a null vtable skips init and
// devices entirely and the plugin is enabled without touching the fakes.
TEST_F(RmaPluginInitTest, NullVtable_EnablesWithoutCallingIntoPlugin) {
  pluginLib_.ncclRma = nullptr;

  ASSERT_EQ(runInit(), ncclSuccess);
  EXPECT_EQ(pluginLib_.state, ncclRmaPluginStateEnabled);
  EXPECT_EQ(FakeRma::initCalls, 0);
  EXPECT_EQ(FakeRma::devicesCalls, 0);
  EXPECT_EQ(FakeRma::finalizeCalls, 0);
  EXPECT_EQ(comm_->rmaContext, nullptr);
}

}  // namespace
