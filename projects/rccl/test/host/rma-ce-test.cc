/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host-only microtests for src/rma/rma_ce.cc (AICOMRCCL-2347).
 *************************************************************************/

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "ScopedHook.h"
#include "fakes/ce_fakes.h"
#include "fakes/dev_runtime_fakes.h"
#include "fakes/hip_fakes.h"
#include "fakes/nccl_stubs.h"
#include "fakes/rma_fakes.h"

#include "nccl.h"
#include "comm.h"
#include "rma/rma_ce.h"

// rma_ce.cc defines ncclRmaCePutLaunch / ncclRmaCeWaitLaunch, which rma-test.cc
// needs as controllable seams in this same binary (rma.cc dispatches to them).
// Rename the unit's own entry points so both can coexist: rma.cc keeps binding
// to the seams in rma_fakes.cc, and the tests below call the real ones.
// The static helpers (ncclRmaCePutLaunchPersist/NonPersist) are distinct tokens
// and so are untouched.
#define ncclRmaCePutLaunch ncclRmaCePutLaunchUut
#define ncclRmaCeWaitLaunch ncclRmaCeWaitLaunchUut
#include RMA_CE_CC_PATH
#undef ncclRmaCePutLaunch
#undef ncclRmaCeWaitLaunch

namespace {

// Minimal comm for the uninitialised guard: both entry points read
// rmaState.rmaCeState.initialized before anything else. ncclComm is ~3.8 MB, so
// it is heap-allocated rather than held by value.
class RmaCeLaunchTest : public ::testing::Test {
protected:
  std::unique_ptr<ncclComm> comm_;
  std::unique_ptr<ncclKernelPlan> plan_;

  void SetUp() override {
    comm_ = std::make_unique<ncclComm>();   // value-initialised, so initialized == false
    plan_ = std::make_unique<ncclKernelPlan>();
  }

  void TearDown() override {
    ResetCeFakes();
    ResetDevRuntimeFakes();
    ResetRmaFakes();
    ResetHipFakes();
  }
};

// ---------------------------------------------------------------------------
// ncclRmaCeInit (rma_ce.cc:23)
// ---------------------------------------------------------------------------

// Brings up CE state for a comm. The fixture supplies the two things the unit
// cannot compute for itself -- the registered window, and a stream/event pair --
// and lets everything else run for real so the layout arithmetic is the unit's.
class RmaCeInitTest : public ::testing::Test {
protected:
  static constexpr int kNRanks = 4;

  std::unique_ptr<ncclComm> comm_;
  // The window the register seam hands back, and the host object it shadows to.
  ncclWindow_vidmem win_{};
  ncclDevrWindow winHost_{};
  // Backing for the registered region, so the unit's pointer arithmetic lands
  // somewhere real. Sized the way the unit sizes it.
  std::vector<uint64_t> region_;
  size_t registeredBytes_ = 0;

  void SetUp() override {
    ResetCeFakes();
    ResetDevRuntimeFakes();
    ResetHipFakes();

    comm_ = std::make_unique<ncclComm>();
    comm_->nRanks = kNRanks;
    comm_->config.numRmaCtx = 1;

    win_.winHost = &winHost_;
    region_.assign(3 * kNRanks + 2, 0);

    // Hand back the fixture's window and record the size the unit asked for.
    g_devrWindowRegisterInGroup = [this](ncclComm*, void*, size_t size, int,
                                         ncclWindow_vidmem** outWin) {
      registeredBytes_ = size;
      if (outWin) *outWin = &win_;
      return ncclSuccess;
    };
    // The unit creates its CE stream and event last; both default to failure.
    g_hipStreamCreateResult = hipSuccess;
    g_hipEventCreateResult = hipSuccess;
    // The unit seeds its ack flags and signal constants through ncclCudaMemcpy /
    // ncclCudaCalloc, which go via stream-capture, memcpy-async and a stream
    // wait -- all behind the existing async-ops knob.
    g_hipAsyncOpsResult = hipSuccess;
  }

  void TearDown() override {
    ResetCeFakes();
    ResetDevRuntimeFakes();
    ResetHipFakes();
  }

  ncclRmaCeCtx* Ctx(int i) {
    return static_cast<ncclRmaCeCtx*>(comm_->rmaState.rmaCeState.rmaCeCtxs[i]);
  }
};

// The flag other entry points gate on is only set once everything above it
// succeeded, so it doubles as "the whole sequence ran".
TEST_F(RmaCeInitTest, Init_AllDependenciesSucceed_MarksStateInitialised) {
  ASSERT_EQ(ncclRmaCeInit(comm_.get()), ncclSuccess);

  EXPECT_TRUE(comm_->rmaState.rmaCeState.initialized);
  EXPECT_NE(comm_->rmaState.rmaCeState.ceStream, nullptr);
  EXPECT_NE(comm_->rmaState.rmaCeState.ceEvent, nullptr);
}

// One context per configured RMA context, and the count is taken from config.
TEST_F(RmaCeInitTest, Init_MultipleConfiguredContexts_CreatesOneEach) {
  comm_->config.numRmaCtx = 3;

  ASSERT_EQ(ncclRmaCeInit(comm_.get()), ncclSuccess);

  ASSERT_EQ(comm_->rmaState.rmaCeState.rmaCeCtxCount, 3);
  for (int i = 0; i < 3; i++) {
    EXPECT_NE(Ctx(i), nullptr) << "context " << i;
    EXPECT_EQ(Ctx(i)->comm, comm_.get()) << "context " << i;
  }
}

// The signal region is one buffer carved into three rank-indexed areas. This is
// the unit's own arithmetic, so it is pinned in both forms it publishes: device
// pointers for the kernels, and byte offsets for the window.
TEST_F(RmaCeInitTest, Init_Succeeds_CarvesSignalRegionByRankCount) {
  ASSERT_EQ(ncclRmaCeInit(comm_.get()), ncclSuccess);

  EXPECT_EQ(registeredBytes_, (3 * kNRanks + 2) * sizeof(uint64_t));

  const ncclRmaCeCtx* ctx = Ctx(0);
  ASSERT_NE(ctx, nullptr);
  uint64_t* base = ctx->signalsDev;
  ASSERT_NE(base, nullptr);
  EXPECT_EQ(ctx->graphSignalsDev, base + kNRanks + 1);
  EXPECT_EQ(ctx->graphAckDev, base + 2 * kNRanks + 2);

  EXPECT_EQ(ctx->signalOffset, 0u);
  EXPECT_EQ(ctx->graphSignalOffset, (kNRanks + 1) * sizeof(uint64_t));
  EXPECT_EQ(ctx->graphAckOffset, (2 * kNRanks + 2) * sizeof(uint64_t));

  // The window handle is taken from the shadow's host object, not the device one.
  EXPECT_EQ(ctx->signalsWin, reinterpret_cast<ncclDevrWindow*>(win_.winHost));
}

// Zero configured contexts is not an error: the stream and event still come up,
// so a comm that never uses RMA CE is still in a defined state.
TEST_F(RmaCeInitTest, Init_NoConfiguredContexts_StillCreatesStreamAndEvent) {
  comm_->config.numRmaCtx = 0;

  ASSERT_EQ(ncclRmaCeInit(comm_.get()), ncclSuccess);

  EXPECT_EQ(comm_->rmaState.rmaCeState.rmaCeCtxCount, 0);
  EXPECT_TRUE(comm_->rmaState.rmaCeState.initialized);
  EXPECT_NE(comm_->rmaState.rmaCeState.ceStream, nullptr);
}

// The symmetric runtime is brought up first, so its failure stops everything.
TEST_F(RmaCeInitTest, Init_DevrInitOnceFails_PropagatesAndLeavesUninitialised) {
  ScopedHook initOnce(g_devrInitOnce, [](ncclComm*) { return ncclSystemError; });

  EXPECT_EQ(ncclRmaCeInit(comm_.get()), ncclSystemError);
  EXPECT_FALSE(comm_->rmaState.rmaCeState.initialized);
  EXPECT_EQ(initOnce.calls, 1);
}

// A failed window registration unwinds mid-context rather than leaving the state
// half-built and marked ready.
TEST_F(RmaCeInitTest, Init_WindowRegisterFails_PropagatesAndLeavesUninitialised) {
  g_devrWindowRegisterInGroup = [](ncclComm*, void*, size_t, int, ncclWindow_vidmem**) {
    return ncclInternalError;
  };

  EXPECT_EQ(ncclRmaCeInit(comm_.get()), ncclInternalError);
  EXPECT_FALSE(comm_->rmaState.rmaCeState.initialized);
}

// The stream and event are created after every context is built, so a failure
// there still must not report the state as ready.
TEST_F(RmaCeInitTest, Init_StreamCreateFails_LeavesUninitialised) {
  g_hipStreamCreateResult = hipErrorInvalidValue;

  EXPECT_EQ(ncclRmaCeInit(comm_.get()), ncclUnhandledCudaError);
  EXPECT_FALSE(comm_->rmaState.rmaCeState.initialized);
}

TEST_F(RmaCeInitTest, Init_EventCreateFails_LeavesUninitialised) {
  g_hipEventCreateResult = hipErrorInvalidValue;

  EXPECT_EQ(ncclRmaCeInit(comm_.get()), ncclUnhandledCudaError);
  EXPECT_FALSE(comm_->rmaState.rmaCeState.initialized);
}

// ---------------------------------------------------------------------------
// ncclRmaCeFinalize (rma_ce.cc:105)
// ---------------------------------------------------------------------------

// Tears down exactly what ncclRmaCeInit builds, so the fixture builds it with the
// real thing rather than a hand-assembled imitation -- a teardown test whose
// input was not produced by the matching setup proves little.
class RmaCeFinalizeTest : public RmaCeInitTest {
protected:
  std::vector<void*> freed_;
  std::vector<ncclWindow_t> deregistered_;

  void SetUp() override {
    RmaCeInitTest::SetUp();
    g_ncclMemFree = [this](void* p) {
      freed_.push_back(p);
      return ncclSuccess;
    };
    g_ncclCommWindowDeregister = [this](ncclComm_t, ncclWindow_t win) {
      deregistered_.push_back(win);
      return ncclSuccess;
    };
    // ncclCudaFree looks up an allocation's base and size for its accounting
    // before releasing it. Report each pointer as its own base.
    g_hipMemGetAddressRange = [](hipDeviceptr_t* base, size_t* size, hipDeviceptr_t ptr) {
      if (base) *base = ptr;
      if (size) *size = 0;
      return hipSuccess;
    };
  }

  void TearDown() override {
    ResetCollectiveStubs();
    RmaCeInitTest::TearDown();
  }
};

// The whole point: after teardown the comm reports no CE state, so a later
// ncclRmaCeInit starts clean and the launch entry points refuse work again.
TEST_F(RmaCeFinalizeTest, Finalize_AfterInit_ResetsStateToUninitialised) {
  ASSERT_EQ(ncclRmaCeInit(comm_.get()), ncclSuccess);

  ASSERT_EQ(ncclRmaCeFinalize(comm_.get()), ncclSuccess);

  EXPECT_FALSE(comm_->rmaState.rmaCeState.initialized);
  EXPECT_EQ(comm_->rmaState.rmaCeState.rmaCeCtxCount, 0);
  EXPECT_EQ(comm_->rmaState.rmaCeState.rmaCeCtxs, nullptr);
}

// The stream and event are owned by this state, so they are released and the
// handles cleared -- leaving a dangling handle behind would outlive the comm.
TEST_F(RmaCeFinalizeTest, Finalize_AfterInit_ClearsStreamAndEvent) {
  ASSERT_EQ(ncclRmaCeInit(comm_.get()), ncclSuccess);
  ASSERT_NE(comm_->rmaState.rmaCeState.ceStream, nullptr);

  ASSERT_EQ(ncclRmaCeFinalize(comm_.get()), ncclSuccess);

  EXPECT_EQ(comm_->rmaState.rmaCeState.ceStream, nullptr);
  EXPECT_EQ(comm_->rmaState.rmaCeState.ceEvent, nullptr);
}

// Releasing the signal window is pure side effect, so it is pinned by observing
// the call. The handle deregistered must be the one the window was registered
// under, not the host shadow the unit reads its fields from.
TEST_F(RmaCeFinalizeTest, Finalize_AfterInit_DeregistersAndFreesEverySignalWindow) {
  comm_->config.numRmaCtx = 3;
  ASSERT_EQ(ncclRmaCeInit(comm_.get()), ncclSuccess);
  std::vector<void*> signalsDev{Ctx(0)->signalsDev, Ctx(1)->signalsDev, Ctx(2)->signalsDev};

  ASSERT_EQ(ncclRmaCeFinalize(comm_.get()), ncclSuccess);

  // Every context, in order -- not just the first, and not one repeated.
  EXPECT_EQ(deregistered_.size(), 3u);
  EXPECT_EQ(freed_, signalsDev);
}

// Finalizing a comm that was never initialised is not an error: every field it
// would release is null, so the guards skip and it reports success.
TEST_F(RmaCeFinalizeTest, Finalize_NeverInitialised_IsANoOpSuccess) {
  EXPECT_EQ(ncclRmaCeFinalize(comm_.get()), ncclSuccess);

  EXPECT_FALSE(comm_->rmaState.rmaCeState.initialized);
  EXPECT_TRUE(freed_.empty());
  EXPECT_TRUE(deregistered_.empty());
}

// Deferred init tasks are owned by the comm and outlive nothing, so teardown
// drains the queue rather than leaving entries pointing at freed state.
TEST_F(RmaCeFinalizeTest, Finalize_PendingInitTasks_DrainsTheQueue) {
  ncclIntruQueueConstruct(&comm_->rmaCeInitTaskQueue);
  for (int i = 0; i < 2; i++) {
    auto* task = static_cast<ncclRmaCeInitTask*>(::calloc(1, sizeof(ncclRmaCeInitTask)));
    ncclIntruQueueEnqueue(&comm_->rmaCeInitTaskQueue, task);
  }

  ASSERT_EQ(ncclRmaCeFinalize(comm_.get()), ncclSuccess);

  EXPECT_TRUE(ncclIntruQueueEmpty(&comm_->rmaCeInitTaskQueue));
}

// A failed deregistration stops teardown and surfaces, rather than carrying on
// and reporting the state as cleanly torn down.
TEST_F(RmaCeFinalizeTest, Finalize_DeregisterFails_PropagatesAndLeavesStateMarked) {
  ASSERT_EQ(ncclRmaCeInit(comm_.get()), ncclSuccess);
  g_ncclCommWindowDeregister = [](ncclComm_t, ncclWindow_t) { return ncclInternalError; };

  EXPECT_EQ(ncclRmaCeFinalize(comm_.get()), ncclInternalError);
  EXPECT_TRUE(comm_->rmaState.rmaCeState.initialized);
}

// ---------------------------------------------------------------------------
// ncclRmaCePutLaunch / ncclRmaCeWaitLaunch (rma_ce.cc:376, :388)
// ---------------------------------------------------------------------------

// Both entry points refuse to touch a communicator whose CE state was never
// brought up, rather than dereferencing it.
TEST_F(RmaCeLaunchTest, PutLaunch_CeNotInitialised_ReturnsInternalError) {
  EXPECT_EQ(ncclRmaCePutLaunchUut(comm_.get(), plan_.get(), nullptr), ncclInternalError);
}

TEST_F(RmaCeLaunchTest, WaitLaunch_CeNotInitialised_ReturnsInternalError) {
  EXPECT_EQ(ncclRmaCeWaitLaunchUut(comm_.get(), plan_.get(), nullptr), ncclInternalError);
}

}  // namespace
