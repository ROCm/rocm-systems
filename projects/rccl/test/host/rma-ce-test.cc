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

// The dispatcher's own contract is the guard and the persistent split; what each
// path then does is that helper's contract, covered separately. The two are told
// apart by the capacity they size their batch-ops params to -- the persistent
// path builds one op at a time, the non-persistent path one per rank.
class RmaCePutLaunchTest : public RmaCeInitTest {
protected:
  std::unique_ptr<ncclKernelPlan> plan_;
  ncclRmaArgs args_{};
  std::vector<int> initCapacities_;

  void SetUp() override {
    RmaCeInitTest::SetUp();
    ASSERT_EQ(ncclRmaCeInit(comm_.get()), ncclSuccess);

    plan_ = std::make_unique<ncclKernelPlan>();
    plan_->rmaArgs = &args_;
    args_.ctx = 0;
    args_.nRmaTasksCe = 0;  // no tasks: the split is the only thing under test

    g_ceInitBatchOpsParams = [this](ncclCeBatchOpsParams*, int capacity) {
      initCapacities_.push_back(capacity);
      return ncclSuccess;
    };
  }
};

// A graph-captured plan takes the persistent path, which sizes its batches for a
// single op because it emits them per task rather than per round.
TEST_F(RmaCePutLaunchTest, PutLaunch_PersistentPlan_TakesPersistentPath) {
  plan_->persistent = true;

  EXPECT_EQ(ncclRmaCePutLaunchUut(comm_.get(), plan_.get(), nullptr), ncclSuccess);

  EXPECT_EQ(initCapacities_, (std::vector<int>{1, 1}));
}

// A non-captured plan takes the other path, which batches across peers and so
// sizes for the rank count. With no tasks it returns before doing even that.
TEST_F(RmaCePutLaunchTest, PutLaunch_NonPersistentPlanWithNoTasks_TakesNonPersistentPath) {
  plan_->persistent = false;

  EXPECT_EQ(ncclRmaCePutLaunchUut(comm_.get(), plan_.get(), nullptr), ncclSuccess);

  EXPECT_TRUE(initCapacities_.empty());
}

// Same path, now with work, so the sizing it uses is visible.
TEST_F(RmaCePutLaunchTest, PutLaunch_NonPersistentPlanWithTasks_SizesBatchesPerRank) {
  plan_->persistent = false;
  args_.nRmaTasksCe = 1;
  auto task = std::make_unique<ncclTaskRma>();
  task->peer = 0;
  ncclIntruQueueEnqueue(&plan_->rmaTaskQueueCe, task.get());

  ncclRmaCePutLaunchUut(comm_.get(), plan_.get(), nullptr);

  ASSERT_GE(initCapacities_.size(), 2u);
  EXPECT_EQ(initCapacities_[0], kNRanks);
  EXPECT_EQ(initCapacities_[1], kNRanks);
}

// A failure inside the chosen path is the dispatcher's result; it does not
// swallow it or substitute one of its own.
TEST_F(RmaCePutLaunchTest, PutLaunch_ChosenPathFails_PropagatesUnchanged) {
  plan_->persistent = true;
  g_ceInitBatchOpsParams = [](ncclCeBatchOpsParams*, int) { return ncclSystemError; };

  EXPECT_EQ(ncclRmaCePutLaunchUut(comm_.get(), plan_.get(), nullptr), ncclSystemError);
}

// ---------------------------------------------------------------------------
// ncclRmaCePutLaunchNonPersist (rma_ce.cc:247)
// ---------------------------------------------------------------------------

// What the unit submits, in order. The batch params are reused between rounds,
// so the ops are snapshotted at launch rather than inspected afterwards.
struct SubmittedOp {
  void* src;
  void* dst;
  size_t size;
};
// One entry of a stream batch-memory-op submission.
struct MemOp {
  unsigned operation;
  const void* address;
  uint64_t value;
};
struct Submission {
  enum Kind { kMemOps, kBatch };
  Kind kind;
  std::vector<SubmittedOp> ops;    // kBatch
  std::vector<MemOp> memOps;       // kMemOps
};

// Drives one non-persistent put launch and records everything it enqueued.
// Tasks are grouped by peer and issued a round at a time, one task per peer per
// round, so the shape of this log is the unit's contract.
class RmaCeNonPersistTest : public RmaCeInitTest {
protected:
  std::unique_ptr<ncclKernelPlan> plan_;
  ncclRmaArgs args_{};
  std::vector<Submission> log_;
  // Peer addresses handed back per (win, offset) lookup, so a test can tell the
  // data destination from the signal destination.
  uint64_t peerData_[8]{};
  uint64_t peerSignal_[8]{};
  std::vector<uint64_t> srcBuf_;

  void SetUp() override {
    RmaCeInitTest::SetUp();
    ASSERT_EQ(ncclRmaCeInit(comm_.get()), ncclSuccess);

    ncclMemoryStackConstruct(&comm_->memPermanent);
    ncclMemoryPoolConstruct(&comm_->memPool_ncclTaskRma);
    srcBuf_.assign(8, 0);

    plan_ = std::make_unique<ncclKernelPlan>();
    plan_->rmaArgs = &args_;
    plan_->persistent = false;
    args_.ctx = 0;

    // Resolve a peer address per lookup. The signal lookup is the one made
    // against the context's own signals window; anything else is data.
    // Offset the returned address by the requested window offset, as the real
    // lookup does -- otherwise a unit that passed the wrong offset would still
    // land on the address a test expects.
    ncclRmaCeCtx* ceCtx = Ctx(0);
    g_devrGetLsaRankPtr = [this, ceCtx](ncclComm*, ncclDevrWindow* win, size_t offset, int lsaRank,
                                        void** outPtr) {
      char* base = (win == ceCtx->signalsWin) ? reinterpret_cast<char*>(&peerSignal_[lsaRank])
                                              : reinterpret_cast<char*>(&peerData_[lsaRank]);
      *outPtr = base + offset;
      return ncclSuccess;
    };
    g_ceLaunchBatchOps = [this](ncclComm*, ncclCeBatchOpsParams* p, hipStream_t,
                                ncclCeCollArgs*) {
      Submission s{Submission::kBatch, {}, {}};
      for (size_t i = 0; i < p->numOps; i++) s.ops.push_back({p->srcs[i], p->dsts[i], p->sizes[i]});
      log_.push_back(std::move(s));
      return ncclSuccess;
    };
    g_cuStreamBatchMemOp = [this](hipStream_t, unsigned int numOps,
                                  hipStreamBatchMemOpParams* ops) {
      Submission s{Submission::kMemOps, {}, {}};
      for (unsigned int i = 0; i < numOps; i++) {
        // The wait and write forms share a layout, so one read covers both.
        s.memOps.push_back({ops[i].writeValue.operation,
                            reinterpret_cast<const void*>(ops[i].writeValue.address),
                            ops[i].writeValue.value64});
      }
      log_.push_back(std::move(s));
      return ncclSuccess;
    };
  }

  void TearDown() override {
    ncclMemoryStackDestruct(&comm_->memPermanent);
    RmaCeInitTest::TearDown();
  }

  // Queue one CE task. bytes == 0 means signal-only; signal == false means data-only.
  void PushTask(int peer, size_t bytes, bool signal, size_t winOffset = 0) {
    auto* t = ncclMemoryPoolAlloc<ncclTaskRma>(&comm_->memPool_ncclTaskRma, &comm_->memPermanent);
    t->peer = peer;
    t->count = bytes;
    t->datatype = ncclUint8;  // one byte per element, so count is the byte count
    t->srcBuff = srcBuf_.data();
    t->peerWinOffset = winOffset;
    t->signalMode = signal ? NCCL_SIGNAL : NCCL_SIGNAL_NONE;
    ncclIntruQueueEnqueue(&plan_->rmaTaskQueueCe, t);
    args_.nRmaTasksCe++;
  }

  // The batches in submission order, ignoring the staging writes.
  std::vector<std::vector<SubmittedOp>> Batches() const {
    std::vector<std::vector<SubmittedOp>> out;
    for (const auto& s : log_) {
      if (s.kind == Submission::kBatch) out.push_back(s.ops);
    }
    return out;
  }
};

// A data-carrying task becomes one copy from the task's own buffer to the peer
// address resolved for it, sized by count and datatype.
TEST_F(RmaCeNonPersistTest, NonPersist_OneDataTask_CopiesTaskBufferToResolvedPeer) {
  // Non-zero window offset, so the destination is pinned to the task's offset
  // rather than just to the peer.
  PushTask(/*peer=*/2, /*bytes=*/64, /*signal=*/false, /*winOffset=*/8);

  ASSERT_EQ(ncclRmaCePutLaunchUut(comm_.get(), plan_.get(), nullptr), ncclSuccess);

  auto batches = Batches();
  ASSERT_EQ(batches.size(), 2u);           // data batch, then signal batch
  ASSERT_EQ(batches[0].size(), 1u);
  EXPECT_EQ(batches[0][0].src, srcBuf_.data());
  EXPECT_EQ(batches[0][0].dst, reinterpret_cast<char*>(&peerData_[2]) + 8);
  EXPECT_EQ(batches[0][0].size, 64u);
  EXPECT_TRUE(batches[1].empty());         // nothing signalled
}

// Tasks for different peers travel together: one round, one op per peer.
TEST_F(RmaCeNonPersistTest, NonPersist_TasksForDifferentPeers_BatchedIntoOneRound) {
  PushTask(/*peer=*/1, 32, false);
  PushTask(/*peer=*/3, 48, false);

  ASSERT_EQ(ncclRmaCePutLaunchUut(comm_.get(), plan_.get(), nullptr), ncclSuccess);

  auto batches = Batches();
  ASSERT_EQ(batches.size(), 2u);
  ASSERT_EQ(batches[0].size(), 2u);
  EXPECT_EQ(batches[0][0].dst, &peerData_[1]);
  EXPECT_EQ(batches[0][1].dst, &peerData_[3]);
}

// Two tasks for the same peer cannot share a batch, because a batched copy does
// not order its own operations. They are issued a round apart instead.
TEST_F(RmaCeNonPersistTest, NonPersist_TasksForSamePeer_IssuedInSeparateRounds) {
  PushTask(/*peer=*/1, 32, false);
  PushTask(/*peer=*/1, 48, false);

  ASSERT_EQ(ncclRmaCePutLaunchUut(comm_.get(), plan_.get(), nullptr), ncclSuccess);

  auto batches = Batches();
  ASSERT_EQ(batches.size(), 4u);           // two rounds of (data, signal)
  ASSERT_EQ(batches[0].size(), 1u);
  EXPECT_EQ(batches[0][0].size, 32u);
  ASSERT_EQ(batches[2].size(), 1u);
  EXPECT_EQ(batches[2][0].size, 48u);
}

// The sequence number is staged to device memory before the batch that copies it
// onward, because the copy reads the staged slot.
TEST_F(RmaCeNonPersistTest, NonPersist_SignallingTask_StagesSequenceBeforeCopyingIt) {
  PushTask(/*peer=*/2, 16, /*signal=*/true);

  ASSERT_EQ(ncclRmaCePutLaunchUut(comm_.get(), plan_.get(), nullptr), ncclSuccess);

  ASSERT_EQ(log_.size(), 3u);
  EXPECT_EQ(log_[0].kind, Submission::kMemOps);
  EXPECT_EQ(log_[1].kind, Submission::kBatch);   // data
  EXPECT_EQ(log_[2].kind, Submission::kBatch);   // signal
  ASSERT_EQ(log_[2].ops.size(), 1u);
  EXPECT_EQ(log_[2].ops[0].dst, &peerSignal_[2]);
  EXPECT_EQ(log_[2].ops[0].size, sizeof(uint64_t));
}

// The sequence a peer is signalled with advances per round, so a receiver can
// tell a repeated signal from a new one.
TEST_F(RmaCeNonPersistTest, NonPersist_RepeatedSignalsToSamePeer_AdvanceTheSequence) {
  PushTask(/*peer=*/1, 0, true);
  PushTask(/*peer=*/1, 0, true);

  ASSERT_EQ(ncclRmaCePutLaunchUut(comm_.get(), plan_.get(), nullptr), ncclSuccess);

  std::vector<uint64_t> staged;
  for (const auto& s : log_) {
    for (const auto& op : s.memOps) staged.push_back(op.value);
  }
  EXPECT_EQ(staged, (std::vector<uint64_t>{1, 2}));
}

// A task carrying no bytes is signal-only: nothing is copied for it.
TEST_F(RmaCeNonPersistTest, NonPersist_ZeroByteTask_EnqueuesNoDataCopy) {
  PushTask(/*peer=*/2, /*bytes=*/0, /*signal=*/true);

  ASSERT_EQ(ncclRmaCePutLaunchUut(comm_.get(), plan_.get(), nullptr), ncclSuccess);

  auto batches = Batches();
  ASSERT_EQ(batches.size(), 2u);
  EXPECT_TRUE(batches[0].empty());        // no data op
  EXPECT_EQ(batches[1].size(), 1u);       // signal still sent
}

// An unresolvable peer address is rejected rather than copied into.
TEST_F(RmaCeNonPersistTest, NonPersist_PeerAddressUnresolved_ReturnsInvalidArgument) {
  g_devrGetLsaRankPtr = [](ncclComm*, ncclDevrWindow*, size_t, int, void** outPtr) {
    *outPtr = nullptr;
    return ncclSuccess;
  };
  PushTask(/*peer=*/1, 32, false);

  EXPECT_EQ(ncclRmaCePutLaunchUut(comm_.get(), plan_.get(), nullptr), ncclInvalidArgument);
}

// ---------------------------------------------------------------------------
// ncclRmaCePutLaunchPersist (rma_ce.cc:162)
// ---------------------------------------------------------------------------

// The graph-captured sibling of the non-persistent path. Same recording fixture;
// only the plan differs. It issues one batch pair per task in queue order rather
// than grouping by peer into rounds, and it signals differently: an ack
// handshake before the copies, and a device-resident constant written to the
// graph signal slot instead of a staged sequence.
class RmaCePersistTest : public RmaCeNonPersistTest {
protected:
  void SetUp() override {
    RmaCeNonPersistTest::SetUp();
    plan_->persistent = true;
  }
};

// Data movement is the same as the non-persistent path: the task's own buffer to
// the address resolved for its peer and window offset.
TEST_F(RmaCePersistTest, Persist_OneDataTask_CopiesTaskBufferToResolvedPeer) {
  PushTask(/*peer=*/2, /*bytes=*/64, /*signal=*/false, /*winOffset=*/8);

  ASSERT_EQ(ncclRmaCePutLaunchUut(comm_.get(), plan_.get(), nullptr), ncclSuccess);

  auto batches = Batches();
  ASSERT_EQ(batches.size(), 2u);
  ASSERT_EQ(batches[0].size(), 1u);
  EXPECT_EQ(batches[0][0].src, srcBuf_.data());
  EXPECT_EQ(batches[0][0].dst, reinterpret_cast<char*>(&peerData_[2]) + 8);
  EXPECT_EQ(batches[0][0].size, 64u);
}

// Under graph capture the sender waits for the receiver's ack and clears it
// before writing, so a replayed graph cannot outrun the receiver. The handshake
// is submitted before the copies it guards.
TEST_F(RmaCePersistTest, Persist_SignallingTask_WaitsForAckAndClearsItBeforeCopying) {
  PushTask(/*peer=*/3, /*bytes=*/32, /*signal=*/true);
  const void* ackAddr = &Ctx(0)->graphAckDev[3];

  ASSERT_EQ(ncclRmaCePutLaunchUut(comm_.get(), plan_.get(), nullptr), ncclSuccess);

  ASSERT_GE(log_.size(), 1u);
  ASSERT_EQ(log_[0].kind, Submission::kMemOps);
  ASSERT_EQ(log_[0].memOps.size(), 2u);
  EXPECT_EQ(log_[0].memOps[0].operation, hipStreamMemOpWaitValue64);
  EXPECT_EQ(log_[0].memOps[0].address, ackAddr);
  EXPECT_EQ(log_[0].memOps[0].value, 1u);
  EXPECT_EQ(log_[0].memOps[1].operation, hipStreamMemOpWriteValue64);
  EXPECT_EQ(log_[0].memOps[1].address, ackAddr);
  EXPECT_EQ(log_[0].memOps[1].value, 0u);
  // ...and the copies follow it.
  EXPECT_EQ(log_[1].kind, Submission::kBatch);
}

// The graph signal is a copy of a device-resident constant into the peer's graph
// signal slot -- a fixed value, not the advancing sequence the non-graph path
// uses, because a captured graph replays the same ops every time.
TEST_F(RmaCePersistTest, Persist_SignallingTask_CopiesDeviceConstantToGraphSignalSlot) {
  PushTask(/*peer=*/1, /*bytes=*/0, /*signal=*/true);
  ncclRmaCeCtx* ceCtx = Ctx(0);
  const char* expectedDst = reinterpret_cast<char*>(&peerSignal_[1])
                          + ceCtx->graphSignalOffset + comm_->rank * sizeof(uint64_t);

  ASSERT_EQ(ncclRmaCePutLaunchUut(comm_.get(), plan_.get(), nullptr), ncclSuccess);

  auto batches = Batches();
  ASSERT_EQ(batches.size(), 2u);
  ASSERT_EQ(batches[1].size(), 1u);
  EXPECT_EQ(batches[1][0].src, ceCtx->signalConstOneDev);
  EXPECT_EQ(batches[1][0].dst, expectedDst);
  EXPECT_EQ(batches[1][0].size, sizeof(uint64_t));
}

// A task that signals nothing needs no handshake, so none is submitted.
TEST_F(RmaCePersistTest, Persist_NonSignallingTask_SubmitsNoAckHandshake) {
  PushTask(/*peer=*/2, /*bytes=*/32, /*signal=*/false);

  ASSERT_EQ(ncclRmaCePutLaunchUut(comm_.get(), plan_.get(), nullptr), ncclSuccess);

  for (const auto& s : log_) EXPECT_NE(s.kind, Submission::kMemOps);
}

// Tasks are issued one batch pair each, in queue order. Two tasks for the same
// peer need no round logic here, because each pair is already ordered against
// the next by the stream.
TEST_F(RmaCePersistTest, Persist_TasksForSamePeer_IssueOneBatchPairEach) {
  PushTask(/*peer=*/1, 32, false);
  PushTask(/*peer=*/1, 48, false);

  ASSERT_EQ(ncclRmaCePutLaunchUut(comm_.get(), plan_.get(), nullptr), ncclSuccess);

  auto batches = Batches();
  ASSERT_EQ(batches.size(), 4u);
  ASSERT_EQ(batches[0].size(), 1u);
  EXPECT_EQ(batches[0][0].size, 32u);
  ASSERT_EQ(batches[2].size(), 1u);
  EXPECT_EQ(batches[2][0].size, 48u);
}

// A task carrying no bytes is signal-only here too.
TEST_F(RmaCePersistTest, Persist_ZeroByteTask_EnqueuesNoDataCopy) {
  PushTask(/*peer=*/2, /*bytes=*/0, /*signal=*/true);

  ASSERT_EQ(ncclRmaCePutLaunchUut(comm_.get(), plan_.get(), nullptr), ncclSuccess);

  auto batches = Batches();
  ASSERT_EQ(batches.size(), 2u);
  EXPECT_TRUE(batches[0].empty());
  EXPECT_EQ(batches[1].size(), 1u);
}

// An unresolvable peer address is rejected rather than copied into.
TEST_F(RmaCePersistTest, Persist_PeerAddressUnresolved_ReturnsInvalidArgument) {
  g_devrGetLsaRankPtr = [](ncclComm*, ncclDevrWindow*, size_t, int, void** outPtr) {
    *outPtr = nullptr;
    return ncclSuccess;
  };
  PushTask(/*peer=*/1, 32, false);

  EXPECT_EQ(ncclRmaCePutLaunchUut(comm_.get(), plan_.get(), nullptr), ncclInvalidArgument);
}

// Both entry points refuse to touch a communicator whose CE state was never
// brought up, rather than dereferencing it.
TEST_F(RmaCeLaunchTest, PutLaunch_CeNotInitialised_ReturnsInternalError) {
  EXPECT_EQ(ncclRmaCePutLaunchUut(comm_.get(), plan_.get(), nullptr), ncclInternalError);
}

TEST_F(RmaCeLaunchTest, WaitLaunch_CeNotInitialised_ReturnsInternalError) {
  EXPECT_EQ(ncclRmaCeWaitLaunchUut(comm_.get(), plan_.get(), nullptr), ncclInternalError);
}

}  // namespace
