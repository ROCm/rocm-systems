/*************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
//
// Host-only microtests for src/group.cc: ncclGroupEndInternal's blocking-vs-non-blocking
// dispatch, planner-state reclaim, async-job launch/abort, the NCCL_API wrappers, the
// group-debug args check, comm-chain unlink/cleanup on the fail path, the job-free callbacks,
// the management-task drain, and the symmetric-register CE-init branch. Non-blocking groups
// must hand pending init work to a background thread and return ncclInProgress immediately,
// never run it synchronously on the caller; ncclCommInitRankConfig(blocking=0) relies on that
// to poll/timeout/abort.
//
// Compiles the hipified src/group.cc directly (GROUP_CC_PATH) so file-static helpers are
// reachable; external symbols come from fakes/ (comm_fakes, collective_stubs, ...). No GPU,
// no librccl.so, no HIP runtime.

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cstdlib>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "nccl.h"
#include "comm.h"
#include "group.h"

#include "fakes/comm_fakes.h"    // controllable ncclCommSetAsyncError seam
#include "fakes/recorder_fakes.h"  // g_recorderResult, shared with the other micro binaries
#include "fakes/nccl_fakes.h"    // g_loadParam, used by param_redirect.h
#include "fakes/collective_stubs.h"  // g_ncclArgsGlobalCheck
#include "fakes/ce_fakes.h"      // g_ncclCeInit, g_ncclCeEnsureAllReduceStaging
#include "fakes/hip_fakes.h"     // g_hipThreadExchangeStreamCaptureMode
#include "fakes/transport_stubs.h"  // g_ncclTransportP2pSetup
#include "ScopedHook.h"          // RAII install/restore for controllable seams

// Route group.cc's NCCL_PARAM sites through g_loadParam instead of the real
// ncclLoadParam (see param_redirect.h); must precede the unit under test.
#include "fakes/param_redirect.h"

// Controllable thread-creation seam: when g_threadCreateShouldFail() is true, the redefined
// STDTHREADCREATE_GOTO takes its failure branch without spawning a thread, so group.cc's
// fail: path is exercisable deterministically. Redefinition survives into group.cc's include.
static std::function<bool()> g_threadCreateShouldFail = [] { return false; };
#include "checks.h"
#undef STDTHREADCREATE_GOTO
#define STDTHREADCREATE_GOTO(var, func, RES, label, ...) \
  do { \
    if (g_threadCreateShouldFail && g_threadCreateShouldFail()) { \
      (RES) = ncclSystemError; \
      goto label; \
    } else { \
      STDTHREADCREATE_IMPL( \
          var, func, \
          do { (RES) = ncclSystemError; goto label; } while (0), \
          __VA_ARGS__); \
    } \
  } while (0)

#include "fakes/nvtx_redirect.h"  // neuter / block nvtx.h before group.cc includes it

// Pulls in group.cc's file-static helpers (groupLaunch, asyncJobLaunch, ...); must come last.
#include GROUP_CC_PATH

namespace {

// Stands in for a real communicator-init job. What's under test is which *thread* runs it.
ncclResult_t FakeInitJobSucceeds(struct ncclAsyncJob*) { return ncclSuccess; }

// Pins that ncclGroupJobComplete (production's own ncclCommGetAsyncError entry point) actually
// propagates a background job's failure, not just ncclGroupJobAbort's parallel path.
ncclResult_t FakeInitJobFails(struct ncclAsyncJob*) { return ncclSystemError; }

// Records isThreadMain, which gates a cudaSetDevice/ncclOsSetAffinity in production.
bool g_lastJobIsThreadMain = false;
ncclResult_t FakeInitJobRecordsIsThreadMain(struct ncclAsyncJob* job) {
  g_lastJobIsThreadMain = job->isThreadMain;
  return ncclSuccess;
}

// Resets every group.cc thread-local. Shared by every fixture that touches this state so
// there's one definition of "clean", not three near-duplicates.
void ResetGroupThreadLocals() {
  ncclGroupDepth = 0;
  ncclGroupError = ncclSuccess;
  for (int type = 0; type < ncclGroupTaskTypeNum; ++type) {
    ncclGroupCommHead[type] = nullptr;
  }
  ncclGroupCommPreconnectHead = nullptr;
  ncclGroupBlocking = -1;
  ncclIntruQueueConstruct(&ncclAsyncJobs);
  ncclProfilerApiState = {};
}

// Releases whatever a test left queued. Call from TearDown: an ASSERT_* in the test body would skip it.
template <typename T, T* T::*next, typename Release>
void DrainQueue(struct ncclIntruQueue<T, next>* queue, Release release) {
  while (!ncclIntruQueueEmpty(queue)) {
    release(ncclIntruQueueDequeue(queue));
  }
}

// Cannot fire today (thread create and nonBlockingInit=true are consecutive in group.cc, and the
// create-failure path deletes groupJob before either runs): forward defence only. Even if it did
// fire, it only half-mitigates: the follow-up ncclGroupJobComplete/Abort call gates its own body,
// refcount decrement included, on this same flag, so it would join the thread here but still leak
// groupJob rather than delete it. Call before handing gj to the real API: a no-op when the flag is
// already true (the normal case).
void ForceJoinIfNonBlockingInitGateIsBroken(struct ncclGroupJob* gj) {
  if (gj && !gj->nonBlockingInit && gj->base.thread.joinable()) {
    COMPILER_ATOMIC_STORE(&gj->abortFlag, true, std::memory_order_relaxed);
    gj->base.thread.join();
  }
}

// (comm, state) pairs, not just state: a recorder that only keeps state can't tell "the right
// comm got the right state" from "some comm got some state", which hides a gate inversion where
// the wrong comm in a multi-comm scenario is the one that got called. Named so it can be spelled
// without a raw template-argument comma inside an EXPECT_EQ(...) macro invocation.
using AsyncState = std::pair<struct ncclComm*, ncclResult_t>;

class GroupEndInternalTest : public ::testing::Test {
 protected:
  std::unique_ptr<ncclComm> comm_;
  std::unique_ptr<ncclAsyncJob> job_;

  // Every ncclCommSetAsyncError (comm, state) pair pushed; the meta thread can write concurrently
  // with a main-thread read, so both sides lock asyncStatesMutex_.
  std::vector<AsyncState> asyncStates_;
  std::mutex asyncStatesMutex_;

  void SetUp() override {
    ResetCommFakes();
    ResetRecorderFakes();  // recorder_fakes.cc is linked here too; g_recorderResult is process state

    // ncclGroupEndInternal runs on this thread, so the thread-locals it inspects are these.
    ResetGroupThreadLocals();

    comm_ = std::make_unique<ncclComm>();  // value-initialised => zeroed
    comm_->groupJob = nullptr;

    g_commSetAsyncError = [this](struct ncclComm* comm, ncclResult_t state) {
      std::lock_guard<std::mutex> lock(asyncStatesMutex_);
      asyncStates_.push_back({comm, state});
      return ncclSuccess;
    };
  }

  void TearDown() override {
    // Join/release a spawned background job before the fixture goes away. A comm on the fail:
    // path already has groupJob nulled by group.cc (it's already deleted), so this is skipped.
    if (comm_ && comm_->groupJob) {
      ForceJoinIfNonBlockingInitGateIsBroken(comm_->groupJob);
      ncclGroupJobComplete(comm_->groupJob);
      comm_->groupJob = nullptr;
    }
    ResetCommFakes();
    ResetRecorderFakes();  // recorder_fakes.cc is linked here too; g_recorderResult is process state
    ResetGroupThreadLocals();  // a failed ASSERT_* before ncclGroupEndInternal would leak depth
  }

  // Queues one pending async job on comm_ and enters a group, mirroring the state
  // ncclCommInitRankConfig leaves just before ncclGroupEnd.
  void EnterGroupWithOnePendingJob(int blocking) {
    comm_->config.blocking = blocking;

    job_ = std::make_unique<ncclAsyncJob>();
    job_->func = FakeInitJobSucceeds;
    job_->comm = comm_.get();
    job_->state = ncclGroupJobRunning;

    ncclGroupStartInternal();  // ncclGroupDepth = 1
    ncclGroupBlocking = blocking;
    ncclIntruQueueEnqueue(&ncclAsyncJobs, job_.get());
  }

  // Unlinks and aborts each distinct groupJob once. All ForceJoins precede any abort: an abort can delete its groupJob.
  void AbortLinkedGroupJobs(std::initializer_list<struct ncclComm*> comms) {
    std::vector<struct ncclGroupJob*> groupJobs;
    for (struct ncclComm* comm : comms) {
      if (comm == nullptr || comm->groupJob == nullptr) {
        continue;
      }
      if (std::find(groupJobs.begin(), groupJobs.end(), comm->groupJob) == groupJobs.end()) {
        groupJobs.push_back(comm->groupJob);
      }
      comm->groupJob = nullptr;
    }
    for (struct ncclGroupJob* groupJob : groupJobs) {
      ForceJoinIfNonBlockingInitGateIsBroken(groupJob);
    }
    for (struct ncclGroupJob* groupJob : groupJobs) {
      ncclGroupJobAbort(groupJob);
    }
  }
};

// A non-blocking group with pending init work must return ncclInProgress (handing off to a
// background thread), not run the jobs synchronously and block forever on a missing peer.
TEST_F(GroupEndInternalTest, NonBlockingGroupWithPendingJob_ReturnsInProgress) {
  EnterGroupWithOnePendingJob(/*blocking=*/0);

  EXPECT_EQ(ncclInProgress, ncclGroupEndInternal());

  // Never joins the background thread, so copy asyncStates_ out under the lock rather than
  // iterate the live vector unlocked.
  std::vector<AsyncState> statesSoFar;
  {
    std::lock_guard<std::mutex> lock(asyncStatesMutex_);
    statesSoFar = asyncStates_;
  }
  EXPECT_NE(statesSoFar.end(),
            std::find(statesSoFar.begin(), statesSoFar.end(), AsyncState{comm_.get(), ncclInProgress}));
}

// The non-blocking path also publishes a group job on the comm, for later poll/abort.
TEST_F(GroupEndInternalTest, NonBlockingGroupWithPendingJob_PublishesGroupJobOnComm) {
  EnterGroupWithOnePendingJob(/*blocking=*/0);

  (void)ncclGroupEndInternal();

  EXPECT_NE(nullptr, comm_->groupJob);
}

// ncclGroupJobComplete is production's real ncclCommGetAsyncError entry point; nothing asserted
// its return value before. A no-op'd ncclAsyncJobComplete call would report ncclSuccess for a
// genuinely failed init -- silently wrong, not a crash.
TEST_F(GroupEndInternalTest, GroupJobComplete_PropagatesBackgroundJobFailure) {
  EnterGroupWithOnePendingJob(/*blocking=*/0);
  job_->func = FakeInitJobFails;

  ASSERT_EQ(ncclInProgress, ncclGroupEndInternal());
  ASSERT_NE(nullptr, comm_->groupJob);
  struct ncclGroupJob* groupJob = comm_->groupJob;
  comm_->groupJob = nullptr;  // this test owns completing it now, not TearDown's own safety net

  EXPECT_EQ(ncclSystemError, ncclGroupJobComplete(groupJob));
}

// If the background thread fails to launch, the fail: path destroys the group job; every
// adopting comm's groupJob must be cleared or a later ncclGroupJobAbort dereferences freed memory.
TEST_F(GroupEndInternalTest, ThreadCreateFailure_LeavesNoDanglingGroupJob) {
  EnterGroupWithOnePendingJob(/*blocking=*/0);
  ScopedHook failThreadCreate(g_threadCreateShouldFail, [] { return true; });

  EXPECT_EQ(ncclSystemError, ncclGroupEndInternal());
  EXPECT_EQ(nullptr, comm_->groupJob);
  EXPECT_EQ(1, failThreadCreate.calls);
}

// Contrast: a blocking group runs jobs on the caller, leaving no group job to poll.
TEST_F(GroupEndInternalTest, BlockingGroupWithPendingJob_RunsSynchronouslyAndReturnsSuccess) {
  EnterGroupWithOnePendingJob(/*blocking=*/1);
  job_->func = FakeInitJobRecordsIsThreadMain;
  g_lastJobIsThreadMain = false;

  EXPECT_EQ(ncclSuccess, ncclGroupEndInternal());
  EXPECT_EQ(nullptr, comm_->groupJob);
  EXPECT_TRUE(g_lastJobIsThreadMain)
    << "the single-job fast path must mark itself as running on the caller's own thread";
}

// Multi-rank symmetric task prep must defer until every local comm has enqueued its job;
// running the first inline can block in bootstrap consensus before its siblings enter it.
TEST_F(GroupEndInternalTest, MultiRankSymmetricCommEnqueuesAsyncJob) {
  auto comm = std::make_unique<ncclComm>();  // value-initialised => zeroed
  comm->intraRanks = 2;
  comm->symmetricSupport = 1;
  comm->p2pCrossClique = 0;

  ncclIntruQueue<ncclAsyncJob, &ncclAsyncJob::next> asyncCollJobs;
  ncclIntruQueueConstruct(&asyncCollJobs);
  ncclSimInfo_t simInfo{};

  ASSERT_EQ(ncclSuccess,
            ncclPrepareTasksAndCollPreconnect(comm.get(), &simInfo, &asyncCollJobs));
  ASSERT_FALSE(ncclIntruQueueEmpty(&asyncCollJobs));

  ncclAsyncJob* queued = ncclIntruQueueDequeue(&asyncCollJobs);
  ASSERT_NE(nullptr, queued);
  EXPECT_EQ(ncclPrepareTasksAndCollPreconnectFunc, queued->func);
  EXPECT_EQ(ncclGroupJobRunning, queued->state);

  auto* prepareJob =
    reinterpret_cast<ncclPrepareTasksAndCollPreconnectJob*>(queued);
  EXPECT_EQ(comm.get(), prepareJob->comm);
  EXPECT_EQ(&simInfo, prepareJob->simInfo);
  EXPECT_TRUE(ncclIntruQueueEmpty(&asyncCollJobs));

  queued->destructor(queued);
}

TEST_F(GroupEndInternalTest, DepthZero_ReturnsInvalidUsage) {
  // ncclGroupDepth is already 0 from SetUp's ResetGroupThreadLocals; no group was ever started.
  EXPECT_EQ(ncclInvalidUsage, ncclGroupEndInternal());
  EXPECT_EQ(0, ncclGroupDepth);
}

TEST_F(GroupEndInternalTest, ProfilerGroupDepthAboveZero_IsDecremented) {
  EnterGroupWithOnePendingJob(/*blocking=*/1);
  job_->func = FakeInitJobSucceeds;
  ncclProfilerApiState.profilerGroupDepth = 2;

  EXPECT_EQ(ncclSuccess, ncclGroupEndInternal());

  EXPECT_EQ(1, ncclProfilerApiState.profilerGroupDepth) << "must decrement exactly once per call";
}

TEST_F(GroupEndInternalTest, InvalidGroupBlockingSentinel_ReturnsInternalError) {
  EnterGroupWithOnePendingJob(/*blocking=*/0);
  job_->func = FakeInitJobSucceeds;
  ncclGroupBlocking = -2;  // neither 0 nor 1: EnterGroupWithOnePendingJob's own value overridden

  EXPECT_EQ(ncclInternalError, ncclGroupEndInternal());
  // Entered at depth 0, so this pins the "> 0" guard; the depth-2 test above passes with it deleted.
  EXPECT_EQ(0, ncclProfilerApiState.profilerGroupDepth) << "the > 0 guard must skip the decrement at depth 0";
}

// Only the outermost ncclGroupEnd dispatches; ncclGroupError is poisoned so a lost early exit returns ncclSystemError.
TEST_F(GroupEndInternalTest, NestedGroupDepthAboveOne_DecrementsAndReturnsSuccessImmediately) {
  ncclGroupDepth = 2;
  ncclGroupError = ncclSystemError;

  EXPECT_EQ(ncclSuccess, ncclGroupEndInternal());

  EXPECT_EQ(1, ncclGroupDepth);
}

// The queued job is load-bearing: with nothing queued, skipping the ncclGroupError guard still returns it unchanged.
TEST_F(GroupEndInternalTest, GroupErrorSet_PropagatesAndRunsFailCleanup) {
  EnterGroupWithOnePendingJob(/*blocking=*/0);
  job_->func = FakeInitJobSucceeds;
  ncclGroupError = ncclSystemError;

  EXPECT_EQ(ncclSystemError, ncclGroupEndInternal());

  EXPECT_EQ(0, ncclGroupDepth);
  // Not ncclAsyncJobs emptiness: groupLocalResetJobState reconstructs that queue on every fail: exit anyway.
  EXPECT_EQ(std::vector<AsyncState>({{comm_.get(), ncclSystemError}}), asyncStates_)
    << "fail: path's groupCleanup must flag the queued job's non-blocking comm";
}

// Drives ncclGroupEndInternal through ncclGroupCommHead[type] directly (never exercised at this
// entry-point level before): the outer hasCommHead scan, and the type-loop that links each
// chained comm to the new groupJob and increments its refcount, mirroring what the asyncJobs
// loop right above it already does for job-based comms.
class GroupCommHeadChainTest : public GroupEndInternalTest {
 protected:
  // Fixture members, not test-local: the background meta-thread can still touch these when an
  // early ASSERT_* exits the test body, so TearDown() must join/abort on every exit path, not
  // just the one path the test body's own explicit drain takes.
  std::unique_ptr<ncclComm> commX_, commY_;
  uint32_t commXAbortFlag_ = 0, commXAbortFlagDev_ = 0, commYAbortFlag_ = 0, commYAbortFlagDev_ = 0;
  // Fixture member, not test-local: commY_->groupJob can still point at this when an early
  // ASSERT_* exits the test body, and TearDown dereferences it (ForceJoin/Abort) before the test
  // ever reaches its own reset line. A test-local here would make that a dangling-pointer read.
  struct ncclGroupJob otherGroupJob_ {};
  // Fixture member, not test-local: TearDown's ncclGroupJobAbort calls can still be joining
  // background threads that could call ncclCeInit; a test-local ScopedHook would restore the
  // real (fail-loud) hook before those joins finish.
  std::optional<ScopedHook<ncclResult_t(struct ncclComm*)>> ceHook_;

  void SetUp() override {
    GroupEndInternalTest::SetUp();
    ceHook_.emplace(g_ncclCeInit, [](struct ncclComm*) { return ncclSuccess; });
  }

  void TearDown() override {
    AbortLinkedGroupJobs({comm_.get(), commX_.get(), commY_.get()});
    GroupEndInternalTest::TearDown();
  }
};

TEST_F(GroupCommHeadChainTest, LinksEachCommAndIncrementsRefCount) {
  // SymRegister, not Collective: the background thread's groupLaunchLegacy processes a non-empty
  // groupCommHead[Collective] unconditionally via ncclPrepareTasksAndCollPreconnect (a real
  // preconnect path, out of scope here). SymRegister's ncclCommGroupRegisterSymmetric is a no-op
  // on a comm with every devrState/ce/suspend queue empty, which these synthetic comms are.
  // Defensive: if that ever stops being true, this asserts 0 calls instead of the whole binary
  // aborting on a background thread via g_ncclCeInit's fail-loud default (ceHook_ installed in
  // SetUp, a fixture member so it outlives TearDown's joins, not just this test body).
  const int type = ncclGroupTaskTypeSymRegister;
  commX_ = std::make_unique<ncclComm>();
  commY_ = std::make_unique<ncclComm>();
  commX_->config.blocking = 0;
  commY_->config.blocking = 0;
  // asyncJobLaunch writes through job->abortFlag/abortFlagDev unconditionally on every job it
  // touches (only childAbortFlag is null-checked), so every comm entering this machinery needs
  // real storage here, matching how a real comm's abortFlag/abortFlagDev are never null.
  commX_->abortFlag = &commXAbortFlag_;
  commX_->abortFlagDev = &commXAbortFlagDev_;
  commY_->abortFlag = &commYAbortFlag_;
  commY_->abortFlagDev = &commYAbortFlagDev_;
  commX_->groupNext[type] = commY_.get();
  commY_->groupNext[type] = nullptr;
  ncclGroupCommHead[type] = commX_.get();

  // commY_ already belongs to a different (fake) groupJob: the type-loop's identity guard
  // (comm->groupJob == NULL) must skip re-adopting it, though the loop keeps walking past it
  // (the async-error call right before the guard is unconditional).
  commY_->groupJob = &otherGroupJob_;

  EnterGroupWithOnePendingJob(/*blocking=*/0);  // also gives the call real work via ncclAsyncJobs
  job_->func = FakeInitJobSucceeds;

  ASSERT_EQ(ncclInProgress, ncclGroupEndInternal());
  ASSERT_NE(nullptr, comm_->groupJob);
  struct ncclGroupJob* groupJob = comm_->groupJob;
  EXPECT_EQ(groupJob, commX_->groupJob) << "the chain's head must adopt the new groupJob";
  EXPECT_EQ(&otherGroupJob_, commY_->groupJob) << "an already-owned comm must not be silently re-adopted";
  // ASSERT: at refcount 1 the abort below deletes groupJob, so every later read (TearDown's too) would be a UAF.
  ASSERT_EQ(2, groupJob->groupRefCount) << "job_'s comm plus commX_ only; commY_ already owned elsewhere";
  EXPECT_EQ(nullptr, ncclGroupCommHead[type]);
  // The type-loop records these on this thread before the spawn, so no join is needed; the lock covers the meta thread.
  {
    std::lock_guard<std::mutex> lock(asyncStatesMutex_);
    EXPECT_NE(asyncStates_.end(),
              std::find(asyncStates_.begin(), asyncStates_.end(), AsyncState{commX_.get(), ncclInProgress}))
      << "commX_ (chain head) must receive ncclCommSetAsyncError";
    EXPECT_NE(asyncStates_.end(),
              std::find(asyncStates_.begin(), asyncStates_.end(), AsyncState{commY_.get(), ncclInProgress}))
      << "commY_ must receive it too even though its groupJob link is skipped (unconditional call)";
  }

  // This abort call is what actually performs the join (ForceJoin is only a forward-defence
  // no-op in the normal case where nonBlockingInit is already true), so groupJob->base.result is
  // only safe to read after it, not before.
  ncclGroupJobAbort(groupJob);  // commX_
  commX_->groupJob = nullptr;
  EXPECT_EQ(ncclSuccess, groupJob->base.result)
    << "the background launch must have actually succeeded, not just been dispatched";
  EXPECT_EQ(0, ceHook_->calls) << "these synthetic comms never populate ceInitTaskQueue";

  commY_->groupJob = nullptr;  // otherGroupJob_ outlives this in TearDown; drop the dangling link
}

// reclaimPlannerState: pure struct manipulation on comm->planner, called from the fail path
// (via groupCleanup) and groupLaunchLegacy's simInfo branch; neither was exercised before.
class ReclaimPlannerStateTest : public ::testing::Test {
 protected:
  std::unique_ptr<ncclComm> comm_;

  void SetUp() override {
    comm_ = std::make_unique<ncclComm>();  // value-initialised => zeroed
    ncclMemoryStackConstruct(&comm_->memPermanent);
    ncclMemoryPoolConstruct(&comm_->memPool_ncclKernelPlan);
    ncclMemoryPoolConstruct(&comm_->memPool_ncclProxyOp);
    ncclIntruQueueConstruct(&comm_->planner.collCleanupQueue);
    ncclIntruQueueConstruct(&comm_->planner.planQueue);
  }

  void TearDown() override {
    ncclMemoryStackDestruct(&comm_->memPermanent);
  }
};

// ncclCommCallback must be first so a `ncclCommCallback*` can be reinterpret_cast back to this.
struct CountingCallback {
  struct ncclCommCallback base;
  int calls = 0;
  struct ncclComm* lastComm = nullptr;
};

// Allocates a plan with proxyOpQueue constructed; does not enqueue it onto planner.planQueue or
// attach any proxyOp, since callers vary on both.
struct ncclKernelPlan* MakePlan(struct ncclComm* comm, bool persistent) {
  struct ncclKernelPlan* plan =
    ncclMemoryPoolAlloc<struct ncclKernelPlan>(&comm->memPool_ncclKernelPlan, &comm->memPermanent);
  plan->persistent = persistent;
  ncclIntruQueueConstruct(&plan->proxyOpQueue);
  return plan;
}

TEST_F(ReclaimPlannerStateTest, DrainsCollCleanupQueueAndReclaimsNonPersistentPlan) {
  CountingCallback cb{};
  cb.base.fn = [](struct ncclComm* comm, struct ncclCommCallback* cbBase) -> ncclResult_t {
    auto* self = reinterpret_cast<CountingCallback*>(cbBase);
    self->calls++;
    self->lastComm = comm;
    return ncclSuccess;
  };
  ncclIntruQueueEnqueue(&comm_->planner.collCleanupQueue, &cb.base);

  struct ncclKernelPlan* plan = MakePlan(comm_.get(), /*persistent=*/false);
  struct ncclProxyOp* pxop =
    ncclMemoryPoolAlloc<struct ncclProxyOp>(&comm_->memPool_ncclProxyOp, &comm_->memPermanent);
  ncclIntruQueueEnqueue(&plan->proxyOpQueue, pxop);
  ncclIntruQueueEnqueue(&comm_->planner.planQueue, plan);

  reclaimPlannerState(comm_.get());

  EXPECT_EQ(1, cb.calls);
  EXPECT_EQ(comm_.get(), cb.lastComm) << "the callback must be invoked with the owning comm, not NULL";
  EXPECT_EQ(reinterpret_cast<struct ncclMemoryPool::Cell*>(plan), comm_->memPool_ncclKernelPlan.head)
    << "the plan cell should be back on the free list";
  // The plan's own proxyOpQueue must drain back to its pool too -- a separate call easy to lose
  // in a refactor without this leaving anything else observably wrong (silent pool leak).
  EXPECT_EQ(reinterpret_cast<struct ncclMemoryPool::Cell*>(pxop), comm_->memPool_ncclProxyOp.head)
    << "the plan's queued proxy op must be pooled back too";
  EXPECT_EQ(reinterpret_cast<struct ncclComm*>(0x1), comm_->preconnectNext);
  EXPECT_EQ(INT_MAX, comm_->planner.bcast_info.minBcastPeer);
  EXPECT_EQ(INT_MIN, comm_->planner.bcast_info.maxBcastPeer);
}

// A persistent plan being skipped looks identical to the whole loop never running -- both leave
// memPool_ncclKernelPlan.head null. Enqueuing a non-persistent plan alongside it and checking
// the pool head lands on that plan specifically proves the loop ran and discriminated.
// (planQueue itself isn't asserted empty: reclaimPlannerState memsets comm->planner afterward
// regardless, so that check can't ever fail.)
TEST_F(ReclaimPlannerStateTest, DequeuesButDoesNotFreeAPersistentPlan) {
  struct ncclKernelPlan* persistentPlan = MakePlan(comm_.get(), /*persistent=*/true);
  struct ncclProxyOp* persistentPxop =
    ncclMemoryPoolAlloc<struct ncclProxyOp>(&comm_->memPool_ncclProxyOp, &comm_->memPermanent);
  ncclIntruQueueEnqueue(&persistentPlan->proxyOpQueue, persistentPxop);
  struct ncclKernelPlan* nonPersistentPlan = MakePlan(comm_.get(), /*persistent=*/false);
  ncclIntruQueueEnqueue(&comm_->planner.planQueue, persistentPlan);
  ncclIntruQueueEnqueue(&comm_->planner.planQueue, nonPersistentPlan);

  reclaimPlannerState(comm_.get());

  auto* head = comm_->memPool_ncclKernelPlan.head;
  ASSERT_EQ(reinterpret_cast<struct ncclMemoryPool::Cell*>(nonPersistentPlan), head)
    << "only the non-persistent plan may be pooled back";
  EXPECT_EQ(nullptr, head->next) << "the persistent plan must not also land in the free list behind it";
  // The persistent plan is still alive and still owns this op, so the proxyOpQueue drain must
  // not touch it either -- catches that drain being hoisted out of the persistence guard.
  EXPECT_EQ(nullptr, comm_->memPool_ncclProxyOp.head)
    << "a persistent plan's proxy ops must not be pooled back while the plan is still alive";
}

// hasSeen is always seeded 1 (so a clear is observable); only p2pOnly/transportComm vary.
void SetConn(struct ncclConnector& conn, int p2pOnly, struct ncclTransportComm* transportComm) {
  conn.p2pOnly = p2pOnly;
  conn.hasSeen = 1;
  conn.transportComm = transportComm;
}

TEST_F(ReclaimPlannerStateTest, ClearsP2pOnlyOnlyForConnectionsWithNoTransportComm) {
  // Needs NCCL_MAX_CONNS >= 3 for "index 0 / middle / last" below to mean distinct slots.
  static_assert(NCCL_MAX_CONNS >= 3, "peer10->send[1] below assumes a middle index exists");
  const int kRanks = 2;
  const int kConns = kRanks * NCCL_MAX_CONNS;
  const int kMaskWords = MAXCHANNELS / CHANNELS_PER_MASK_WORD;
  comm_->nRanks = kRanks;
  auto connectSend = std::make_unique<struct channelMasks[]>(kConns);
  auto connectRecv = std::make_unique<struct channelMasks[]>(kConns);
  for (int i = 0; i < kConns; ++i) {
    for (int j = 0; j < kMaskWords; ++j) {
      connectSend[i].masks[j] = 0xFF;
      connectRecv[i].masks[j] = 0xFF;
    }
  }
  comm_->connectSend = connectSend.get();
  comm_->connectRecv = connectRecv.get();

  // Two channels (one with a null peer, to check the skip doesn't also skip its neighbor),
  // connections at index 0/middle/last (not just index 0), and a p2pOnly==0-but-disconnected
  // case on each of send/recv (peer10) proving the guard is a conjunction, not just transportComm.
  auto peer00 = std::make_unique<struct ncclChannelPeer>();
  SetConn(peer00->send[0], /*p2pOnly=*/1, /*transportComm=*/nullptr);  // disconnected -> cleared
  SetConn(peer00->recv[NCCL_MAX_CONNS - 1], /*p2pOnly=*/1,
          reinterpret_cast<struct ncclTransportComm*>(0x2));  // connected -> survives

  auto peer01 = std::make_unique<struct ncclChannelPeer>();
  SetConn(peer01->send[NCCL_MAX_CONNS - 1], /*p2pOnly=*/1, /*transportComm=*/nullptr);  // disconnected -> cleared
  SetConn(peer01->recv[0], /*p2pOnly=*/1, reinterpret_cast<struct ncclTransportComm*>(0x3));  // connected -> survives

  auto peer10 = std::make_unique<struct ncclChannelPeer>();
  SetConn(peer10->send[1], /*p2pOnly=*/1, /*transportComm=*/nullptr);  // disconnected -> cleared
  SetConn(peer10->send[0], /*p2pOnly=*/0, /*transportComm=*/nullptr);  // not p2pOnly -> survives
  SetConn(peer10->recv[0], /*p2pOnly=*/0, /*transportComm=*/nullptr);  // not p2pOnly -> survives
  SetConn(peer10->recv[1], /*p2pOnly=*/1, /*transportComm=*/nullptr);  // disconnected AND p2pOnly -> cleared

  struct ncclChannelPeer* peersChan0[kRanks] = {peer00.get(), peer01.get()};
  struct ncclChannelPeer* peersChan1[kRanks] = {nullptr, peer10.get()};  // rank 0 has no peer here
  comm_->channels[0].peers = peersChan0;
  comm_->channels[1].peers = peersChan1;

  reclaimPlannerState(comm_.get());

  for (int i = 0; i < kConns; ++i) {
    for (int j = 0; j < kMaskWords; ++j) {
      EXPECT_EQ(0u, connectSend[i].masks[j]) << "connectSend[" << i << "].masks[" << j << "]";
      EXPECT_EQ(0u, connectRecv[i].masks[j]) << "connectRecv[" << i << "].masks[" << j << "]";
    }
  }

  EXPECT_EQ(0, peer00->send[0].p2pOnly);
  EXPECT_EQ(0, peer00->send[0].hasSeen);
  EXPECT_EQ(1, peer00->recv[NCCL_MAX_CONNS - 1].p2pOnly) << "connected peer's last conn slot must survive";
  EXPECT_EQ(1, peer00->recv[NCCL_MAX_CONNS - 1].hasSeen);

  EXPECT_EQ(0, peer01->send[NCCL_MAX_CONNS - 1].p2pOnly);
  EXPECT_EQ(0, peer01->send[NCCL_MAX_CONNS - 1].hasSeen);
  EXPECT_EQ(1, peer01->recv[0].p2pOnly) << "connected peer's first conn slot must survive";
  EXPECT_EQ(1, peer01->recv[0].hasSeen);

  EXPECT_EQ(0, peer10->send[1].p2pOnly) << "the second channel's connections must be reached too";
  EXPECT_EQ(0, peer10->send[1].hasSeen);
  EXPECT_EQ(1, peer10->send[0].hasSeen) << "not p2pOnly, so disconnected alone must not clear it";
  EXPECT_EQ(1, peer10->recv[0].hasSeen) << "not p2pOnly, so disconnected alone must not clear it";
  EXPECT_EQ(0, peer10->recv[1].p2pOnly) << "recv side must also actually clear when both conditions hold";
  EXPECT_EQ(0, peer10->recv[1].hasSeen);

  comm_->channels[0].peers = nullptr;  // outlive peer00/peer01/peersChan0, drop dangling refs first
  comm_->channels[1].peers = nullptr;
  comm_->connectSend = nullptr;  // outlive connectSend/connectRecv (unique_ptr locals) too
  comm_->connectRecv = nullptr;
}

// Known bug reproduction, not correct-behavior coverage: reclaimPlannerState saves/restores
// comm->planner.peers across its memset but not comm->planner.rmaTaskQueues, unlike every other
// call site that touches this struct. rmaTaskQueues is allocated once at comm init and never
// reallocated, so this silently nulls it on any group reaching reclaimPlannerState (even a plain
// ncclGroupSimulateEnd() call); production's RMA enqueue path then indexes it unconditionally and
// segfaults. This will start failing once reclaimPlannerState is fixed to restore rmaTaskQueues
// like it does peers -- delete this test then (or turn it into real coverage).
TEST_F(ReclaimPlannerStateTest, KnownBug_RmaTaskQueuesNotRestoredAfterReclaim) {
  auto rmaTaskQueues =
    std::make_unique<struct ncclIntruQueue<struct ncclTaskRma, &ncclTaskRma::next>[]>(1);
  ncclIntruQueueConstruct(&rmaTaskQueues[0]);
  comm_->config.numRmaCtx = 1;
  comm_->planner.rmaTaskQueues = rmaTaskQueues.get();

  // planner.peers gets a real pointer too, so its survival (the contrast this test draws) is
  // actually pinned rather than assumed: nRanks stays 0, so group.cc's post-memset peers-memset
  // touches zero bytes and can't corrupt it.
  auto peers = std::make_unique<ncclKernelPlanner::Peer[]>(1);
  comm_->planner.peers = peers.get();

  reclaimPlannerState(comm_.get());

  EXPECT_EQ(peers.get(), comm_->planner.peers) << "peers must survive the memset";
  // The bug: this should equal rmaTaskQueues.get(), mirroring how planner.peers just did.
  EXPECT_EQ(nullptr, comm_->planner.rmaTaskQueues);

  comm_->planner.peers = nullptr;  // outlive the local, as elsewhere in this fixture.
}

// ncclAsyncLaunch: the sync-vs-async dispatch gate every async job funnels through.
class AsyncLaunchTest : public ::testing::Test {
 protected:
  std::unique_ptr<ncclComm> comm_;
  ncclAsyncJob job_{};
  int funcCalls_ = 0;
  int undoCalls_ = 0;
  int destructorCalls_ = 0;
  ncclResult_t funcResult_ = ncclSuccess;
  std::vector<std::string> callOrder_;  // records func/undo/destructor in the order they ran

  static AsyncLaunchTest* s_active;

  void SetUp() override {
    ResetGroupThreadLocals();
    comm_ = std::make_unique<ncclComm>();
    s_active = this;
  }

  void TearDown() override {
    // Not just this class's own SetUp: other TUs sharing this binary/thread (e.g. rccl_wrap.cc's
    // rcclDdaEnabled reading ncclGroupDepth) can run next, especially under --gtest_shuffle.
    ResetGroupThreadLocals();
    s_active = nullptr;
  }

  static ncclResult_t Func(struct ncclAsyncJob*) {
    s_active->funcCalls_++;
    s_active->callOrder_.push_back("func");
    return s_active->funcResult_;
  }
  static void Undo(struct ncclAsyncJob*) {
    s_active->undoCalls_++;
    s_active->callOrder_.push_back("undo");
  }
  static void Destructor(void*) {
    s_active->destructorCalls_++;
    s_active->callOrder_.push_back("destructor");
  }
};
AsyncLaunchTest* AsyncLaunchTest::s_active = nullptr;

TEST_F(AsyncLaunchTest, DepthZero_RunsSynchronouslyAndDoesNotEnqueue) {
  comm_->destroyFlag = 1;  // destroyFlag is copied unconditionally, even on this synchronous path

  EXPECT_EQ(ncclSuccess, ncclAsyncLaunch(&job_, Func, Undo, Destructor, comm_.get()));

  EXPECT_EQ(1, funcCalls_);
  EXPECT_EQ(0, undoCalls_) << "undo only runs when func fails";
  EXPECT_EQ(1, destructorCalls_);
  EXPECT_TRUE(ncclIntruQueueEmpty(&ncclAsyncJobs));
  EXPECT_EQ(1, job_.destroyFlag);
}

TEST_F(AsyncLaunchTest, DepthZero_FuncFailureCallsUndoThenDestructor) {
  funcResult_ = ncclSystemError;

  EXPECT_EQ(ncclSystemError, ncclAsyncLaunch(&job_, Func, Undo, Destructor, comm_.get()));

  EXPECT_EQ(1, undoCalls_);
  EXPECT_EQ(1, destructorCalls_);
  EXPECT_EQ(std::vector<std::string>({"func", "undo", "destructor"}), callOrder_)
    << "undo must run before destructor frees the job it operates on";
}

// Every real production caller (all seven, in src/init.cc) passes a null undo; the non-null
// Undo used elsewhere in this file is a test-only convenience that would hide a dropped "&& undo"
// null guard.
TEST_F(AsyncLaunchTest, DepthZero_FuncFailureWithNullUndoStillDestructs) {
  funcResult_ = ncclSystemError;

  EXPECT_EQ(ncclSystemError, ncclAsyncLaunch(&job_, Func, nullptr, Destructor, comm_.get()));

  EXPECT_EQ(0, undoCalls_) << "no undo was given, so none can have run";
  EXPECT_EQ(1, destructorCalls_);
}

TEST_F(AsyncLaunchTest, PositiveDepth_EnqueuesInsteadOfRunning) {
  ncclGroupDepth = 1;
  comm_->config.blocking = 1;
  uint32_t abortFlag = 0, abortFlagDev = 0, childAbortFlag = 0, childAbortFlagDev = 0;
  comm_->abortFlag = &abortFlag;
  comm_->abortFlagDev = &abortFlagDev;
  comm_->childAbortFlag = &childAbortFlag;
  comm_->childAbortFlagDev = &childAbortFlagDev;
  job_.state = ncclGroupJobJoined;  // seeded away from ncclGroupJobRunning's 0 default

  EXPECT_EQ(ncclSuccess, ncclAsyncLaunch(&job_, Func, Undo, Destructor, comm_.get()));

  EXPECT_EQ(0, funcCalls_) << "must not run on the caller's thread at positive depth";
  EXPECT_FALSE(ncclIntruQueueEmpty(&ncclAsyncJobs));
  EXPECT_EQ(&job_, ncclIntruQueueHead(&ncclAsyncJobs));
  EXPECT_EQ(comm_.get(), job_.comm);
  EXPECT_EQ(ncclGroupJobRunning, job_.state);
  // asyncJobLaunch/ncclAsyncJobMain call through these three; nothing else here observes them.
  EXPECT_EQ(&Func, job_.func);
  EXPECT_EQ(&Undo, job_.undo);
  EXPECT_EQ(&Destructor, job_.destructor);
  EXPECT_EQ(comm_->abortFlag, job_.abortFlag);
  EXPECT_EQ(comm_->abortFlagDev, job_.abortFlagDev);
  EXPECT_EQ(comm_->childAbortFlag, job_.childAbortFlag);
  EXPECT_EQ(comm_->childAbortFlagDev, job_.childAbortFlagDev);

  // Outlive the locals above (comm_/job_ live on as fixture members past this function).
  comm_->abortFlag = comm_->abortFlagDev = comm_->childAbortFlag = comm_->childAbortFlagDev = nullptr;
  job_.abortFlag = job_.abortFlagDev = job_.childAbortFlag = job_.childAbortFlagDev = nullptr;
}

TEST_F(AsyncLaunchTest, PositiveDepth_FirstCommSetsGroupBlockingFromItsConfig) {
  ncclGroupDepth = 1;
  comm_->config.blocking = 1;

  ASSERT_EQ(ncclSuccess, ncclAsyncLaunch(&job_, Func, Undo, Destructor, comm_.get()));

  EXPECT_EQ(1, ncclGroupBlocking);
}

// Mirrors the case above with config.blocking = 0: distinguishes "copies comm's config" from a
// mutant that hardcodes ncclGroupBlocking = 1 (which the blocking=1 case above can't catch).
TEST_F(AsyncLaunchTest, PositiveDepth_FirstCommSetsGroupBlockingFromItsConfig_NonBlocking) {
  ncclGroupDepth = 1;
  comm_->config.blocking = 0;

  ASSERT_EQ(ncclSuccess, ncclAsyncLaunch(&job_, Func, Undo, Destructor, comm_.get()));

  EXPECT_EQ(0, ncclGroupBlocking);
}

TEST_F(AsyncLaunchTest, PositiveDepth_MismatchedBlockingModeRejectsWithoutEnqueuing) {
  ncclGroupDepth = 1;
  ncclGroupBlocking = 1;  // an earlier comm in this group was blocking
  comm_->config.blocking = 0;  // this one is not

  EXPECT_EQ(ncclInvalidArgument, ncclAsyncLaunch(&job_, Func, Undo, Destructor, comm_.get()));

  EXPECT_TRUE(ncclIntruQueueEmpty(&ncclAsyncJobs)) << "a rejected job is never queued";
  EXPECT_EQ(0, funcCalls_);
  EXPECT_EQ(0, undoCalls_) << "the job never ran, so there is nothing to undo";
  EXPECT_EQ(1, destructorCalls_) << "still freed even though it was never queued";
}

TEST_F(AsyncLaunchTest, DestroyFlagForcesBlockingRegardlessOfConfig) {
  ncclGroupDepth = 1;
  comm_->destroyFlag = 1;
  comm_->config.blocking = 0;

  ASSERT_EQ(ncclSuccess, ncclAsyncLaunch(&job_, Func, Undo, Destructor, comm_.get()));

  EXPECT_EQ(1, ncclGroupBlocking);
  EXPECT_FALSE(ncclIntruQueueEmpty(&ncclAsyncJobs));
  EXPECT_EQ(1, job_.destroyFlag);
}

// Spins on *job->abortFlag until nonzero or a 5s deadline elapses. Used by the ordering-sensitive
// TwoOwners case: a job that returns instantly can't distinguish "abortFlag set before the join"
// from "after". The deadline unwinds a swapped-order mutant (self-reports via ncclSystemError)
// instead of hanging; the happy path still returns in microseconds.
ncclResult_t SpinUntilAbortFlagObserved(struct ncclAsyncJob* job) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (std::chrono::steady_clock::now() < deadline) {
    if (job->abortFlag && COMPILER_ATOMIC_LOAD(job->abortFlag, std::memory_order_acquire) != 0) {
      return ncclSuccess;
    }
    std::this_thread::sleep_for(std::chrono::microseconds(200));
  }
  return ncclSystemError;
}

// ncclGroupJobAbort mirrors ncclGroupJobComplete but also raises abortFlag before joining, and
// leaves comm->groupJob dangling afterward (unlike the fail: path, which unlinks it) -- callers
// must not touch it again once abort has run.
class GroupJobAbortTest : public GroupEndInternalTest {
 protected:
  // Fixture members, not TwoOwners-local: the background meta-thread can still touch these when
  // an early ASSERT_* exits the test body, so TearDown() must join it on every exit path, not
  // just the one the test body's own two abort calls take.
  std::unique_ptr<ncclComm> comm2_;
  ncclAsyncJob job2_{};
  uint32_t abortFlagStorage1_ = 0, abortFlagDevStorage1_ = 0;
  uint32_t abortFlagStorage2_ = 0, abortFlagDevStorage2_ = 0;

  void TearDown() override {
    // One abort per distinct groupJob, never a refcount drain: an early ASSERT_* may leak (LSan-visible), not UAF.
    AbortLinkedGroupJobs({comm_.get(), comm2_.get()});
    GroupEndInternalTest::TearDown();
  }
};

TEST_F(GroupJobAbortTest, NotNonBlockingInit_ReturnsSuccessWithoutJoining) {
  struct ncclGroupJob groupJob{};
  groupJob.nonBlockingInit = false;  // explicit: the "no job" shape, not an incidental zero-init
  groupJob.groupRefCount = 7;  // seeded away from 0 so an out-of-guard decrement is observable

  EXPECT_EQ(ncclSuccess, ncclGroupJobAbort(&groupJob));

  EXPECT_FALSE(groupJob.joined) << "nothing to join, so the exchange must not fire";
  EXPECT_EQ(7, groupJob.groupRefCount) << "the decrement lives inside the nonBlockingInit guard";
}

// Mirrors the case above against ncclGroupJobComplete, its structural twin: same nonBlockingInit
// guard around the same joined-exchange and refcount decrement.
TEST_F(GroupJobAbortTest, Complete_NotNonBlockingInit_ReturnsSuccessWithoutJoining) {
  struct ncclGroupJob groupJob{};
  groupJob.nonBlockingInit = false;
  groupJob.groupRefCount = 7;

  EXPECT_EQ(ncclSuccess, ncclGroupJobComplete(&groupJob));

  EXPECT_FALSE(groupJob.joined) << "nothing to join, so the exchange must not fire";
  EXPECT_EQ(7, groupJob.groupRefCount) << "the decrement lives inside the nonBlockingInit guard";
}

// A single-owner call can't distinguish "really joined" from "did nothing" (return is always
// ncclSuccess, object deleted at refcount 0). Two comms sharing one groupJob keeps it alive after
// the first abort, long enough to inspect it, and the second call exercises the joined-exchange
// guard against a double join.
TEST_F(GroupJobAbortTest, NonBlockingInit_TwoOwners_FirstCallAbortsAndJoins_SecondCallIsANoop) {
  // 2 queued jobs take asyncJobLaunch's polling branch, which needs real (non-null) abortFlag
  // storage. job_/job2_ spin on the forwarded store (SpinUntilAbortFlagObserved), pinning the
  // store-before-join order. Results are seeded off ncclSuccess so "both report success" can't
  // be satisfied by the jobs never having run.
  comm2_ = std::make_unique<ncclComm>();
  comm2_->config.blocking = 0;
  job2_.func = SpinUntilAbortFlagObserved;
  job2_.result = ncclInternalError;
  job2_.comm = comm2_.get();
  job2_.state = ncclGroupJobRunning;
  job2_.abortFlag = &abortFlagStorage2_;
  job2_.abortFlagDev = &abortFlagDevStorage2_;

  EnterGroupWithOnePendingJob(/*blocking=*/0);  // queues comm_/job_ and starts the group
  job_->func = SpinUntilAbortFlagObserved;
  job_->result = ncclInternalError;
  job_->abortFlag = &abortFlagStorage1_;
  job_->abortFlagDev = &abortFlagDevStorage1_;
  ncclIntruQueueEnqueue(&ncclAsyncJobs, &job2_);

  ASSERT_EQ(ncclInProgress, ncclGroupEndInternal());
  ASSERT_NE(nullptr, comm_->groupJob);
  ASSERT_EQ(comm_->groupJob, comm2_->groupJob) << "both comms must adopt the same groupJob";
  struct ncclGroupJob* groupJob = comm_->groupJob;
  ASSERT_EQ(2, groupJob->groupRefCount);
  // ncclGroupJobAbort's entire body is gated on this flag; assert it here so a regression that
  // clears it is attributed correctly instead of masked by the no-op call still returning success.
  ASSERT_TRUE(groupJob->nonBlockingInit);

  EXPECT_EQ(ncclSuccess, ncclGroupJobAbort(groupJob));

  // refcount 2 -> 1: the object is still alive, safe to inspect.
  EXPECT_TRUE(groupJob->joined);
  EXPECT_TRUE(groupJob->abortFlag);
  EXPECT_EQ(1, groupJob->groupRefCount);
  // Deterministic proof the join ran: a dropped ncclAsyncJobComplete call would otherwise pass
  // every assertion above while leaving the meta thread running.
  EXPECT_FALSE(groupJob->base.thread.joinable());
  // Both spinners observing success is only possible if the store happened before the join
  // returned above; a swapped-order mutant times both out to ncclSystemError instead.
  EXPECT_EQ(ncclSuccess, job_->result);
  EXPECT_EQ(ncclSuccess, job2_.result);
  // The device-visible abort flags asyncJobLaunch forwards; nothing above observes abortFlagDev.
  EXPECT_EQ(1u, abortFlagStorage1_);
  EXPECT_EQ(1u, abortFlagDevStorage1_);
  EXPECT_EQ(1u, abortFlagStorage2_);
  EXPECT_EQ(1u, abortFlagDevStorage2_);

  // Confirms the second call below still lands in the gated body: a regression flipping this
  // back to false would silently leak groupJob instead of deleting it.
  EXPECT_TRUE(groupJob->nonBlockingInit);

  // Decrements refcount 1 -> 0 and deletes groupJob; joined must make this a no-op, not a
  // second pthread_join on the same thread.
  EXPECT_EQ(ncclSuccess, ncclGroupJobAbort(groupJob));

  comm_->groupJob = nullptr;  // groupJob is gone now; drop both comms' dangling pointers to it.
  comm2_->groupJob = nullptr;
}

// The bodies behind the three NCCL_API entry points; ncclGroupStart/End enter via _impl.
// ncclGroupEndInternal's own guard returns (depth-zero, nested-depth, ncclGroupError,
// invalid-blocking) are covered directly on GroupEndInternalTest instead, since that fixture
// already builds the comm/job state some of them need.
class GroupApiWrapperTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ResetGroupThreadLocals();
    ResetRecorderFakes();
  }
  void TearDown() override {
    ResetGroupThreadLocals();  // several tests here deliberately leave ncclGroupDepth nonzero
    ResetRecorderFakes();
  }
};

TEST_F(GroupApiWrapperTest, GroupStart_IncrementsDepthAndReturnsSuccess) {
  EXPECT_EQ(ncclSuccess, ncclGroupStart_impl());
  EXPECT_EQ(1, ncclGroupDepth);
}

TEST_F(GroupApiWrapperTest, GroupStart_RecorderFailure_PropagatesAndLeavesDepthAtZero) {
  g_recorderResult = ncclSystemError;

  EXPECT_EQ(ncclSystemError, ncclGroupStart_impl());
  EXPECT_EQ(0, ncclGroupDepth) << "NCCLCHECK must short-circuit before ncclGroupStartInternal runs";
}

TEST_F(GroupApiWrapperTest, GroupEnd_NoPendingWork_DecrementsDepthAndReturnsSuccess) {
  ncclGroupDepth = 1;

  EXPECT_EQ(ncclSuccess, ncclGroupEnd_impl());
  EXPECT_EQ(0, ncclGroupDepth);
}

TEST_F(GroupApiWrapperTest, GroupEnd_RecorderFailure_PropagatesAndLeavesDepthUnchanged) {
  ncclGroupDepth = 1;
  g_recorderResult = ncclSystemError;

  EXPECT_EQ(ncclSystemError, ncclGroupEnd_impl());
  EXPECT_EQ(1, ncclGroupDepth) << "NCCLCHECK must short-circuit before ncclGroupEndInternal runs";
}

TEST_F(GroupApiWrapperTest, GroupSimulateEnd_NoPendingWork_DecrementsDepthAndReturnsSuccess) {
  ncclGroupDepth = 1;
  ncclSimInfo_t simInfo = NCCL_SIM_INFO_INITIALIZER;

  EXPECT_EQ(ncclSuccess, ncclGroupSimulateEnd(&simInfo));
  EXPECT_EQ(0, ncclGroupDepth);
}

// A zero-initialised simInfo (magic 0, not NCCL_SIM_INFO_INITIALIZER's real value) is rejected
// only if the pointer itself really reaches ncclGroupEndInternal's own magic check.
TEST_F(GroupApiWrapperTest, GroupSimulateEnd_ForwardsSimInfoBadMagicIsRejected) {
  ncclGroupDepth = 1;
  ncclSimInfo_t simInfo{};  // zero-initialised, not via NCCL_SIM_INFO_INITIALIZER: magic stays 0

  EXPECT_EQ(ncclInvalidArgument, ncclGroupSimulateEnd(&simInfo));
  EXPECT_EQ(0, ncclGroupDepth) << "depth is decremented before the magic check either way";
}

// ncclCommGroupArgsGlobalCheck: dequeues comm->argsInfoQueue, calls ncclArgsGlobalCheck per
// entry. g_ncclArgsGlobalCheck (fakes/collective_stubs.h) is this file's one controllable
// exception to that symbol's usual fail-loud floor.
class ArgsGlobalCheckTest : public ::testing::Test {
 protected:
  std::unique_ptr<ncclComm> comm_;
  struct ncclGroupDebugJob job_{};

  void SetUp() override {
    comm_ = std::make_unique<ncclComm>();
    ncclIntruQueueConstruct(&comm_->argsInfoQueue);
    job_.comm = comm_.get();
    ResetCollectiveStubs();
  }

  void TearDown() override {
    DrainQueue(&comm_->argsInfoQueue, free);
    ResetCollectiveStubs();
  }

  struct ncclArgsInfo* EnqueueOne() {
    auto* info = static_cast<struct ncclArgsInfo*>(calloc(1, sizeof(struct ncclArgsInfo)));
    ncclIntruQueueEnqueue(&comm_->argsInfoQueue, info);
    return info;
  }
};

TEST_F(ArgsGlobalCheckTest, EmptyQueue_ReturnsSuccessWithoutCallingTheHook) {
  ScopedHook hook(g_ncclArgsGlobalCheck, [](struct ncclArgsInfo*) { return ncclSuccess; });

  EXPECT_EQ(ncclSuccess, ncclCommGroupArgsGlobalCheck(&job_.base));

  EXPECT_EQ(0, hook.calls);
}

// Whether production's own free(argsInfo) runs isn't observable here without a sanitizer; this
// pins that every entry is dequeued and seen by the hook, in order, which is what's checkable.
TEST_F(ArgsGlobalCheckTest, NonEmptyQueue_CallsHookOncePerEntryInOrder) {
  struct ncclArgsInfo* first = EnqueueOne();
  struct ncclArgsInfo* second = EnqueueOne();
  std::vector<struct ncclArgsInfo*> seen;
  ScopedHook hook(g_ncclArgsGlobalCheck, [&](struct ncclArgsInfo* info) {
    seen.push_back(info);
    return ncclSuccess;
  });

  EXPECT_EQ(ncclSuccess, ncclCommGroupArgsGlobalCheck(&job_.base));

  EXPECT_EQ(2, hook.calls);
  EXPECT_EQ(std::vector<struct ncclArgsInfo*>({first, second}), seen);
  EXPECT_TRUE(ncclIntruQueueEmpty(&comm_->argsInfoQueue));
}

TEST_F(ArgsGlobalCheckTest, HookFailsOnSecondEntry_StopsDequeuingAndPropagatesTheError) {
  EnqueueOne();
  struct ncclArgsInfo* failing = EnqueueOne();
  struct ncclArgsInfo* neverReached = EnqueueOne();
  ScopedHook hook(g_ncclArgsGlobalCheck, [&](struct ncclArgsInfo* info) {
    return info == failing ? ncclInternalError : ncclSuccess;
  });

  EXPECT_EQ(ncclInternalError, ncclCommGroupArgsGlobalCheck(&job_.base));

  EXPECT_EQ(2, hook.calls);
  ASSERT_FALSE(ncclIntruQueueEmpty(&comm_->argsInfoQueue));
  EXPECT_EQ(neverReached, ncclIntruQueueHead(&comm_->argsInfoQueue));
  // `failing` is deliberately left unfreed here: production's fail: path dequeues without
  // freeing it (mirrored here), and no test-side check can tell that apart from "already freed
  // by a future production change" -- guessing wrong there double-frees. Leaks one entry under
  // ASan, same as ~18 other pre-existing allocations in this binary; not a new gate.
}

// groupJobUnlinkComms: undoes the comm->groupJob links the non-blocking path published, so a
// later ncclGroupJobAbort on a stale groupJob can't dereference freed memory. Only clears comms
// that adopted this exact groupJob; a comm reassigned to a different groupJob must survive.
class GroupJobUnlinkCommsTest : public ::testing::Test {
 protected:
  struct ncclGroupJob groupJob_ {};
  struct ncclGroupJob otherGroupJob_ {};
  std::unique_ptr<ncclComm> commA_, commB_, commOther_, commChain1_, commChain2_, commChainOther_;
  ncclAsyncJob jobA_ {}, jobB_ {}, jobOther_ {};

  void SetUp() override {
    commA_ = std::make_unique<ncclComm>();
    commB_ = std::make_unique<ncclComm>();
    commOther_ = std::make_unique<ncclComm>();
    commChain1_ = std::make_unique<ncclComm>();
    commChain2_ = std::make_unique<ncclComm>();
    commChainOther_ = std::make_unique<ncclComm>();
    ncclIntruQueueConstruct(&groupJob_.asyncJobs);
  }
};

TEST_F(GroupJobUnlinkCommsTest, ClearsOnlyCommsThatAdoptedThisGroupJob) {
  // asyncJobs list: commA_/commB_ adopted groupJob_, commOther_ adopted a different one.
  commA_->groupJob = &groupJob_;
  commB_->groupJob = &groupJob_;
  commOther_->groupJob = &otherGroupJob_;
  jobA_.comm = commA_.get();
  jobB_.comm = commB_.get();
  jobOther_.comm = commOther_.get();
  ncclIntruQueueEnqueue(&groupJob_.asyncJobs, &jobA_);
  ncclIntruQueueEnqueue(&groupJob_.asyncJobs, &jobB_);
  ncclIntruQueueEnqueue(&groupJob_.asyncJobs, &jobOther_);

  // groupCommHead chain (one task type): commChain1_/commChain2_ adopted groupJob_,
  // commChainOther_ (in the middle of the chain) adopted a different one.
  const int type = ncclGroupTaskTypeCollective;
  commChain1_->groupJob = &groupJob_;
  commChainOther_->groupJob = &otherGroupJob_;
  commChain2_->groupJob = &groupJob_;
  commChain1_->groupNext[type] = commChainOther_.get();
  commChainOther_->groupNext[type] = commChain2_.get();
  commChain2_->groupNext[type] = nullptr;
  groupJob_.groupCommHead[type] = commChain1_.get();

  groupJobUnlinkComms(&groupJob_);

  EXPECT_EQ(nullptr, commA_->groupJob);
  EXPECT_EQ(nullptr, commB_->groupJob);
  EXPECT_EQ(&otherGroupJob_, commOther_->groupJob) << "a different groupJob's link must survive";
  EXPECT_EQ(nullptr, commChain1_->groupJob);
  EXPECT_EQ(nullptr, commChain2_->groupJob) << "the chain's far end must be reached too, not just the head";
  EXPECT_EQ(&otherGroupJob_, commChainOther_->groupJob)
    << "a mid-chain comm on a different groupJob must survive, and its neighbor must still be reached";
}

// The fail-path shape of a group with comm heads but no async jobs, e.g. a collective-only group.
TEST_F(GroupJobUnlinkCommsTest, EmptyAsyncJobsStillClearsTheCommHeadChain) {
  const int type = ncclGroupTaskTypeCollective;
  commChain1_->groupJob = &groupJob_;
  commChain2_->groupJob = &groupJob_;
  commChain1_->groupNext[type] = commChain2_.get();
  commChain2_->groupNext[type] = nullptr;
  groupJob_.groupCommHead[type] = commChain1_.get();

  // Forked first: without the empty-queue guard the null head segfaults, which this reports as a plain failure.
  ASSERT_EXIT(
      {
        groupJobUnlinkComms(&groupJob_);
        std::exit(0);
      },
      ::testing::ExitedWithCode(0), "");
  groupJobUnlinkComms(&groupJob_);

  EXPECT_EQ(nullptr, commChain1_->groupJob);
  EXPECT_EQ(nullptr, commChain2_->groupJob);
}

// ~ncclAsyncJob joins a joinable thread, so a free callback that skips its delete returns before this flag is set.
std::shared_ptr<std::atomic<bool>> StartSlowJobThread(struct ncclAsyncJob* base) {
  auto finished = std::make_shared<std::atomic<bool>>(false);
  base->thread = std::thread([finished] {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    finished->store(true);
  });
  return finished;
}

TEST(JobFreeCallbacksTest, TaskPrepareJobFreeDestroysTheJob) {
  auto* job = new ncclTaskPrepareJob();
  auto finished = StartSlowJobThread(&job->base);
  ncclTaskPrepareJobFree(job);
  EXPECT_TRUE(finished->load());
}

TEST(JobFreeCallbacksTest, MgmtTaskJobFreeDestroysTheJob) {
  auto* job = new ncclMgmtTaskJob();
  auto finished = StartSlowJobThread(&job->base);
  ncclMgmtTaskJobFree(job);
  EXPECT_TRUE(finished->load());
}

TEST(JobFreeCallbacksTest, PreconnectJobFreeDestroysTheJob) {
  auto* job = new ncclPreconnectJob();
  // Real storage, not nullptr: ncclCollPreconnectFunc owns and frees it; a destructor that also did would double-free.
  job->algoNeedConnect = new bool[1];
  bool* algoNeedConnect = job->algoNeedConnect;  // save before job itself is freed below
  auto finished = StartSlowJobThread(&job->base);
  ncclPreconnectJobFree(job);
  EXPECT_TRUE(finished->load());
  delete[] algoNeedConnect;
}

TEST(JobFreeCallbacksTest, GroupSymmetricJobFreeDestroysTheJob) {
  auto* job = new ncclGroupSymmetricJob();
  auto finished = StartSlowJobThread(&job->base);
  ncclGroupSymmetricJobFree(job);
  EXPECT_TRUE(finished->load());
}

// Of the sites installing the callbacks above, only this job is reachable mid-run: it is on gjob->asyncJobs.
TEST(JobFreeCallbacksTest, LegacyP2pPreconnectJobIsWiredToPreconnectJobFree) {
  auto comm = std::make_unique<ncclComm>();
  uint32_t abortFlag = 0, abortFlagDev = 0;
  comm->abortFlag = &abortFlag;
  comm->abortFlagDev = &abortFlagDev;
  comm->config.blocking = 1;  // keeps the final drain off ncclCommSetAsyncError
  auto gjob = std::make_unique<ncclGroupJob>();
  ncclIntruQueueConstruct(&gjob->asyncJobs);
  gjob->groupCommPreconnectHead = comm.get();
  void (*destructor)(void*) = nullptr;
  ScopedHook mode(g_hipThreadExchangeStreamCaptureMode, [](hipStreamCaptureMode* m) {
    *m = hipStreamCaptureModeRelaxed;
    return hipSuccess;
  });
  ScopedHook setup(g_ncclTransportP2pSetup, [&](struct ncclComm*, struct ncclTopoGraph*, int, bool*) {
    destructor = ncclIntruQueueHead(&gjob->asyncJobs)->destructor;
    return ncclSuccess;
  });

  EXPECT_EQ(ncclSuccess, groupLaunchLegacy(&gjob->base));

  ASSERT_EQ(1, setup.calls);
  EXPECT_EQ(&ncclPreconnectJobFree, destructor);
  EXPECT_TRUE(ncclIntruQueueEmpty(&gjob->asyncJobs)) << "the job must be drained and freed after launch";
}

// ncclMgmtTaskJobFunc: drains comm->mgmtTaskQueue, running each queued task's own func then
// destructor. destroyFlag=true dequeues and runs exactly one task; false drains the whole queue.
class MgmtTaskJobFuncTest : public ::testing::Test {
 protected:
  std::unique_ptr<ncclComm> comm_;
  struct ncclMgmtTaskJob job_ {};
  std::vector<std::string> callOrder_;

  void SetUp() override {
    comm_ = std::make_unique<ncclComm>();
    ncclIntruQueueConstruct(&comm_->mgmtTaskQueue);
    job_.comm = comm_.get();
    // isThreadMain=true short-circuits production's ncclOsCpuCount/ncclOsSetAffinity call, which
    // this binary hard-aborts on (collective_stubs.cc's "fail-loud floor" for that seam).
    job_.base.isThreadMain = true;
  }

  void TearDown() override {
    DrainQueue(&comm_->mgmtTaskQueue, [](struct ncclAsyncJob* task) { delete static_cast<RecordingTask*>(task); });
  }

  // Each queued task records into callOrder_ via its own func/destructor; funcResult_ controls
  // what the func returns (only meaningful for the task built with FailingFunc below).
  ncclResult_t funcResult_ = ncclSuccess;

  struct RecordingTask : ncclAsyncJob {
    MgmtTaskJobFuncTest* owner;
    std::string label;
  };

  RecordingTask* MakeTask(const std::string& label, ncclResult_t (*func)(struct ncclAsyncJob*)) {
    auto* task = new RecordingTask();
    task->owner = this;
    task->label = label;
    task->func = func;
    task->destructor = [](void* t) {
      auto* self = static_cast<RecordingTask*>(static_cast<ncclAsyncJob*>(t));
      self->owner->callOrder_.push_back(self->label + ":destructor");
      delete self;
    };
    return task;
  }

  static ncclResult_t SucceedingFunc(struct ncclAsyncJob* t) {
    auto* self = static_cast<RecordingTask*>(t);
    self->owner->callOrder_.push_back(self->label + ":func");
    return ncclSuccess;
  }
  static ncclResult_t FailingFunc(struct ncclAsyncJob* t) {
    auto* self = static_cast<RecordingTask*>(t);
    self->owner->callOrder_.push_back(self->label + ":func");
    return self->owner->funcResult_;
  }
};

TEST_F(MgmtTaskJobFuncTest, NotDestroyFlag_DrainsWholeQueueRunningFuncThenDestructorPerTask) {
  ncclIntruQueueEnqueue(&comm_->mgmtTaskQueue, static_cast<ncclAsyncJob*>(MakeTask("first", SucceedingFunc)));
  ncclIntruQueueEnqueue(&comm_->mgmtTaskQueue, static_cast<ncclAsyncJob*>(MakeTask("second", SucceedingFunc)));
  comm_->destroyFlag = 0;

  EXPECT_EQ(ncclSuccess, ncclMgmtTaskJobFunc(&job_.base));

  EXPECT_EQ(std::vector<std::string>({"first:func", "first:destructor", "second:func", "second:destructor"}),
            callOrder_);
  EXPECT_TRUE(ncclIntruQueueEmpty(&comm_->mgmtTaskQueue));
}

TEST_F(MgmtTaskJobFuncTest, DestroyFlag_DequeuesAndRunsExactlyOneTask) {
  ncclIntruQueueEnqueue(&comm_->mgmtTaskQueue, static_cast<ncclAsyncJob*>(MakeTask("first", SucceedingFunc)));
  ncclIntruQueueEnqueue(&comm_->mgmtTaskQueue, static_cast<ncclAsyncJob*>(MakeTask("second", SucceedingFunc)));
  comm_->destroyFlag = 1;

  EXPECT_EQ(ncclSuccess, ncclMgmtTaskJobFunc(&job_.base));

  EXPECT_EQ(std::vector<std::string>({"first:func", "first:destructor"}), callOrder_)
    << "destroyFlag must dequeue and run only the head task, leaving the rest queued";
  EXPECT_FALSE(ncclIntruQueueEmpty(&comm_->mgmtTaskQueue));
}

TEST_F(MgmtTaskJobFuncTest, FailingTaskStopsDequeuingAndStillRunsItsOwnDestructor) {
  funcResult_ = ncclSystemError;
  ncclIntruQueueEnqueue(&comm_->mgmtTaskQueue, static_cast<ncclAsyncJob*>(MakeTask("failing", FailingFunc)));
  auto* neverReached = MakeTask("neverReached", SucceedingFunc);
  ncclIntruQueueEnqueue(&comm_->mgmtTaskQueue, static_cast<ncclAsyncJob*>(neverReached));
  comm_->destroyFlag = 0;

  EXPECT_EQ(ncclSystemError, ncclMgmtTaskJobFunc(&job_.base));

  EXPECT_EQ(std::vector<std::string>({"failing:func", "failing:destructor"}), callOrder_);
  EXPECT_EQ(neverReached, ncclIntruQueueHead(&comm_->mgmtTaskQueue));
}

// ncclCommGroupRegisterSymmetric: only the ceInitTaskQueue branch. rmaCeInitTaskQueue is not
// covered because this binary links the real src/rma/rma_ce.cc, whose ncclRmaCeInit needs the
// dev_runtime fixture; the other four queues need that fixture or new seams. Deferred.
class GroupRegisterSymmetricCeInitTest : public ::testing::Test {
 protected:
  std::unique_ptr<ncclComm> comm_;
  std::unique_ptr<ncclComm> taskComm_;  // deliberately distinct from comm_/job_.comm; see task1->comm below.
  struct ncclGroupSymmetricJob job_ {};

  void SetUp() override {
    comm_ = std::make_unique<ncclComm>();
    taskComm_ = std::make_unique<ncclComm>();
    ncclIntruQueueConstruct(&comm_->devrState.regTaskQueue);
    ncclIntruQueueConstruct(&comm_->devrState.commCreateTaskQueue);
    ncclIntruQueueConstruct(&comm_->ceInitTaskQueue);
    ncclIntruQueueConstruct(&comm_->rmaCeInitTaskQueue);
    ncclIntruQueueConstruct(&comm_->suspendTaskQueue);
    ncclIntruQueueConstruct(&comm_->resumeTaskQueue);
    job_.comm = comm_.get();
    ResetCeFakes();
  }

  void TearDown() override {
    DrainQueue(&comm_->ceInitTaskQueue, free);
    ResetCeFakes();
  }
};

TEST_F(GroupRegisterSymmetricCeInitTest, DrainsCeInitTaskQueueCallingHookPerEntry) {
  auto* task1 = static_cast<struct ncclCeInitTask*>(calloc(1, sizeof(struct ncclCeInitTask)));
  // Not job_.comm: production calls ncclCeInit(task->comm), and a regression to the job's comm hides if they match.
  task1->comm = taskComm_.get();
  ncclIntruQueueEnqueue(&comm_->ceInitTaskQueue, task1);
  // A second task on a second comm: with one entry, a drain loop and a single dequeue are indistinguishable.
  auto secondTaskComm = std::make_unique<ncclComm>();
  auto* task2 = static_cast<struct ncclCeInitTask*>(calloc(1, sizeof(struct ncclCeInitTask)));
  task2->comm = secondTaskComm.get();
  ncclIntruQueueEnqueue(&comm_->ceInitTaskQueue, task2);
  std::vector<struct ncclComm*> seen;
  ScopedHook hook(g_ncclCeInit, [&](struct ncclComm* c) {
    seen.push_back(c);
    return ncclSuccess;
  });

  EXPECT_EQ(ncclSuccess, ncclCommGroupRegisterSymmetric(&job_.base));

  EXPECT_EQ(2, hook.calls);
  EXPECT_EQ(std::vector<struct ncclComm*>({taskComm_.get(), secondTaskComm.get()}), seen)
    << "must be each task's own comm, not the job's, in FIFO order";
  EXPECT_TRUE(ncclIntruQueueEmpty(&comm_->ceInitTaskQueue));
}

TEST_F(GroupRegisterSymmetricCeInitTest, CeInitFailurePropagatesAndStopsDequeuing) {
  auto* failing = static_cast<struct ncclCeInitTask*>(calloc(1, sizeof(struct ncclCeInitTask)));
  auto* neverReached = static_cast<struct ncclCeInitTask*>(calloc(1, sizeof(struct ncclCeInitTask)));
  ncclIntruQueueEnqueue(&comm_->ceInitTaskQueue, failing);
  ncclIntruQueueEnqueue(&comm_->ceInitTaskQueue, neverReached);
  ScopedHook hook(g_ncclCeInit, [&](struct ncclComm*) { return ncclSystemError; });

  EXPECT_EQ(ncclSystemError, ncclCommGroupRegisterSymmetric(&job_.base));

  EXPECT_EQ(1, hook.calls);
  EXPECT_EQ(neverReached, ncclIntruQueueHead(&comm_->ceInitTaskQueue));
  // `failing` leaks (LSan-visible), a production bug: the fail: path dequeues it unfreed, like ArgsGlobalCheckTest's.
}

// groupCleanup: the fail-path unwind for a group that never launched. Walks groupCommHeadPtr per
// task type (reclaiming planner state and clearing queues along the way), then drains
// asyncJobsPtr itself. Every comm gets ncclCommSetAsyncError(error) unless it is blocking.
class GroupCleanupTest : public ::testing::Test {
 protected:
  // AsyncState (comm, state) pairs: see the namespace-scope alias's own comment for why.
  std::vector<std::unique_ptr<ncclComm>> comms_;
  std::vector<AsyncState> asyncStates_;
  int funcCalls_ = 0;
  int destructorCalls_ = 0;

  static GroupCleanupTest* s_active;

  void SetUp() override {
    ResetGroupThreadLocals();
    ResetCommFakes();
    s_active = this;
    g_commSetAsyncError = [this](struct ncclComm* comm, ncclResult_t state) {
      asyncStates_.push_back({comm, state});
      return ncclSuccess;
    };
  }

  void TearDown() override {
    ResetCommFakes();
    ResetGroupThreadLocals();
    for (auto& comm : comms_) {
      ncclMemoryStackDestruct(&comm->memPermanent);
      ncclMemoryStackDestruct(&comm->memScoped);
    }
    s_active = nullptr;
  }

  static ncclResult_t CountingFunc(struct ncclAsyncJob*) {
    s_active->funcCalls_++;
    return ncclSuccess;
  }
  static void CountingDestructor(void*) {
    s_active->destructorCalls_++;
  }

  // Builds a comm joined to ncclGroupCommHead[type] via the real production helper (handles the
  // groupNext sentinel and, for Collective/RawTask, the memScoped push planner init).
  ncclComm* MakeJoinedComm(int type, int blocking) {
    comms_.push_back(std::make_unique<ncclComm>());
    ncclComm* comm = comms_.back().get();
    comm->config.blocking = blocking;
    comm->groupNext[type] = reinterpret_cast<struct ncclComm*>(NCCL_COMM_GROUP_INVALID);
    ncclMemoryStackConstruct(&comm->memPermanent);
    ncclMemoryStackConstruct(&comm->memScoped);
    ncclMemoryPoolConstruct(&comm->memPool_ncclKernelPlan);
    ncclMemoryPoolConstruct(&comm->memPool_ncclProxyOp);
    ncclMemoryPoolConstruct(&comm->memPool_ncclRawTask);
    ncclIntruQueueConstruct(&comm->planner.collCleanupQueue);
    ncclIntruQueueConstruct(&comm->planner.planQueue);
    ncclIntruQueueConstruct(&comm->rawTaskQueue.genericQueue);
    ncclIntruQueueConstruct(&comm->rawTaskQueue.bcastQueue);
    ncclIntruQueueConstruct(&comm->mgmtTaskQueue);
    ncclGroupCommJoin(comm, type);
    return comm;
  }
};
GroupCleanupTest* GroupCleanupTest::s_active = nullptr;

TEST_F(GroupCleanupTest, CollectiveType_ReclaimsPlannerAndSetsAsyncErrorOnlyForNonBlocking) {
  const int type = ncclGroupTaskTypeCollective;
  ncclComm* blocking = MakeJoinedComm(type, /*blocking=*/1);
  ncclComm* nonBlocking = MakeJoinedComm(type, /*blocking=*/0);
  struct ncclKernelPlan* blockingPlan = MakePlan(blocking, /*persistent=*/false);
  ncclIntruQueueEnqueue(&blocking->planner.planQueue, blockingPlan);
  struct ncclKernelPlan* nonBlockingPlan = MakePlan(nonBlocking, /*persistent=*/false);
  ncclIntruQueueEnqueue(&nonBlocking->planner.planQueue, nonBlockingPlan);

  groupCleanup(ncclGroupCommHead, &ncclAsyncJobs, ncclSystemError);

  EXPECT_EQ(nullptr, ncclGroupCommHead[type]);
  EXPECT_EQ(reinterpret_cast<struct ncclComm*>(NCCL_COMM_GROUP_INVALID), blocking->groupNext[type]);
  EXPECT_EQ(reinterpret_cast<struct ncclComm*>(NCCL_COMM_GROUP_INVALID), nonBlocking->groupNext[type]);
  EXPECT_EQ(reinterpret_cast<struct ncclMemoryPool::Cell*>(blockingPlan), blocking->memPool_ncclKernelPlan.head)
    << "reclaimPlannerState must run for every comm regardless of blocking";
  EXPECT_EQ(reinterpret_cast<struct ncclMemoryPool::Cell*>(nonBlockingPlan), nonBlocking->memPool_ncclKernelPlan.head);
  EXPECT_EQ(std::vector<AsyncState>({{nonBlocking, ncclSystemError}}), asyncStates_)
    << "only the non-blocking comm, specifically, may receive ncclCommSetAsyncError";
}

TEST_F(GroupCleanupTest, RawTaskType_DrainsRawTaskQueueAndResetsClassifiedQueues) {
  const int type = ncclGroupTaskTypeRawTask;
  ncclComm* comm = MakeJoinedComm(type, /*blocking=*/0);
  // Two entries on genericQueue: with one, a drain loop and a single dequeue are indistinguishable.
  struct ncclRawTask* generic =
    ncclMemoryPoolAlloc<struct ncclRawTask>(&comm->memPool_ncclRawTask, &comm->memPermanent);
  ncclIntruQueueEnqueue(&comm->rawTaskQueue.genericQueue, generic);
  struct ncclRawTask* generic2 =
    ncclMemoryPoolAlloc<struct ncclRawTask>(&comm->memPool_ncclRawTask, &comm->memPermanent);
  ncclIntruQueueEnqueue(&comm->rawTaskQueue.genericQueue, generic2);
  struct ncclRawTask* bcast =
    ncclMemoryPoolAlloc<struct ncclRawTask>(&comm->memPool_ncclRawTask, &comm->memPermanent);
  ncclIntruQueueEnqueue(&comm->rawTaskQueue.bcastQueue, bcast);
  // All six share the same element type (ncclTaskTuningInfo). Populate every one so the "reset"
  // assertions below prove something drained, rather than asserting an already-empty precondition.
  ncclTaskTuningInfo symNode{}, legacyNode{}, allgathervNode{}, p2pNode{}, rmaNode{}, ceNode{};
  ncclIntruQueueEnqueue(&comm->classifiedTaskQueues.symTaskQueue, &symNode);
  ncclIntruQueueEnqueue(&comm->classifiedTaskQueues.legacyTaskQueue, &legacyNode);
  ncclIntruQueueEnqueue(&comm->classifiedTaskQueues.allgathervTaskQueue, &allgathervNode);
  ncclIntruQueueEnqueue(&comm->classifiedTaskQueues.p2pTaskQueue, &p2pNode);
  ncclIntruQueueEnqueue(&comm->classifiedTaskQueues.rmaTaskQueue, &rmaNode);
  ncclIntruQueueEnqueue(&comm->classifiedTaskQueues.ceTaskQueue, &ceNode);

  groupCleanup(ncclGroupCommHead, &ncclAsyncJobs, ncclSuccess);

  EXPECT_TRUE(ncclIntruQueueEmpty(&comm->rawTaskQueue.genericQueue));
  EXPECT_TRUE(ncclIntruQueueEmpty(&comm->rawTaskQueue.bcastQueue));
  // All three freed cells must come back, not just one: genericQueue drains first (FIFO: generic,
  // generic2) then bcastQueue, and ncclMemoryPoolFree pushes at the head (LIFO), so the free list
  // must read bcast, generic2, generic in that order.
  auto* poolHead = comm->memPool_ncclRawTask.head;
  ASSERT_NE(nullptr, poolHead);
  EXPECT_EQ(reinterpret_cast<struct ncclMemoryPool::Cell*>(bcast), poolHead);
  ASSERT_NE(nullptr, poolHead->next);
  EXPECT_EQ(reinterpret_cast<struct ncclMemoryPool::Cell*>(generic2), poolHead->next);
  ASSERT_NE(nullptr, poolHead->next->next);
  EXPECT_EQ(reinterpret_cast<struct ncclMemoryPool::Cell*>(generic), poolHead->next->next);
  EXPECT_TRUE(ncclIntruQueueEmpty(&comm->classifiedTaskQueues.symTaskQueue));
  EXPECT_TRUE(ncclIntruQueueEmpty(&comm->classifiedTaskQueues.legacyTaskQueue));
  EXPECT_TRUE(ncclIntruQueueEmpty(&comm->classifiedTaskQueues.allgathervTaskQueue));
  EXPECT_TRUE(ncclIntruQueueEmpty(&comm->classifiedTaskQueues.p2pTaskQueue));
  EXPECT_TRUE(ncclIntruQueueEmpty(&comm->classifiedTaskQueues.rmaTaskQueue));
  EXPECT_TRUE(ncclIntruQueueEmpty(&comm->classifiedTaskQueues.ceTaskQueue));
}

// Distinguishes this drain (destructor only) from ncclMgmtTaskJobFunc's own drain (func then
// destructor): a regression that made this branch also call func would pass MgmtTaskJobFuncTest
// but silently double-run production work here.
TEST_F(GroupCleanupTest, MgmtTaskType_DrainsQueueRunningDestructorOnlyNotFunc) {
  const int type = ncclGroupTaskTypeMgmtTask;
  ncclComm* comm = MakeJoinedComm(type, /*blocking=*/0);
  ncclAsyncJob task1{}, task2{};
  task1.func = task2.func = CountingFunc;
  task1.destructor = task2.destructor = CountingDestructor;
  ncclIntruQueueEnqueue(&comm->mgmtTaskQueue, &task1);
  ncclIntruQueueEnqueue(&comm->mgmtTaskQueue, &task2);

  groupCleanup(ncclGroupCommHead, &ncclAsyncJobs, ncclSuccess);

  EXPECT_EQ(0, funcCalls_) << "groupCleanup's mgmt-task branch must only destruct, never run func";
  EXPECT_EQ(2, destructorCalls_);
  EXPECT_TRUE(ncclIntruQueueEmpty(&comm->mgmtTaskQueue));
}

// The final "reset everything" pass over asyncJobsPtr itself, isolated from the per-type
// groupCommHeadPtr loop above via an all-null heads array and a local job queue.
TEST_F(GroupCleanupTest, AsyncJobsPtrDrain_DestructsAllAndSetsAsyncErrorOnlyForNonDestroyNonBlocking) {
  struct ncclComm* emptyHeads[ncclGroupTaskTypeNum] = {};
  auto commA = std::make_unique<ncclComm>();  // destroyFlag=0, blocking=1: no SetAsyncError
  commA->config.blocking = 1;
  auto commB = std::make_unique<ncclComm>();  // destroyFlag=0, blocking=0: the only one that qualifies
  commB->config.blocking = 0;
  auto commC = std::make_unique<ncclComm>();  // destroyFlag=1, blocking=0: still no SetAsyncError
  commC->config.blocking = 0;

  ncclIntruQueue<ncclAsyncJob, &ncclAsyncJob::next> jobs;
  ncclIntruQueueConstruct(&jobs);
  // jobD has no comm at all: documents intent for the `job->comm &&` guard, not just
  // destroyFlag/blocking. Not a reliable mutation-catch for that specific conjunct though:
  // ncclComm is ~3.86MB and config sits ~3.85MB in, so a null-pointer-plus-offset read of
  // job->comm->config.blocking lands far past any guard page and isn't guaranteed to fault or
  // to read as a consistent value; verified this by hand, don't rely on this line alone for that.
  ncclAsyncJob jobA{}, jobB{}, jobC{}, jobD{};
  jobA.comm = commA.get();
  jobA.destroyFlag = 0;
  jobB.comm = commB.get();
  jobB.destroyFlag = 0;
  jobC.comm = commC.get();
  jobC.destroyFlag = 1;
  jobD.comm = nullptr;
  jobD.destroyFlag = 0;
  jobA.destructor = jobB.destructor = jobC.destructor = jobD.destructor = CountingDestructor;
  ncclIntruQueueEnqueue(&jobs, &jobA);
  ncclIntruQueueEnqueue(&jobs, &jobB);
  ncclIntruQueueEnqueue(&jobs, &jobC);
  ncclIntruQueueEnqueue(&jobs, &jobD);

  groupCleanup(emptyHeads, &jobs, ncclInternalError);

  EXPECT_EQ(4, destructorCalls_) << "every queued job must be destructed regardless of destroyFlag";
  EXPECT_EQ(std::vector<AsyncState>({{commB.get(), ncclInternalError}}), asyncStates_)
    << "only commB (destroyFlag=0, non-blocking, has a comm) may receive ncclCommSetAsyncError";
  EXPECT_TRUE(ncclIntruQueueEmpty(&jobs));
}

// asyncJobLaunch: called directly (it is static, but reachable from this TU), rather than through
// the full ncclGroupEndInternal/ncclGroupJobAbort machinery, to control job->state/result/thread
// precisely. Covers the branches the rest of this file's indirect exercise of this function
// (through GroupJobAbortTest) never reaches: the job-failure branch, the unexpected-state branch,
// childAbortFlag forwarding, and the fail:goto. The thread-join-failure branch (a real
// std::thread::join() throwing) is not covered here: ncclThreadJoin wraps plain std::thread with
// no fake seam, and there is no safe way to make a just-created joinable thread's join() throw.
class AsyncJobLaunchDirectTest : public ::testing::Test {
 protected:
  // One job plus its own abort-flag storage. Flags precede job so ~ncclAsyncJob joins a live thread before they die.
  // Every job asyncJobLaunch touches needs real abortFlag/abortFlagDev: only childAbortFlag is null-checked.
  struct FlaggedJob {
    uint32_t abortFlag = 0;
    uint32_t abortFlagDev = 0;
    uint32_t childAbortFlag = 0;
    uint32_t childAbortFlagDev = 0;
    ncclAsyncJob job{};

    explicit FlaggedJob(ncclResult_t (*func)(struct ncclAsyncJob*)) {
      job.func = func;
      job.state = ncclGroupJobRunning;
      job.destroyFlag = 0;
      job.abortFlag = &abortFlag;
      job.abortFlagDev = &abortFlagDev;
    }
    FlaggedJob(const FlaggedJob&) = delete;
    FlaggedJob& operator=(const FlaggedJob&) = delete;
  };

  ncclIntruQueue<ncclAsyncJob, &ncclAsyncJob::next> jobs_;

  void SetUp() override {
    ResetGroupThreadLocals();
    ncclIntruQueueConstruct(&jobs_);
  }
  void TearDown() override { ResetGroupThreadLocals(); }
};

// A failing job and an unexpected-state job each set ret through a different branch; combining
// them in one test would race (whichever branch a background thread happens to reach first wins),
// so each gets its own deterministic test instead.
TEST_F(AsyncJobLaunchDirectTest, FailingJobPropagatesResultAsRet) {
  FlaggedJob failing(FakeInitJobFails);
  FlaggedJob other(FakeInitJobSucceeds);  // a second, normal job: forces the multi-job (real thread) branch
  ncclIntruQueueEnqueue(&jobs_, &failing.job);
  ncclIntruQueueEnqueue(&jobs_, &other.job);
  volatile bool groupAbortFlag = false;

  ncclResult_t ret = asyncJobLaunch(&jobs_, &groupAbortFlag);

  EXPECT_EQ(ncclSystemError, ret);
  EXPECT_EQ(ncclGroupJobJoined, failing.job.state);
  EXPECT_EQ(ncclGroupJobJoined, other.job.state);
  // The job-failure branch must trip errorJobAbortFlag, which forwards the abort back onto this
  // same job (and, transitively, every other non-destroyFlag job still in the queue). Without
  // this, ret/state above still look right but a failing init never actually unwinds its siblings.
  EXPECT_EQ(1u, failing.abortFlag);
  EXPECT_EQ(1u, failing.abortFlagDev);
}

TEST_F(AsyncJobLaunchDirectTest, UnexpectedStateJobTripsInternalErrorAndForwardsAbort) {
  FlaggedJob other(FakeInitJobSucceeds);  // a second, normal job: forces the multi-job (real thread) branch
  FlaggedJob spinning(SpinUntilAbortFlagObserved);
  // An invalid sentinel: neither Running, Done, nor Joined. The real background thread (below)
  // has not yet observed abortFlag, so this state is genuinely still stale at the first poll.
  spinning.job.state = static_cast<ncclGroupJobState_t>(3);
  spinning.job.childAbortFlag = &spinning.childAbortFlag;
  spinning.job.childAbortFlagDev = &spinning.childAbortFlagDev;
  ncclIntruQueueEnqueue(&jobs_, &other.job);
  ncclIntruQueueEnqueue(&jobs_, &spinning.job);
  volatile bool groupAbortFlag = false;

  ncclResult_t ret = asyncJobLaunch(&jobs_, &groupAbortFlag);

  EXPECT_EQ(ncclInternalError, ret);
  EXPECT_EQ(ncclGroupJobJoined, other.job.state);
  // spinning.job.state is deliberately not asserted here: the unexpected-state branch does not clear
  // jobsDone for that job, so asyncJobLaunch can return in the same pass it aborts spinning,
  // before its real background thread has had a chance to observe abortFlag and finish (that
  // thread does still get safely joined moments later, by ~ncclAsyncJob's own destructor).
  EXPECT_EQ(1u, spinning.abortFlag) << "the unexpected-state branch forwards the shared abort";
  EXPECT_EQ(1u, spinning.abortFlagDev);
  EXPECT_EQ(1u, spinning.childAbortFlag) << "childAbortFlag must also be forwarded when non-null";
  EXPECT_EQ(1u, spinning.childAbortFlagDev);
}

TEST_F(AsyncJobLaunchDirectTest, GroupAbortFlagAloneForwardsToAJobWithNoFailure) {
  FlaggedJob spinning(SpinUntilAbortFlagObserved);
  spinning.job.childAbortFlag = nullptr;  // null: the childAbortFlag branch must not fire for this job
  // A second job forces the multi-job (real background thread) branch: with only one job queued,
  // asyncJobLaunch takes its single-job fast path and runs func synchronously on this thread,
  // never reaching the abort-forwarding block at all, which would leave spinning spinning forever.
  FlaggedJob other(FakeInitJobSucceeds);
  ncclIntruQueueEnqueue(&jobs_, &spinning.job);
  ncclIntruQueueEnqueue(&jobs_, &other.job);
  volatile bool groupAbortFlag = true;  // set from the start, not derived from a job failure

  ncclResult_t ret = asyncJobLaunch(&jobs_, &groupAbortFlag);

  EXPECT_EQ(ncclSuccess, ret) << "no job failed, so a group-level abort alone must not poison ret";
  EXPECT_EQ(ncclGroupJobJoined, spinning.job.state);
  EXPECT_EQ(1u, spinning.abortFlag);
  EXPECT_EQ(1u, spinning.abortFlagDev);
}

TEST_F(AsyncJobLaunchDirectTest, DestroyFlagJobIsNeverToldToAbort) {
  FlaggedJob failing(FakeInitJobFails);
  FlaggedJob destroying(FakeInitJobSucceeds);
  destroying.job.destroyFlag = 1;  // must never be told to abort, even once errorJobAbortFlag trips
  ncclIntruQueueEnqueue(&jobs_, &failing.job);
  ncclIntruQueueEnqueue(&jobs_, &destroying.job);
  volatile bool groupAbortFlag = false;

  ncclResult_t ret = asyncJobLaunch(&jobs_, &groupAbortFlag);

  EXPECT_EQ(ncclSystemError, ret);
  // Positive control: the abort-forwarding mechanism itself is alive (failing gets it), so
  // destroying's 0 below proves destroyFlag excludes it specifically, not that forwarding never
  // fired at all.
  EXPECT_EQ(1u, failing.abortFlag);
  EXPECT_EQ(0u, destroying.abortFlag) << "destroyFlag must gate out the abort-forwarding branch entirely";
  EXPECT_EQ(0u, destroying.abortFlagDev);
}

// ncclCommGroupRegisterSymmetric runs once per comm on its own thread, so the collective CE staging setup lives here.
class GroupRegisterSymmetricTest : public ::testing::Test {
 protected:
  std::unique_ptr<ncclComm> comm_;
  struct ncclGroupSymmetricJob job_{};

  void SetUp() override {
    ResetCeFakes();
    ResetGroupThreadLocals();
    comm_ = std::make_unique<ncclComm>();
    job_.comm = comm_.get();
  }
  void TearDown() override {
    while (!ncclIntruQueueEmpty(&comm_->ceInitTaskQueue)) {
      free(ncclIntruQueueDequeue(&comm_->ceInitTaskQueue));
    }
    ResetGroupThreadLocals();
    ResetCeFakes();
  }
};

TEST_F(GroupRegisterSymmetricTest, StagingPending_SetsUpOnceAndClearsTheFlag) {
  comm_->ceColl.stagingPending = true;
  ScopedHook staging(g_ncclCeEnsureAllReduceStaging, [&](struct ncclComm* c) {
    EXPECT_EQ(comm_.get(), c);
    return ncclSuccess;
  });

  EXPECT_EQ(ncclSuccess, ncclCommGroupRegisterSymmetric(&job_.base));

  EXPECT_EQ(1, staging.calls);
  EXPECT_FALSE(comm_->ceColl.stagingPending);
}

TEST_F(GroupRegisterSymmetricTest, NothingPending_SkipsStagingSetup) {
  ScopedHook staging(g_ncclCeEnsureAllReduceStaging, [](struct ncclComm*) { return ncclSuccess; });

  EXPECT_EQ(ncclSuccess, ncclCommGroupRegisterSymmetric(&job_.base));

  EXPECT_EQ(0, staging.calls);
}

TEST_F(GroupRegisterSymmetricTest, StagingSetupFails_PropagatesAndStillClearsTheFlag) {
  comm_->ceColl.stagingPending = true;
  ScopedHook staging(g_ncclCeEnsureAllReduceStaging, [](struct ncclComm*) { return ncclSystemError; });

  EXPECT_EQ(ncclSystemError, ncclCommGroupRegisterSymmetric(&job_.base));

  EXPECT_EQ(1, staging.calls);
  EXPECT_FALSE(comm_->ceColl.stagingPending);
}

TEST_F(GroupRegisterSymmetricTest, CeInitQueued_RunsBeforeStagingSetup) {
  comm_->ceColl.stagingPending = true;
  auto* ceTask = static_cast<struct ncclCeInitTask*>(calloc(1, sizeof(struct ncclCeInitTask)));
  ceTask->comm = comm_.get();
  ncclIntruQueueEnqueue(&comm_->ceInitTaskQueue, ceTask);
  std::vector<std::string> order;
  ScopedHook init(g_ncclCeInit, [&](struct ncclComm*) {
    order.push_back("init");
    return ncclSuccess;
  });
  ScopedHook staging(g_ncclCeEnsureAllReduceStaging, [&](struct ncclComm*) {
    order.push_back("staging");
    return ncclSuccess;
  });

  EXPECT_EQ(ncclSuccess, ncclCommGroupRegisterSymmetric(&job_.base));

  EXPECT_EQ(std::vector<std::string>({"init", "staging"}), order);
}

// A group that fails before its SymRegister job runs must not leave the flag for an unrelated later group.
TEST_F(GroupRegisterSymmetricTest, GroupCleanup_ClearsPendingStagingOfSymRegisterComms) {
  comm_->config.blocking = 1;
  for (int type = 0; type < ncclGroupTaskTypeNum; ++type) {
    comm_->groupNext[type] = reinterpret_cast<struct ncclComm*>(NCCL_COMM_GROUP_INVALID);
  }
  comm_->ceColl.stagingPending = true;
  ncclGroupCommJoin(comm_.get(), ncclGroupTaskTypeSymRegister);
  ASSERT_EQ(comm_.get(), ncclGroupCommHead[ncclGroupTaskTypeSymRegister]);

  groupCleanup(ncclGroupCommHead, &ncclAsyncJobs, ncclSystemError);

  EXPECT_FALSE(comm_->ceColl.stagingPending);
  EXPECT_EQ(nullptr, ncclGroupCommHead[ncclGroupTaskTypeSymRegister]);
}

}  // namespace
