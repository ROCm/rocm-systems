/*************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
//
// Host-only microtests for ncclGroupEndInternal's blocking-vs-non-blocking
// path selection (src/group.cc).
//
// These pin the contract that a non-blocking group (config.blocking == 0) with
// pending async init jobs hands the work to a background thread and returns
// ncclInProgress immediately -- it must NOT run the jobs synchronously on the
// caller. That is exactly the guarantee ncclCommInitRankConfig(blocking=0)
// relies on so a caller can poll / time out / abort when a peer rank never
// shows up.
//
// The suite compiles the hipified src/group.cc directly (via GROUP_CC_PATH) so
// the file-static launch helpers are reachable, and satisfies every external
// group.cc references with the module-organised test doubles in fakes/
// (comm_fakes, collective_stubs, ...). No GPU, no librccl.so, no
// HIP runtime.

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <vector>

#include "nccl.h"
#include "comm.h"
#include "group.h"

#include "fakes/comm_fakes.h"    // controllable ncclCommSetAsyncError seam
#include "fakes/nccl_fakes.h"    // g_loadParam, used by param_redirect.h
#include "ScopedHook.h"          // RAII install/restore for controllable seams

// Route group.cc's NCCL_PARAM sites through g_loadParam instead of the real
// ncclLoadParam (see param_redirect.h); must precede the unit under test.
#include "fakes/param_redirect.h"

// Controllable thread-creation seam. group.cc creates the non-blocking init
// background thread via STDTHREADCREATE_GOTO; when the seam returns true the
// macro takes its failure branch (goto fail with ncclSystemError) without
// actually spawning a thread, so the fail: path can be exercised
// deterministically. Tests install the failing behaviour via ScopedHook so it
// is restored automatically. checks.h is included first (with its own guard)
// so the real STDTHREADCREATE_IMPL stays available; only the _GOTO wrapper is
// replaced, and the redefinition survives into the textual include of group.cc
// below.
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

// Pull in the unit under test so its file-static helpers (groupLaunch,
// asyncJobLaunch, groupLaunchNonBlocking, ...) are visible. Must come after the
// headers / fakes it depends on are in scope.
#include GROUP_CC_PATH

namespace {

// A job body that succeeds instantly, standing in for a real communicator-init
// job. The behaviour under test is which *thread* runs this (background vs
// caller), not what the job itself does.
ncclResult_t FakeInitJobSucceeds(struct ncclAsyncJob*) { return ncclSuccess; }

class GroupEndInternalTest : public ::testing::Test {
 protected:
  std::unique_ptr<ncclComm> comm_;
  std::unique_ptr<ncclAsyncJob> job_;

  // Records every ncclCommSetAsyncError state pushed to comm_.
  std::vector<ncclResult_t> asyncStates_;

  void SetUp() override {
    ResetCommFakes();

    // Reset the thread-local group state group.cc reads. ncclGroupEndInternal
    // runs on this thread, so the thread-locals it inspects are these.
    ncclGroupDepth = 0;
    ncclGroupError = ncclSuccess;
    for (int type = 0; type < ncclGroupTaskTypeNum; ++type) ncclGroupCommHead[type] = nullptr;
    ncclGroupCommPreconnectHead = nullptr;
    ncclGroupBlocking = -1;
    ncclIntruQueueConstruct(&ncclAsyncJobs);

    comm_ = std::make_unique<ncclComm>();  // value-initialised => zeroed
    comm_->groupJob = nullptr;

    g_commSetAsyncError = [this](struct ncclComm*, ncclResult_t state) {
      asyncStates_.push_back(state);
      return ncclSuccess;
    };
  }

  void TearDown() override {
    // If a non-blocking init spawned a background job, join it and release the
    // group job before the fixture (and its comm) go away. A comm that reached
    // the fail: path has had its groupJob nulled by group.cc, so this is skipped
    // for it (and must be -- that groupJob is already deleted).
    if (comm_ && comm_->groupJob) {
      ncclGroupJobComplete(comm_->groupJob);
      comm_->groupJob = nullptr;
    }
    ResetCommFakes();
  }

  // Queue a single pending async job owned by comm_ and enter a group whose
  // blocking mode is `blocking`, mirroring the state ncclCommInitRankConfig
  // leaves behind just before ncclGroupEnd.
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
};

// The fix: a non-blocking group carrying pending init work must return
// ncclInProgress (handing the work to a background thread) rather than running
// it on the caller. Before the fix, ncclGroupEndInternal tested the already-
// drained ncclAsyncJobs queue, mis-selected the blocking path, and ran the jobs
// synchronously -- which for a missing peer rank blocks forever.
TEST_F(GroupEndInternalTest, NonBlockingGroupWithPendingJob_ReturnsInProgress) {
  EnterGroupWithOnePendingJob(/*blocking=*/0);

  EXPECT_EQ(ncclInProgress, ncclGroupEndInternal());

  // The non-blocking path publishes ncclInProgress on the comm's async error;
  // that is the state a caller polls on until the background job completes.
  EXPECT_NE(asyncStates_.end(),
            std::find(asyncStates_.begin(), asyncStates_.end(), ncclInProgress));
}

// The non-blocking path also publishes a group job on the communicator, which
// is what lets the caller subsequently poll or abort it.
TEST_F(GroupEndInternalTest, NonBlockingGroupWithPendingJob_PublishesGroupJobOnComm) {
  EnterGroupWithOnePendingJob(/*blocking=*/0);

  (void)ncclGroupEndInternal();

  EXPECT_NE(nullptr, comm_->groupJob);
}

// If the background thread fails to launch, ncclGroupEndInternal takes the fail:
// path and destroys the group job. Every comm that adopted that job (an init
// comm is reachable only via the async-jobs list, never the groupCommHead
// chains) must have comm->groupJob cleared, otherwise a later ncclCommAbort ->
// ncclCommEnsureReady -> ncclGroupJobAbort dereferences freed memory.
TEST_F(GroupEndInternalTest, ThreadCreateFailure_LeavesNoDanglingGroupJob) {
  EnterGroupWithOnePendingJob(/*blocking=*/0);
  ScopedHook failThreadCreate(g_threadCreateShouldFail, [] { return true; });

  // The fail: path really ran (system error), and the comm's groupJob -- which
  // the non-blocking branch published before the thread create -- was cleared
  // rather than left dangling into the just-deleted group job.
  EXPECT_EQ(ncclSystemError, ncclGroupEndInternal());
  EXPECT_EQ(nullptr, comm_->groupJob);
  EXPECT_EQ(1, failThreadCreate.calls);
}

// Contrast: a blocking group runs its jobs on the caller and reports the final
// result (ncclSuccess here), leaving no group job to poll.
TEST_F(GroupEndInternalTest, BlockingGroupWithPendingJob_RunsSynchronouslyAndReturnsSuccess) {
  EnterGroupWithOnePendingJob(/*blocking=*/1);

  EXPECT_EQ(ncclSuccess, ncclGroupEndInternal());
  EXPECT_EQ(nullptr, comm_->groupJob);
}

}  // namespace
