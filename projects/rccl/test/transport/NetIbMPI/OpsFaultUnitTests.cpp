/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>
#include <cstring>
#include <cerrno>

#if defined(MPI_TESTS_ENABLED) && defined(ENABLE_FAULT_INJECTION)
#include "net_ib_ops_fault.h"  /* inner extern "C" ncclIbOpsFault* API + verbs types */

/*
 * Pure-unit harness for the ops-overload fault mechanism
 * (src/transport/net_ib_cast/net_ib_ops_fault.cc).
 *
 * The functional MPI tests (FaultInjCastOps* in FaultInjectTests.cpp) drive the
 * mechanism through the public ncclIbCastFaultOps* API, which always passes a
 * live, registered qp->context. That leaves the inner functions' defensive
 * guards (NULL ctx, unregistered ctx, NULL ops, lost real op) and several shim
 * sub-paths unreachable.
 *
 * These tests call the inner extern "C" ncclIbOpsFault*(ctx, ...) functions
 * directly with a fabricated ibv_context, so the guards and shim error paths are
 * exercised in isolation. They build into rccl-UnitTestsMPI and run under the
 * MPI launcher (main_mpi.cpp initializes MPI), but use no MPI communication, no
 * RDMA, and no hardware: plain CPU. They run on every rank harmlessly (the
 * registry is process-global; each rank uses its own stack context) and need no
 * validateTestPrerequisites gating.
 */

namespace {

// Fake real ops the shims forward to.
// Call counters + programmable poll behavior let tests assert forwarding and
// drive the poll_cq shim's real-return branches (<0, ==0, >0).
int  g_realPostSendCalls = 0;
int  g_realPostRecvCalls = 0;
int  g_realPollCqCalls   = 0;
int  g_pollReturn        = 0;   // value fakePollCq returns (>0 fills that many WCs)
uint32_t g_pollWcQpNum   = 0;   // qp_num stamped on fabricated real WCs

int fakeRealPostSend(struct ibv_qp*, struct ibv_send_wr*, struct ibv_send_wr**) {
  ++g_realPostSendCalls;
  return 0;
}
int fakeRealPostRecv(struct ibv_qp*, struct ibv_recv_wr*, struct ibv_recv_wr**) {
  ++g_realPostRecvCalls;
  return 0;
}
int fakeRealPollCq(struct ibv_cq*, int num_entries, struct ibv_wc* wc) {
  ++g_realPollCqCalls;
  if (g_pollReturn < 0) return g_pollReturn;
  int n = g_pollReturn < num_entries ? g_pollReturn : num_entries;
  for (int i = 0; i < n; i++) {
    memset(&wc[i], 0, sizeof(wc[i]));
    wc[i].status = IBV_WC_SUCCESS;
    wc[i].qp_num = g_pollWcQpNum;
  }
  return n;
}

// A self-contained fake verbs context with one qp + one cq wired to it.
struct FakeCtx {
  struct ibv_context ctx;
  struct ibv_qp      qp;
  struct ibv_cq      cq;

  explicit FakeCtx(uint32_t qpNum = 100, bool withOps = true) {
    memset(&ctx, 0, sizeof(ctx));
    memset(&qp, 0, sizeof(qp));
    memset(&cq, 0, sizeof(cq));
    if (withOps) {
      ctx.ops.post_send = fakeRealPostSend;
      ctx.ops.post_recv = fakeRealPostRecv;
      ctx.ops.poll_cq   = fakeRealPollCq;
    }
    qp.context = &ctx;
    qp.qp_num  = qpNum;
    cq.context = &ctx;
  }
};

void resetCounters() {
  g_realPostSendCalls = g_realPostRecvCalls = g_realPollCqCalls = 0;
  g_pollReturn = 0;
  g_pollWcQpNum = 0;
}

constexpr int kEagain       = EAGAIN;
constexpr int kWcFlushErr   = IBV_WC_WR_FLUSH_ERR;
constexpr int kWcRemAccess  = IBV_WC_REM_ACCESS_ERR;

}  // namespace

// =============================================================================
// Test: InstallRemoveLifecycle
//
// Install swaps the three ops to shims and returns ncclSuccess (success/INFO
// path). A second Install on the same context hits the "already installed"
// early return. Remove restores the original ops. Remove on a context that was
// never installed is a no-op success.
//
// Verifies: Install replaces ops and is idempotent; Remove restores ops and is
//           a no-op on an unregistered context.
// Requires: no setup (fake ibv_context on the stack).
// =============================================================================
TEST(OpsFaultUnit, InstallRemoveLifecycle) {
    resetCounters();
    FakeCtx fake(/*qpNum=*/100);
    auto realSend = fake.ctx.ops.post_send;
    auto realRecv = fake.ctx.ops.post_recv;
    auto realPoll = fake.ctx.ops.poll_cq;

    // First install: success path; ops are replaced by shims (no longer == real).
    EXPECT_EQ(ncclIbOpsFaultInstall(&fake.ctx), ncclSuccess);
    EXPECT_NE(fake.ctx.ops.post_send, realSend);
    EXPECT_NE(fake.ctx.ops.post_recv, realRecv);
    EXPECT_NE(fake.ctx.ops.poll_cq,   realPoll);
    auto shimSend = fake.ctx.ops.post_send;

    // Second install: already-registered early return; ops unchanged.
    EXPECT_EQ(ncclIbOpsFaultInstall(&fake.ctx), ncclSuccess);
    EXPECT_EQ(fake.ctx.ops.post_send, shimSend);

    // Remove restores the original ops.
    EXPECT_EQ(ncclIbOpsFaultRemove(&fake.ctx), ncclSuccess);
    EXPECT_EQ(fake.ctx.ops.post_send, realSend);
    EXPECT_EQ(fake.ctx.ops.post_recv, realRecv);
    EXPECT_EQ(fake.ctx.ops.poll_cq,   realPoll);

    // Remove again: not registered -> no-op success.
    EXPECT_EQ(ncclIbOpsFaultRemove(&fake.ctx), ncclSuccess);
}

// =============================================================================
// Test: InstallNullOpsSkipped
//
// Install must refuse to shim a context whose target ops are NULL (it would have
// nothing to forward to), exercising each arm of the "any target op is NULL"
// check independently.
//
// Verifies: a context with any NULL target op is left unmodified and not
//           registered (subsequent arm returns ncclInvalidArgument).
// Requires: fake contexts with selected ops nulled out.
// =============================================================================
TEST(OpsFaultUnit, InstallNullOpsSkipped) {
    resetCounters();
    FakeCtx fake(/*qpNum=*/101, /*withOps=*/false);  // all three ops NULL
    ASSERT_EQ(fake.ctx.ops.post_send, nullptr);

    EXPECT_EQ(ncclIbOpsFaultInstall(&fake.ctx), ncclSuccess);
    // Ops still NULL: install skipped, context not registered.
    EXPECT_EQ(fake.ctx.ops.post_send, nullptr);
    // Arming must now fail because the context was not registered.
    EXPECT_EQ(ncclIbOpsFaultArmPostSend(&fake.ctx, 0, kEagain), ncclInvalidArgument);

    // Cover each arm of the "any target op is NULL" check independently: a
    // context can have some ops set and others NULL. Each partial-NULL case must
    // also be skipped (not registered).
    {  // only post_recv NULL (post_send set -> first arm false, second true)
        FakeCtx fake2(/*qpNum=*/110);
        fake2.ctx.ops.post_recv = nullptr;
        EXPECT_EQ(ncclIbOpsFaultInstall(&fake2.ctx), ncclSuccess);
        EXPECT_EQ(ncclIbOpsFaultArmPostSend(&fake2.ctx, 0, kEagain), ncclInvalidArgument);
    }
    {  // only poll_cq NULL (first two arms false, third true)
        FakeCtx fake2(/*qpNum=*/111);
        fake2.ctx.ops.poll_cq = nullptr;
        EXPECT_EQ(ncclIbOpsFaultInstall(&fake2.ctx), ncclSuccess);
        EXPECT_EQ(ncclIbOpsFaultArmPostSend(&fake2.ctx, 0, kEagain), ncclInvalidArgument);
    }
}

// =============================================================================
// Test: NullCtxRejected
//
// Every inner API validates its context argument.
//
// Verifies: each entry point returns ncclInvalidArgument on a NULL ctx.
// Requires: no setup.
// =============================================================================
TEST(OpsFaultUnit, NullCtxRejected) {
    EXPECT_EQ(ncclIbOpsFaultInstall(nullptr),                         ncclInvalidArgument);
    EXPECT_EQ(ncclIbOpsFaultRemove(nullptr),                          ncclInvalidArgument);
    EXPECT_EQ(ncclIbOpsFaultArmPostSend(nullptr, 0, kEagain),         ncclInvalidArgument);
    EXPECT_EQ(ncclIbOpsFaultArmPostRecv(nullptr, 0, kEagain),         ncclInvalidArgument);
    EXPECT_EQ(ncclIbOpsFaultArmPollCq(nullptr, 0, kWcFlushErr, -1, false), ncclInvalidArgument);
    EXPECT_EQ(ncclIbOpsFaultClear(nullptr),                           ncclInvalidArgument);
}

// =============================================================================
// Test: ArmOnUnregisteredCtxRejected
//
// Exercises the registry-miss (find == end) guard on each entry for a context
// that was never installed.
//
// Verifies: Arm* return ncclInvalidArgument and Clear returns ncclSuccess (no-op)
//           on an unregistered context.
// Requires: a fake context with valid ops that is never installed.
// =============================================================================
TEST(OpsFaultUnit, ArmOnUnregisteredCtxRejected) {
    FakeCtx fake(/*qpNum=*/102);  // valid ops, but never Install'd

    EXPECT_EQ(ncclIbOpsFaultArmPostSend(&fake.ctx, 0, kEagain),          ncclInvalidArgument);
    EXPECT_EQ(ncclIbOpsFaultArmPostRecv(&fake.ctx, 0, kEagain),          ncclInvalidArgument);
    EXPECT_EQ(ncclIbOpsFaultArmPollCq(&fake.ctx, 0, kWcFlushErr, -1, false), ncclInvalidArgument);
    // Clear on an unregistered context is a benign no-op.
    EXPECT_EQ(ncclIbOpsFaultClear(&fake.ctx), ncclSuccess);
}

// =============================================================================
// Test: ShimPostSendErrnoAndForward
//
// Drives ctx->ops.post_send (the shim) directly after install.
//
// Verifies: when armed the shim returns the injected errno and sets *bad_wr
//           (tolerating bad_wr == NULL); when disarmed it forwards to the real op.
// Requires: installed fake context.
// =============================================================================
TEST(OpsFaultUnit, ShimPostSendErrnoAndForward) {
    resetCounters();
    FakeCtx fake(/*qpNum=*/200);
    ASSERT_EQ(ncclIbOpsFaultInstall(&fake.ctx), ncclSuccess);
    ASSERT_EQ(ncclIbOpsFaultArmPostSend(&fake.ctx, 200, kEagain), ncclSuccess);

    struct ibv_send_wr wr; memset(&wr, 0, sizeof(wr));
    struct ibv_send_wr* bad = nullptr;

    // Armed: returns errno, sets *bad_wr.
    EXPECT_EQ(fake.ctx.ops.post_send(&fake.qp, &wr, &bad), kEagain);
    EXPECT_EQ(bad, &wr);
    EXPECT_EQ(g_realPostSendCalls, 0);  // real op not reached

    // Armed with bad_wr == NULL must not crash.
    EXPECT_EQ(fake.ctx.ops.post_send(&fake.qp, &wr, nullptr), kEagain);

    // Disarm (errnoVal=0): forwards to the real op.
    ASSERT_EQ(ncclIbOpsFaultArmPostSend(&fake.ctx, 200, 0), ncclSuccess);
    EXPECT_EQ(fake.ctx.ops.post_send(&fake.qp, &wr, &bad), 0);
    EXPECT_EQ(g_realPostSendCalls, 1);

    EXPECT_EQ(ncclIbOpsFaultRemove(&fake.ctx), ncclSuccess);
}

// =============================================================================
// Test: ShimPostRecvErrnoAndForward
//
// Symmetric to ShimPostSendErrnoAndForward for the post_recv shim.
//
// Verifies: when armed the shim returns the injected errno and sets *bad_wr
//           (tolerating bad_wr == NULL); when disarmed it forwards to the real op.
// Requires: installed fake context.
// =============================================================================
TEST(OpsFaultUnit, ShimPostRecvErrnoAndForward) {
    resetCounters();
    FakeCtx fake(/*qpNum=*/201);
    ASSERT_EQ(ncclIbOpsFaultInstall(&fake.ctx), ncclSuccess);
    ASSERT_EQ(ncclIbOpsFaultArmPostRecv(&fake.ctx, 201, kEagain), ncclSuccess);

    struct ibv_recv_wr wr; memset(&wr, 0, sizeof(wr));
    struct ibv_recv_wr* bad = nullptr;

    EXPECT_EQ(fake.ctx.ops.post_recv(&fake.qp, &wr, &bad), kEagain);
    EXPECT_EQ(bad, &wr);
    EXPECT_EQ(g_realPostRecvCalls, 0);

    EXPECT_EQ(fake.ctx.ops.post_recv(&fake.qp, &wr, nullptr), kEagain);

    ASSERT_EQ(ncclIbOpsFaultArmPostRecv(&fake.ctx, 201, 0), ncclSuccess);
    EXPECT_EQ(fake.ctx.ops.post_recv(&fake.qp, &wr, &bad), 0);
    EXPECT_EQ(g_realPostRecvCalls, 1);

    EXPECT_EQ(ncclIbOpsFaultRemove(&fake.ctx), ncclSuccess);
}


#endif /* MPI_TESTS_ENABLED && ENABLE_FAULT_INJECTION */
