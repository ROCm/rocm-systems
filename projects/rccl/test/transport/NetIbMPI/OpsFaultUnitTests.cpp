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

// =============================================================================
// Test: ShimPollCqRewriteAndForward
//
// Drives the poll_cq shim's rewrite path (status overwrite under a finite
// budget, then forwarding), plus the real-returns-<0 passthrough and the
// ANY-key lookup fallback (findFault second branch).
//
// Verifies: a matching completion's status is rewritten until the budget is
//           spent, negative real returns pass through, and the ANY key matches a
//           qp_num with no exact entry.
// Requires: installed fake context.
// =============================================================================
TEST(OpsFaultUnit, ShimPollCqRewriteAndForward) {
    resetCounters();
    FakeCtx fake(/*qpNum=*/202);
    ASSERT_EQ(ncclIbOpsFaultInstall(&fake.ctx), ncclSuccess);

    struct ibv_wc wc[4];

    // No fault armed: real WCs pass through unmodified.
    g_pollReturn = 2; g_pollWcQpNum = 202;
    EXPECT_EQ(fake.ctx.ops.poll_cq(&fake.cq, 4, wc), 2);
    EXPECT_EQ(wc[0].status, IBV_WC_SUCCESS);

    // Rewrite mode, finite budget of 1, matching qp_num.
    ASSERT_EQ(ncclIbOpsFaultArmPollCq(&fake.ctx, 202, kWcRemAccess, /*count=*/1, /*idle=*/false),
              ncclSuccess);
    g_pollReturn = 2;
    EXPECT_EQ(fake.ctx.ops.poll_cq(&fake.cq, 4, wc), 2);
    // First WC rewritten; budget now spent.
    EXPECT_EQ((int)wc[0].status, kWcRemAccess);

    // Budget exhausted: next poll leaves status untouched.
    g_pollReturn = 1;
    EXPECT_EQ(fake.ctx.ops.poll_cq(&fake.cq, 4, wc), 1);
    EXPECT_EQ(wc[0].status, IBV_WC_SUCCESS);

    // Real poll returns negative: shim passes the error through.
    g_pollReturn = -1;
    EXPECT_LT(fake.ctx.ops.poll_cq(&fake.cq, 4, wc), 0);

    EXPECT_EQ(ncclIbOpsFaultRemove(&fake.ctx), ncclSuccess);

    // ANY-key rewrite: arm the context-wide key, then deliver a real completion
    // whose qp_num has no exact entry. The lookup must fall back to the ANY entry
    // (findFault second branch) and rewrite the WC.
    resetCounters();
    FakeCtx fake2(/*qpNum=*/777);
    ASSERT_EQ(ncclIbOpsFaultInstall(&fake2.ctx), ncclSuccess);
    ASSERT_EQ(ncclIbOpsFaultArmPollCq(&fake2.ctx, NCCL_IB_OPS_FAULT_QP_ANY, kWcRemAccess,
                                      /*count=*/-1, /*idle=*/false), ncclSuccess);
    g_pollReturn = 1; g_pollWcQpNum = 999;  // qp_num not individually armed
    EXPECT_EQ(fake2.ctx.ops.poll_cq(&fake2.cq, 4, wc), 1);
    EXPECT_EQ((int)wc[0].status, kWcRemAccess);  // matched via ANY fallback
    EXPECT_EQ(ncclIbOpsFaultRemove(&fake2.ctx), ncclSuccess);
}

// =============================================================================
// Test: ShimPollCqSynthesizeIdle
//
// When the real poll_cq returns 0 and an armed QP requested idle injection, the
// shim fabricates one error WC. Covers the specific-qp_num key, the context-wide
// ANY key (qp_num selector branch), and finite/unlimited budget behavior.
//
// Verifies: an idle poll synthesizes a WC for the armed QP; ANY uses qp_num 0;
//           unlimited keeps firing; finite fires once then stops.
// Requires: installed fake context with idle injection armed.
// =============================================================================
TEST(OpsFaultUnit, ShimPollCqSynthesizeIdle) {
    resetCounters();
    struct ibv_wc wc[2];

    // (a) Specific qp_num: synthesized WC carries that qp_num.
    {
        FakeCtx fake(/*qpNum=*/203);
        ASSERT_EQ(ncclIbOpsFaultInstall(&fake.ctx), ncclSuccess);
        ASSERT_EQ(ncclIbOpsFaultArmPollCq(&fake.ctx, 203, kWcRemAccess, /*count=*/1, /*idle=*/true),
                  ncclSuccess);
        g_pollReturn = 0;  // idle
        EXPECT_EQ(fake.ctx.ops.poll_cq(&fake.cq, 2, wc), 1);
        EXPECT_EQ((int)wc[0].status, kWcRemAccess);
        EXPECT_EQ(wc[0].qp_num, 203u);
        EXPECT_EQ(ncclIbOpsFaultRemove(&fake.ctx), ncclSuccess);
    }

    // (b) ANY key: synthesized WC uses qp_num 0 (ANY -> 0 selector branch).
    {
        FakeCtx fake(/*qpNum=*/204);
        ASSERT_EQ(ncclIbOpsFaultInstall(&fake.ctx), ncclSuccess);
        ASSERT_EQ(ncclIbOpsFaultArmPollCq(&fake.ctx, NCCL_IB_OPS_FAULT_QP_ANY, kWcFlushErr,
                                          /*count=*/1, /*idle=*/true), ncclSuccess);
        g_pollReturn = 0;
        EXPECT_EQ(fake.ctx.ops.poll_cq(&fake.cq, 2, wc), 1);
        EXPECT_EQ((int)wc[0].status, kWcFlushErr);
        EXPECT_EQ(wc[0].qp_num, 0u);
        EXPECT_EQ(ncclIbOpsFaultRemove(&fake.ctx), ncclSuccess);
    }

    // (c) Unlimited budget: idle synthesize keeps firing every poll (the
    //     pollInjectCount>0 decrement is skipped for count<0).
    {
        FakeCtx fake(/*qpNum=*/205);
        ASSERT_EQ(ncclIbOpsFaultInstall(&fake.ctx), ncclSuccess);
        ASSERT_EQ(ncclIbOpsFaultArmPollCq(&fake.ctx, 205, kWcRemAccess, /*count=*/-1, /*idle=*/true),
                  ncclSuccess);
        g_pollReturn = 0;
        EXPECT_EQ(fake.ctx.ops.poll_cq(&fake.cq, 2, wc), 1);
        EXPECT_EQ((int)wc[0].status, kWcRemAccess);
        // Fires again - unlimited budget is not consumed.
        EXPECT_EQ(fake.ctx.ops.poll_cq(&fake.cq, 2, wc), 1);
        EXPECT_EQ((int)wc[0].status, kWcRemAccess);
        EXPECT_EQ(ncclIbOpsFaultRemove(&fake.ctx), ncclSuccess);
    }

    // (d) Finite budget exhausts: synthesize fires once, then the spent entry
    //     (pollInjectCount==0) is skipped and the idle poll returns 0.
    {
        FakeCtx fake(/*qpNum=*/206);
        ASSERT_EQ(ncclIbOpsFaultInstall(&fake.ctx), ncclSuccess);
        ASSERT_EQ(ncclIbOpsFaultArmPollCq(&fake.ctx, 206, kWcRemAccess, /*count=*/1, /*idle=*/true),
                  ncclSuccess);
        g_pollReturn = 0;
        EXPECT_EQ(fake.ctx.ops.poll_cq(&fake.cq, 2, wc), 1);  // fires
        EXPECT_EQ(fake.ctx.ops.poll_cq(&fake.cq, 2, wc), 0);  // budget spent -> no synth
        EXPECT_EQ(ncclIbOpsFaultRemove(&fake.ctx), ncclSuccess);
    }
}

// =============================================================================
// Test: ShimLostRealOpDefensive
//
// Defensive guard: if a shim is ever invoked after its registry entry is gone
// (the saved real op is lost), it must fail loudly rather than dereference a
// null pointer. This path cannot occur through the public API on healthy
// hardware; the test documents and locks the guard behavior.
//
// Verifies: a captured shim called after Remove returns EFAULT (post_send/recv)
//           or a negative value (poll_cq) instead of crashing.
// Requires: installed fake context; shim pointers captured before Remove.
// =============================================================================
TEST(OpsFaultUnit, ShimLostRealOpDefensive) {
    resetCounters();
    FakeCtx fake(/*qpNum=*/205);
    ASSERT_EQ(ncclIbOpsFaultInstall(&fake.ctx), ncclSuccess);

    // Capture the installed shim pointers, then remove the registry entry.
    auto shimSend = fake.ctx.ops.post_send;
    auto shimRecv = fake.ctx.ops.post_recv;
    auto shimPoll = fake.ctx.ops.poll_cq;
    ASSERT_EQ(ncclIbOpsFaultRemove(&fake.ctx), ncclSuccess);

    struct ibv_send_wr swr; memset(&swr, 0, sizeof(swr));
    struct ibv_recv_wr rwr; memset(&rwr, 0, sizeof(rwr));
    struct ibv_wc wc[1];
    struct ibv_send_wr* sbad = nullptr;
    struct ibv_recv_wr* rbad = nullptr;

    EXPECT_EQ(shimSend(&fake.qp, &swr, &sbad), EFAULT);
    EXPECT_EQ(shimRecv(&fake.qp, &rwr, &rbad), EFAULT);
    EXPECT_LT(shimPoll(&fake.cq, 1, wc), 0);
}

#endif /* MPI_TESTS_ENABLED && ENABLE_FAULT_INJECTION */
