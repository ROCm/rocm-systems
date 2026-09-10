/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Controllable seams for the non-static externals that the #included RMA units
// reach but do not define themselves. Two units use this file:
//
//   rma_proxy_progress.cc  (rma-proxy-progress-test.cc)
//     - ncclRmaProxyCircularBufEmpty(ctx, peer)   from rma_proxy_launch.cc
//     - ncclRmaProxyDestroyDesc(comm, &desc)      from rma_proxy_launch.cc
//   rma.cc                 (rma-test.cc)
//     - ncclRmaProxyPutLaunch / ncclRmaProxyWaitLaunch   from rma_proxy_launch.cc
//     - ncclRmaCePutLaunch   / ncclRmaCeWaitLaunch       from rma_ce.cc
//
// The rma_ce.cc pair lives here rather than in a separate ce fakes file because
// rma.cc dispatches to both pairs from the same branch, and a test asserting the
// dispatch needs to install all four together.
//
// Everything else those TUs touch is either file-static (reached via the
// #include), header-inline (ncclIntruQueue*, COMPILER_ATOMIC_* macros), or
// already covered by nccl_fakes.cc's no-op ncclDebugLog. The network itself is
// not faked here -- it is a plain ncclRma_t function-pointer vtable that the
// test populates directly (see FakeNet in rma-proxy-progress-test.cc).
//
// Tests install per-test behaviour by overwriting a hook in a fixture's SetUp()
// and ResetRmaFakes() (called from TearDown()) restores the defaults so tests
// don't contaminate each other.

#ifndef RCCL_TEST_HOST_RMA_FAKES_H_
#define RCCL_TEST_HOST_RMA_FAKES_H_

#include <functional>

#include "nccl.h"

struct ncclComm;
struct ncclKernelPlan;
struct ncclRmaProxyCtx;
struct ncclRmaProxyDesc;

// ncclRmaProxyCircularBufEmpty: default mirrors the production predicate
// (empty when consumer index has caught up to producer index, ci >= pi), so
// tests drive the pending scan simply by setting ctx->pis[peer]/ctx->cis[peer].
extern std::function<bool(struct ncclRmaProxyCtx* ctx, int peer)>
    g_rmaCircularBufEmpty;

// ncclRmaProxyDestroyDesc: default nulls the caller's slot (as production does)
// and returns ncclSuccess. Tests that want to observe destruction install a
// hook that records the destroyed descriptor.
extern std::function<ncclResult_t(struct ncclComm* comm,
                                  struct ncclRmaProxyDesc** desc)>
    g_rmaDestroyDesc;

// Launch seams for rma.cc's dispatch targets (real ones in rma_proxy_launch.cc
// / rma_ce.cc, which drag in the GPU stack). Default to ncclSuccess; tests hook
// them to record the stream, or to return an error and drive a NCCLCHECKGOTO.
extern std::function<ncclResult_t(struct ncclComm* comm, struct ncclKernelPlan* plan,
                                  hipStream_t stream)>
    g_rmaProxyWaitLaunch;

extern std::function<ncclResult_t(struct ncclComm* comm, struct ncclKernelPlan* plan,
                                  hipStream_t stream)>
    g_rmaCeWaitLaunch;

// Restore every hook in this file to its default. Call from fixture TearDown().
void ResetRmaFakes();

#endif  // RCCL_TEST_HOST_RMA_FAKES_H_
