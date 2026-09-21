/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Controllable seams for the two non-static externals that the #included
// rma_proxy_progress.cc reaches but does not define itself:
//
//   - ncclRmaProxyCircularBufEmpty(ctx, peer)  (defined in rma_proxy_launch.cc)
//   - ncclRmaProxyDestroyDesc(comm, &desc)     (defined in rma_proxy_launch.cc)
//
// Everything else the compiled TU touches is either file-static (reached via
// the #include), header-inline (ncclIntruQueue*, COMPILER_ATOMIC_* macros), or
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

// Restore every hook in this file to its default. Call from fixture TearDown().
void ResetRmaFakes();

#endif  // RCCL_TEST_HOST_RMA_FAKES_H_
