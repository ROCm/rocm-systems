/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "nccl.h"
#include "comm.h"        // NCCL_GIN_MAX_CONNECTIONS (via transitive gin headers)
#include "rma/rma_proxy.h"

#include "rma_fakes.h"

// ---------------------------------------------------------------------------
// Hook defaults
// ---------------------------------------------------------------------------

static bool DefaultRmaCircularBufEmpty(struct ncclRmaProxyCtx* ctx, int peer) {
  // Empty when the consumer index has caught up to the producer index. Use
  // exact equality rather than production's `>=` so the fake does not bake in
  // production's latent wraparound bug: once pi wraps past 2^32 while ci has
  // not, `ci >= pi` would spuriously report empty with descriptors still live.
  // (The production fix belongs in rma_proxy_launch.cc: `(pi - ci) == 0`.)
  return ctx->cis[peer] == ctx->pis[peer];
}

static ncclResult_t DefaultRmaDestroyDesc(struct ncclComm* /*comm*/,
                                          struct ncclRmaProxyDesc** desc) {
  *desc = nullptr;
  return ncclSuccess;
}

std::function<bool(struct ncclRmaProxyCtx* ctx, int peer)>
    g_rmaCircularBufEmpty = DefaultRmaCircularBufEmpty;

std::function<ncclResult_t(struct ncclComm* comm, struct ncclRmaProxyDesc** desc)>
    g_rmaDestroyDesc = DefaultRmaDestroyDesc;

// ---------------------------------------------------------------------------
// Externals the compiled TU links against
// ---------------------------------------------------------------------------

bool ncclRmaProxyCircularBufEmpty(struct ncclRmaProxyCtx* ctx, int peer) {
  return g_rmaCircularBufEmpty(ctx, peer);
}

ncclResult_t ncclRmaProxyDestroyDesc(struct ncclComm* comm, struct ncclRmaProxyDesc** desc) {
  return g_rmaDestroyDesc(comm, desc);
}

// ---------------------------------------------------------------------------
// Reset
// ---------------------------------------------------------------------------

void ResetRmaFakes() {
  g_rmaCircularBufEmpty = DefaultRmaCircularBufEmpty;
  g_rmaDestroyDesc      = DefaultRmaDestroyDesc;
}
