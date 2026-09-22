/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Implementation of the src/misc/ibvwrap.cc fakes. See ibvwrap_fakes.h.

#include "ibvwrap_fakes.h"

#include "ibvwrap.h"
#include "signature-drift.h"

static ncclResult_t DefaultIbvAllocPd(struct ibv_pd** ret, struct ibv_context*) {
  if (ret) *ret = nullptr;
  return ncclSystemError;
}
std::function<ncclResult_t(struct ibv_pd**, struct ibv_context*)> g_ibvAllocPd = DefaultIbvAllocPd;

static ncclResult_t DefaultIbvDeallocPd(struct ibv_pd*) { return ncclSuccess; }
std::function<ncclResult_t(struct ibv_pd*)> g_ibvDeallocPd = DefaultIbvDeallocPd;

static struct ibv_mr* DefaultIbvDirectRegMr(struct ibv_pd*, void*, size_t, int) { return nullptr; }
std::function<struct ibv_mr*(struct ibv_pd*, void*, size_t, int)> g_ibvDirectRegMr = DefaultIbvDirectRegMr;

static struct ibv_mr* DefaultIbvDirectRegMrIova2(struct ibv_pd*, void*, size_t, uint64_t, int) { return nullptr; }
std::function<struct ibv_mr*(struct ibv_pd*, void*, size_t, uint64_t, int)> g_ibvDirectRegMrIova2 =
    DefaultIbvDirectRegMrIova2;

static ncclResult_t DefaultIbvDeregMr(struct ibv_mr*) { return ncclSuccess; }
std::function<ncclResult_t(struct ibv_mr*)> g_ibvDeregMr = DefaultIbvDeregMr;

ASSERT_HOOK_MATCHES_PROD(g_ibvAllocPd, wrap_ibv_alloc_pd);
ASSERT_HOOK_MATCHES_PROD(g_ibvDeallocPd, wrap_ibv_dealloc_pd);
ASSERT_HOOK_MATCHES_PROD(g_ibvDirectRegMr, wrap_direct_ibv_reg_mr);
ASSERT_HOOK_MATCHES_PROD(g_ibvDirectRegMrIova2, wrap_direct_ibv_reg_mr_iova2);
ASSERT_HOOK_MATCHES_PROD(g_ibvDeregMr, wrap_ibv_dereg_mr);

ncclResult_t wrap_ibv_alloc_pd(struct ibv_pd** ret, struct ibv_context* context) {
  return g_ibvAllocPd(ret, context);
}
ncclResult_t wrap_ibv_dealloc_pd(struct ibv_pd* pd) { return g_ibvDeallocPd(pd); }
struct ibv_mr* wrap_direct_ibv_reg_mr(struct ibv_pd* pd, void* addr, size_t length, int access) {
  return g_ibvDirectRegMr(pd, addr, length, access);
}
struct ibv_mr* wrap_direct_ibv_reg_mr_iova2(struct ibv_pd* pd, void* addr, size_t length, uint64_t iova,
                                            int access) {
  return g_ibvDirectRegMrIova2(pd, addr, length, iova, access);
}
ncclResult_t wrap_ibv_dereg_mr(struct ibv_mr* mr) { return g_ibvDeregMr(mr); }

void ResetIbvwrapFakes() {
  g_ibvAllocPd = DefaultIbvAllocPd;
  g_ibvDeallocPd = DefaultIbvDeallocPd;
  g_ibvDirectRegMr = DefaultIbvDirectRegMr;
  g_ibvDirectRegMrIova2 = DefaultIbvDirectRegMrIova2;
  g_ibvDeregMr = DefaultIbvDeregMr;
}
