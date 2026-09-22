/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Controllable seams for the IB verbs wrappers defined by src/misc/ibvwrap.cc.
// Only the protection-domain and memory-registration entry points are faked;
// add more here as further units come under test.
//
// Defaults are fail-loud for acquisitions (alloc_pd fails with *ret = NULL, the
// reg_mr entry points return NULL, matching the real wrappers on failure) and
// succeed for releases (dealloc_pd / dereg_mr), so an un-hooked call path can
// never report a resource it did not actually get.

#ifndef RCCL_TEST_HOST_IBVWRAP_FAKES_H_
#define RCCL_TEST_HOST_IBVWRAP_FAKES_H_

#include <cstddef>
#include <cstdint>
#include <functional>

#include "nccl.h"

struct ibv_context;
struct ibv_pd;
struct ibv_mr;

extern std::function<ncclResult_t(struct ibv_pd** /*ret*/, struct ibv_context* /*context*/)> g_ibvAllocPd;
extern std::function<ncclResult_t(struct ibv_pd* /*pd*/)> g_ibvDeallocPd;
extern std::function<struct ibv_mr*(struct ibv_pd* /*pd*/, void* /*addr*/, size_t /*length*/, int /*access*/)>
    g_ibvDirectRegMr;
extern std::function<struct ibv_mr*(struct ibv_pd* /*pd*/, void* /*addr*/, size_t /*length*/, uint64_t /*iova*/,
                                    int /*access*/)>
    g_ibvDirectRegMrIova2;
extern std::function<ncclResult_t(struct ibv_mr* /*mr*/)> g_ibvDeregMr;

void ResetIbvwrapFakes();

#endif  // RCCL_TEST_HOST_IBVWRAP_FAKES_H_
