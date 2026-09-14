/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Window-shape queries defined by src/dev_runtime.cc. Default to "not
// registered" so the NVLS/CollNet registration arms stay off unless a test asks.

#ifndef RCCL_TEST_HOST_DEV_RUNTIME_FAKES_H_
#define RCCL_TEST_HOST_DEV_RUNTIME_FAKES_H_

#include <functional>

#include "nccl.h"

struct ncclComm;
struct ncclDevrWindow;

extern bool g_devrWindowIsMultiSegment;  // UNDRIVEN
extern bool g_devrWindowHasSysmemSegment;  // UNDRIVEN

// dev_runtime.h entry points rma_ce.cc reaches while setting up CE state.
// ncclDevrGetLsaRankPtr is the one tests drive: it is how rma_ce.cc resolves a
// peer's window address, so a test controls what address comes back.
extern std::function<ncclResult_t(struct ncclComm*)> g_devrInitOnce;
extern std::function<ncclResult_t(struct ncclComm*, int /*peerWorldRank*/,
                                  int* /*peerLsaRank*/)>
    g_devrWorldToLsaRank;
extern std::function<ncclResult_t(struct ncclComm*, struct ncclDevrWindow*, size_t /*offset*/,
                                  int /*lsaRank*/, void** /*outPtr*/)>
    g_devrGetLsaRankPtr;

void ResetDevRuntimeFakes();

#endif  // RCCL_TEST_HOST_DEV_RUNTIME_FAKES_H_
