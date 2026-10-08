/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Seams bootstrap_stubs.cc OWNS, declared once so a signature change is a compile error rather than a link mismatch.

#ifndef RCCL_TEST_HOST_BOOTSTRAP_STUBS_H_
#define RCCL_TEST_HOST_BOOTSTRAP_STUBS_H_

#include <cstdint>
#include <functional>

#include "nccl.h"

struct ncclBootstrapHandle;
struct ncclComm;

// src/bootstrap.cc. Fail-loud by default: only a test that scripts one may reach it.
extern std::function<ncclResult_t(int /*nHandles*/, void* /*handle*/, struct ncclComm* /*comm*/,
                                  struct ncclComm* /*parent*/)>
    g_bootstrapInit;

extern std::function<ncclResult_t(uint64_t /*commHash*/, struct ncclComm* /*comm*/, struct ncclComm* /*parent*/,
                                  int /*color*/, int /*key*/, int* /*parentRanks*/)>
    g_bootstrapSplit;

extern std::function<ncclResult_t(struct ncclBootstrapHandle* /*handle*/, bool /*idFromEnv*/)> g_bootstrapCreateRoot;

extern bool g_bootstrapNetInitFail;

// A std::function, not a result code: tests must write the allgathered (color, key) table into allData.
extern std::function<ncclResult_t(void* /*commState*/, void* /*allData*/, int /*size*/)> g_bootstrapAllGather;

extern ncclResult_t g_bootstrapGetUniqueIdResult;
extern int g_bootstrapGetUniqueIdCalls;
extern uint64_t g_bootstrapHandleMagic;
// Whole-handle payload the bootstrapGetUniqueId fake writes on success; magic is then overwritten from the global.
extern struct ncclBootstrapHandle g_bootstrapHandleTemplate;
// Declared separately: consumers cannot see ncclBootstrapHandle's complete type, which resetting needs.
void ResetBootstrapHandleTemplate();

// A std::function on top of the result code: the grow path validates the magic the coordinator broadcast back.
extern ncclResult_t g_bcastGrowHandleResult;
extern int g_bcastGrowHandleCalls;
extern bool g_bcastGrowHandleIsRoot;
ncclResult_t DefaultBcastGrowHandle(struct ncclBootstrapHandle* handle, struct ncclComm* parent, bool isRoot);
extern std::function<ncclResult_t(struct ncclBootstrapHandle*, struct ncclComm*, bool)> g_bcastGrowHandle;

void ResetBootstrapStubs();

#endif  // RCCL_TEST_HOST_BOOTSTRAP_STUBS_H_
