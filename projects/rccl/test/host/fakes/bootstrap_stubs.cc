/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Fail-loud stub floor for the bootstrap subsystem, satisfying link-time symbol closure for host-only microtests.

#include <cstdint>
#include <cstdlib>
#include <functional>

#include "nccl.h"
#include "bootstrap.h"

// A std::function seam, not a result code: commGetSplitInfo needs a test to write the (color, key) table into allData.
extern std::function<ncclResult_t(void* commState, void* allData, int size)>
    g_bootstrapAllGather;
ncclResult_t bootstrapAllGather(void* commState, void* allData, int size) {
  return g_bootstrapAllGather(commState, allData, size);
}
ncclResult_t bootstrapClose(void* commState) { ::abort(); }
ncclResult_t bootstrapCreateRoot(struct ncclBootstrapHandle* handle, bool idFromEnv) { ::abort(); }
extern ncclResult_t g_bootstrapGetUniqueIdResult;
extern ncclResult_t g_bcastGrowHandleResult;
extern uint64_t g_bootstrapHandleMagic;
extern int g_bcastGrowHandleCalls;
extern bool g_bcastGrowHandleIsRoot;
struct ncclBootstrapHandle g_bootstrapHandleTemplate{};
extern int g_bootstrapGetUniqueIdCalls;
void ResetBootstrapHandleTemplate() { g_bootstrapHandleTemplate = ncclBootstrapHandle{}; }

// Writes the WHOLE handle, not just magic: a caller that memcpy's too few bytes is invisible if addr/nRanks stay 0.
ncclResult_t bootstrapGetUniqueId(struct ncclBootstrapHandle* handle, struct ncclComm* comm) {
  ++g_bootstrapGetUniqueIdCalls;
  if (handle && g_bootstrapGetUniqueIdResult == ncclSuccess) {
    *handle = g_bootstrapHandleTemplate;
    handle->magic = g_bootstrapHandleMagic;
  }
  return g_bootstrapGetUniqueIdResult;
}

ncclResult_t bcastGrowHandle(struct ncclBootstrapHandle* handle, struct ncclComm* parent, bool isRoot) {
  ++g_bcastGrowHandleCalls;
  g_bcastGrowHandleIsRoot = isRoot;
  return g_bcastGrowHandleResult;
}
ncclResult_t bootstrapInit(int nHandles, void* handle, struct ncclComm* comm, struct ncclComm* parent) { ::abort(); }
ncclResult_t bootstrapIntraNodeBarrier(void* commState, int* ranks, int rank, int nranks, int tag) { ::abort(); }
ncclResult_t bootstrapSplit(unsigned long, struct ncclComm*, struct ncclComm*, int, int, int*) { ::abort(); }
