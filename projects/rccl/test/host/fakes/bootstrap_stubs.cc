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

#include "fakes/bootstrap_stubs.h"

// A std::function seam, not a result code: commGetSplitInfo needs a test to write the (color, key) table into allData.
extern std::function<ncclResult_t(void* commState, void* allData, int size)>
    g_bootstrapAllGather;
ncclResult_t bootstrapAllGather(void* commState, void* allData, int size) {
  return g_bootstrapAllGather(commState, allData, size);
}
ncclResult_t bootstrapClose(void* commState) { ::abort(); }
static ncclResult_t DefaultBootstrapCreateRoot(struct ncclBootstrapHandle*, bool) { ::abort(); }
std::function<ncclResult_t(struct ncclBootstrapHandle*, bool)> g_bootstrapCreateRoot = DefaultBootstrapCreateRoot;
ncclResult_t bootstrapCreateRoot(struct ncclBootstrapHandle* handle, bool idFromEnv) {
  return g_bootstrapCreateRoot(handle, idFromEnv);
}
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

extern std::function<ncclResult_t(struct ncclBootstrapHandle*, struct ncclComm*, bool)> g_bcastGrowHandle;

ncclResult_t bcastGrowHandle(struct ncclBootstrapHandle* handle, struct ncclComm* parent, bool isRoot) {
  ++g_bcastGrowHandleCalls;
  g_bcastGrowHandleIsRoot = isRoot;
  return g_bcastGrowHandle(handle, parent, isRoot);
}
static ncclResult_t DefaultBootstrapInit(int, void*, struct ncclComm*, struct ncclComm*) { ::abort(); }
std::function<ncclResult_t(int, void*, struct ncclComm*, struct ncclComm*)> g_bootstrapInit = DefaultBootstrapInit;
ncclResult_t bootstrapInit(int nHandles, void* handle, struct ncclComm* comm, struct ncclComm* parent) {
  return g_bootstrapInit(nHandles, handle, comm, parent);
}

ncclResult_t bootstrapIntraNodeBarrier(void* commState, int* ranks, int rank, int nranks, int tag) { ::abort(); }

static ncclResult_t DefaultBootstrapSplit(uint64_t, struct ncclComm*, struct ncclComm*, int, int, int*) { ::abort(); }
std::function<ncclResult_t(uint64_t, struct ncclComm*, struct ncclComm*, int, int, int*)> g_bootstrapSplit =
    DefaultBootstrapSplit;
ncclResult_t bootstrapSplit(unsigned long commHash, struct ncclComm* comm, struct ncclComm* parent, int color, int key,
                            int* parentRanks) {
  return g_bootstrapSplit(commHash, comm, parent, color, key, parentRanks);
}

void ResetBootstrapStubs() {
  g_bootstrapInit = DefaultBootstrapInit;
  g_bootstrapSplit = DefaultBootstrapSplit;
  g_bootstrapCreateRoot = DefaultBootstrapCreateRoot;
}
