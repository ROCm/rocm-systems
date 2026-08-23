/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Fail-loud stub floor for the bootstrap subsystem, shared by host-only
// microtests. These satisfy a unit-under-test's link-time symbol closure; the
// shallower tests never call them (abort-on-call). A test that needs to drive
// one of these replaces that individual entry with a real fake.

#include <cstdint>
#include <cstdlib>
#include <functional>

#include "nccl.h"
// Full definition of ncclBootstrapHandle: the bootstrapGetUniqueId fake stamps
// handle->magic, so a forward declaration is not enough.
#include "bootstrap.h"

// Controllable (was a fail-loud abort): commGetSplitInfo (init.cc:2496) needs a
// test to WRITE the allgathered (color, key) table into allData, so this is a
// std::function seam rather than a result code. Defined in init_fakes.cc and
// declared here rather than including init_fakes.h, which would drag the HIP and
// nccl fake headers into this stub TU. Defaults to failure so the other three
// bootstrapAllGather call sites in init.cc stay fail-fast.
extern std::function<ncclResult_t(void* commState, void* allData, int size)>
    g_bootstrapAllGather;
ncclResult_t bootstrapAllGather(void* commState, void* allData, int size) {
  return g_bootstrapAllGather(commState, allData, size);
}
ncclResult_t bootstrapClose(void* commState) { ::abort(); }
ncclResult_t bootstrapCreateRoot(struct ncclBootstrapHandle* handle, bool idFromEnv) { ::abort(); }
// ncclCommGetUniqueId() seams. bootstrapGetUniqueId was a fail-loud abort;
// bcastGrowHandle had NO fake at all -- it is defined in src/bootstrap.cc, which
// this target does not compile, and the binary only linked because
// --gc-sections dropped ncclCommGetUniqueId_impl entirely. Both are defined in
// init_fakes.cc; declared here rather than including init_fakes.h, which would
// drag the HIP/nccl fake headers into this stub TU.
extern ncclResult_t g_bootstrapGetUniqueIdResult;
extern ncclResult_t g_bcastGrowHandleResult;
extern uint64_t g_bootstrapHandleMagic;
extern int g_bcastGrowHandleCalls;
extern bool g_bcastGrowHandleIsRoot;

ncclResult_t bootstrapGetUniqueId(struct ncclBootstrapHandle* handle, struct ncclComm* comm) {
  // Stamp the handle so a test can prove THIS is what gets copied out.
  if (handle && g_bootstrapGetUniqueIdResult == ncclSuccess) handle->magic = g_bootstrapHandleMagic;
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
