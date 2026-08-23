/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Fail-loud stub floor for the bootstrap subsystem, shared by host-only
// microtests. These satisfy a unit-under-test's link-time symbol closure; the
// shallower tests never call them (abort-on-call). A test that needs to drive
// one of these replaces that individual entry with a real fake.

#include <cstdlib>
#include <functional>

#include "nccl.h"

struct ncclBootstrapHandle;
struct ncclComm;

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
ncclResult_t bootstrapGetUniqueId(struct ncclBootstrapHandle* handle, struct ncclComm* comm) { ::abort(); }
ncclResult_t bootstrapInit(int nHandles, void* handle, struct ncclComm* comm, struct ncclComm* parent) { ::abort(); }
ncclResult_t bootstrapIntraNodeBarrier(void* commState, int* ranks, int rank, int nranks, int tag) { ::abort(); }
ncclResult_t bootstrapSplit(unsigned long, struct ncclComm*, struct ncclComm*, int, int, int*) { ::abort(); }
