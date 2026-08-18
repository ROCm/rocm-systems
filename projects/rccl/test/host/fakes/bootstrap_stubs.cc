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

#include "nccl.h"

struct ncclBootstrapHandle;
struct ncclComm;

ncclResult_t bootstrapAllGather(void* commState, void* allData, int size) { ::abort(); }
ncclResult_t bootstrapClose(void* commState) { ::abort(); }
ncclResult_t bootstrapCreateRoot(struct ncclBootstrapHandle* handle, bool idFromEnv) { ::abort(); }
ncclResult_t bootstrapGetUniqueId(struct ncclBootstrapHandle* handle, struct ncclComm* comm) { ::abort(); }
ncclResult_t bootstrapInit(int nHandles, void* handle, struct ncclComm* comm, struct ncclComm* parent) { ::abort(); }
ncclResult_t bootstrapIntraNodeBarrier(void* commState, int* ranks, int rank, int nranks, int tag) { ::abort(); }
ncclResult_t bootstrapSplit(unsigned long, struct ncclComm*, struct ncclComm*, int, int, int*) { ::abort(); }
