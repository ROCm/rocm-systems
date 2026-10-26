/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef GIN_HOST_ANVIL_SDMA_H_
#define GIN_HOST_ANVIL_SDMA_H_

#include "nccl.h"
#include "nccl_gin.h"
#include "plugin/nccl_net.h"

extern ncclGin_t ncclGinAnvilSdmaPlugin;

struct ncclComm;
ncclResult_t ncclGinAnvilBindResourceWindowSignals(struct ncclComm* comm, void* resourceUserPtr, size_t arenaByteOffset,
                                                   int nContexts, int nSignalsPerContext);

void ncclGinAnvilSetInitContext(void* initCtx, struct ncclComm* comm);

/** Clears plugin host singleton state (pending signal binds, MR refcounts). For unit tests. */
void ncclGinAnvilPluginTestResetHostState(void);

/** Fabric MR bookkeeping, for unit tests: the entry count of each per-buffer map,
 *  and the reference count of the (collComm, data) key, or 0 if it is not registered. */
struct ncclGinAnvilFabricTestState {
  size_t handlers;
  size_t handles;
  size_t refcounts;
  int refcount;
};
void ncclGinAnvilPluginTestFabricState(void* collComm, void* data, struct ncclGinAnvilFabricTestState* out);

#endif
