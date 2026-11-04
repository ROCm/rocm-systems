/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef GIN_HOST_ABCE_SDMA_H_
#define GIN_HOST_ABCE_SDMA_H_

#include "nccl.h"
#include "nccl_gin.h"
#include "plugin/nccl_net.h"

extern ncclGin_t ncclGinAbceSdmaPlugin;

struct ncclComm;
ncclResult_t ncclGinAbceBindResourceWindowSignals(struct ncclComm* comm, void* resourceUserPtr, size_t arenaByteOffset,
                                                   int nSignalSlots, int nSignalsPerContext);

void ncclGinAbceSetInitContext(void* initCtx, struct ncclComm* comm);

/** Clears plugin host singleton state (pending signal binds, MR refcounts). For unit tests. */
void ncclGinAbcePluginTestResetHostState(void);

/** Live per-comm init contexts, one per init() the plugin has not seen finalize() for. For tests. */
int ncclGinAbcePluginTestLiveInitCtxCount(void);

#endif
