/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef GIN_HOST_ANVIL_H_
#define GIN_HOST_ANVIL_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "nccl.h"
#include "gin/gin_host.h"
#include "plugin/nccl_net.h"

// Called from gin_host.cc (with ncclComm context)
ncclResult_t ncclGinAnvilCreateContext(struct ncclComm *comm, void *collComm, int devId,
                                       int nSignals, int nCounters, void **outGinCtx,
                                       ncclNetDeviceHandle_v11_t **outDevHandle);
ncclResult_t ncclGinAnvilRegister(ncclGin_t *ginComm, void *ginCtx, void *addr, size_t size,
                                  int type, int mr_flags, void **mhandle, void **ginHandle);
ncclResult_t ncclGinAnvilDeregister(ncclGin_t *ginComm, void *ginCtx, void *mhandle);
ncclResult_t ncclGinAnvilDestroyContext(ncclGin_t *ginComm, void *ginCtx);
ncclResult_t ncclGinAnvilProgress(ncclGin_t *ginComm, void *ginCtx);
ncclResult_t ncclGinAnvilQueryLastError(ncclGin_t *ginComm, void *ginCtx, bool *hasError);

// The built-in plugin instance
extern ncclGin_t ncclGinAnvilPlugin;

#endif
