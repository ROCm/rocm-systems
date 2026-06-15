/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef GIN_HOST_ROCSHMEM_H_
#define GIN_HOST_ROCSHMEM_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "nccl.h"
#include "gin/gin_host.h"
#include "plugin/nccl_net.h"

// Called from gin_host.cc (with ncclComm context)
ncclResult_t ncclGinRocshmemApiCreateContext(struct ncclComm *comm, void *collComm, int devId,
                                          int nSignals, int nCounters, void **outGinCtx,
                                          ncclNetDeviceHandle_v11_t **outDevHandle);
ncclResult_t ncclGinRocshmemApiRegister(ncclGin_t *ginComm, void *ginCtx, void *addr, size_t size,
                                     int type, int mr_flags, void **mhandle, void **ginHandle);
ncclResult_t ncclGinRocshmemApiDeregister(ncclGin_t *ginComm, void *ginCtx, void *mhandle);
ncclResult_t ncclGinRocshmemApiDestroyContext(ncclGin_t *ginComm, void *ginCtx);
ncclResult_t ncclGinRocshmemApiProgress(ncclGin_t *ginComm, void *ginCtx);
ncclResult_t ncclGinRocshmemApiQueryLastError(ncclGin_t *ginComm, void *ginCtx, bool *hasError);

// Called from gin_plugin_rocshmem_api.cc (plugin interface, no ncclComm)
ncclResult_t ncclGinRocshmemApiCreateContextFromPlugin(int nSignals, int nCounters,
                                                     void **outGinCtx,
                                                     ncclNetDeviceHandle_v11_t **outDevHandle);
ncclResult_t ncclGinRocshmemApiRegisterFromPlugin(void *addr, size_t size, int type,
                                                uint64_t mr_flags, void **mhandle, void **ginHandle);
ncclResult_t ncclGinRocshmemApiDeregisterFromPlugin(void *mhandle);
ncclResult_t ncclGinRocshmemApiDestroyContextFromPlugin(void *ginCtx);

// The built-in plugin instance
extern ncclGin_v11_t ncclGinRocshmemApiPlugin;

#endif
