/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef GIN_HOST_ROCSHMEM_GDA_H_
#define GIN_HOST_ROCSHMEM_GDA_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "nccl.h"
#include "gin/gin_host.h"
#include "plugin/nccl_net.h"

// Called from gin_host.cc (with ncclComm context)
ncclResult_t ncclGinRocshmemGdaCreateContext(struct ncclComm *comm, void *collComm, int devId,
                                             int nSignals, int nCounters, void **outGinCtx,
                                             ncclNetDeviceHandle_v11_t **outDevHandle);
ncclResult_t ncclGinRocshmemGdaRegister(ncclGin_t *ginComm, void *ginCtx, void *addr, size_t size,
                                        int type, int mr_flags, void **mhandle, void **ginHandle);
ncclResult_t ncclGinRocshmemGdaDeregister(ncclGin_t *ginComm, void *ginCtx, void *mhandle);
ncclResult_t ncclGinRocshmemGdaDestroyContext(ncclGin_t *ginComm, void *ginCtx);
ncclResult_t ncclGinRocshmemGdaProgress(ncclGin_t *ginComm, void *ginCtx);
ncclResult_t ncclGinRocshmemGdaQueryLastError(ncclGin_t *ginComm, void *ginCtx, bool *hasError);

// Called from gin_plugin_rocshmem_gda.cc (plugin interface, no ncclComm)
ncclResult_t ncclGinRocshmemGdaCreateContextFromPlugin(int nSignals, int nCounters,
                                                       void **outGinCtx,
                                                       ncclNetDeviceHandle_v11_t **outDevHandle);
ncclResult_t ncclGinRocshmemGdaRegisterFromPlugin(void *addr, size_t size, int type,
                                                  uint64_t mr_flags, void **mhandle, void **ginHandle);
ncclResult_t ncclGinRocshmemGdaDeregisterFromPlugin(void *mhandle);
ncclResult_t ncclGinRocshmemGdaDestroyContextFromPlugin(void *ginCtx);

// The built-in plugin instance
extern ncclGin_t ncclGinRocshmemGdaPlugin;

#endif
