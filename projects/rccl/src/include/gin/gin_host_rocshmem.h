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

// Ensure rocSHMEM is initialized before Anvil SDMA setup (librccl is GIN-only).
ncclResult_t ncclGinRocshmemEnsureInit(struct ncclComm* comm);

// Called from gin_host.cc (with ncclComm context)
ncclResult_t ncclGinRocshmemCreateContext(struct ncclComm *comm, void *collComm, int devId,
                                          int nSignals, int nCounters, void **outGinCtx,
                                          ncclNetDeviceHandle_v11_t **outDevHandle);
ncclResult_t ncclGinRocshmemRegister(ncclGin_t *ginComm, void *ginCtx, void *addr, size_t size,
                                     int type, int mr_flags, void **mhandle, void **ginHandle);
ncclResult_t ncclGinRocshmemDeregister(ncclGin_t *ginComm, void *ginCtx, void *mhandle);
ncclResult_t ncclGinRocshmemDestroyContext(ncclGin_t *ginComm, void *ginCtx);
ncclResult_t ncclGinRocshmemFinalizeIfOwned(struct ncclComm* comm);
ncclResult_t ncclGinRocshmemProgress(ncclGin_t *ginComm, void *ginCtx);
ncclResult_t ncclGinRocshmemQueryLastError(ncclGin_t *ginComm, void *ginCtx, bool *hasError);

// Called from gin_plugin_rocshmem.cc (plugin interface, no ncclComm)
ncclResult_t ncclGinRocshmemCreateContextFromPlugin(int nSignals, int nCounters,
                                                     void **outGinCtx,
                                                     ncclNetDeviceHandle_v11_t **outDevHandle);
ncclResult_t ncclGinRocshmemRegisterFromPlugin(void *addr, size_t size, int type,
                                                uint64_t mr_flags, void **mhandle, void **ginHandle);
ncclResult_t ncclGinRocshmemDeregisterFromPlugin(void *mhandle);
ncclResult_t ncclGinRocshmemDestroyContextFromPlugin(void *ginCtx);

// The built-in plugin instance
extern ncclGin_t ncclGinRocshmem;

#endif
