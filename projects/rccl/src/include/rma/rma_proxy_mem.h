/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _NCCL_RMA_PROXY_MEM_H_
#define _NCCL_RMA_PROXY_MEM_H_

#include <string.h>
#include "nccl.h"
#include "checks.h"
#include "alloc.h"
#include "gdrwrap.h"

// [RCCL] CPU-accessible alloc/free helpers shared by rma_proxy.cc and the
// split-out rma_proxy_launch.cc. Originally lived in gin_host_proxy.cc; the
// NCCL 2.29.7 patch removed them upstream but AMD's rma_proxy still needs them.
// They mirror the gin_host_proxy templates and likewise ignore the manager
// argument for now. Defined here (file-local via static, matching gdrwrap.h's
// convention) so both translation units can instantiate them.
template <typename T>
static ncclResult_t allocMemCPUAccessible(T **ptr, T **devPtr, size_t nelem, int /*host_flags*/,
                                          void **gdrHandle, struct ncclMemManager* manager,
                                          bool forceHost = false) {
  if (ncclGdrCopy && !forceHost) {
    NCCLCHECK(ncclGdrCudaCalloc(ptr, devPtr, nelem, gdrHandle, manager));
  } else {
    NCCLCHECK(ncclCuMemHostAlloc((void **)ptr, NULL, nelem * sizeof(T)));
    memset((void *)*ptr, 0, nelem * sizeof(T));
    *devPtr = *ptr;
    if (gdrHandle) *gdrHandle = NULL;
  }
  return ncclSuccess;
}

template <typename T>
static ncclResult_t freeMemCPUAccessible(T *ptr, void *gdrHandle, struct ncclMemManager* manager) {
  if (gdrHandle != NULL) {
    NCCLCHECK(ncclGdrCudaFree(gdrHandle, manager));
  } else {
    NCCLCHECK(ncclCuMemHostFree(ptr));
  }
  return ncclSuccess;
}

#endif // _NCCL_RMA_PROXY_MEM_H_
