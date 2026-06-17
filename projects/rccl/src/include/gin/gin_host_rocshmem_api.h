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

// Init context: allocated by init(), populated by ncclGinRocshmemSetInitContext()
// after init() succeeds, passed to connect() as ctx parameter.
struct ginRocshmemInitCtx {
  struct ncclComm *comm;
};

// Set RCCL-internal state into the plugin init context.
// Called from gin.cc immediately after a rocshmem plugin's init() succeeds.
void ncclGinRocshmemSetInitContext(void *initCtx, struct ncclComm *comm);

#endif
