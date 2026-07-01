/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef GIN_HOST_ROCSHMEM_GDA_H_
#define GIN_HOST_ROCSHMEM_GDA_H_

#include "nccl.h"
#include "gin/gin_host.h"
#include "plugin/nccl_net.h"

struct ncclComm;

// Init context: allocated by init(), populated by ncclGinRocshmemSetInitContext()
// after init() succeeds, passed to connect() as ctx parameter.
// Shared by all rocshmem GIN plugins (GDA and API).
struct ginRocshmemInitCtx {
  struct ncclComm *comm;
};

// Set RCCL-internal state into the plugin init context.
// Called from gin.cc immediately after a rocshmem plugin's init() succeeds.
// The GIN vtable init() doesn't receive comm, so this post-init injection is
// needed to pass comm->bootstrap to connect().
static inline void ncclGinRocshmemSetInitContext(void *initCtx, struct ncclComm *comm) {
  struct ginRocshmemInitCtx *ctx = (struct ginRocshmemInitCtx *)initCtx;
  ctx->comm = comm;
}

// The built-in plugin instance
extern ncclGin_t ncclGinRocshmemGdaPlugin;

#endif
