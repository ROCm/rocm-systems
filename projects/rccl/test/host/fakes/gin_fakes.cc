/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Implementation of the GIN fakes. See gin_fakes.h.

#include "gin_fakes.h"

// src/plugin/gin.cc
ncclResult_t g_ncclGinInitResult = ncclSuccess;
ncclResult_t ncclGinInit(struct ncclComm*) { return g_ncclGinInitResult; }
ncclResult_t ncclGinInitFromParent(struct ncclComm*, struct ncclComm*) { return g_ncclGinInitResult; }

// src/gin/gin_host.cc
bool g_ginHasError = false;
ncclResult_t ncclGinQueryLastError(struct ncclGinState*, bool* hasError) {
  if (hasError) *hasError = g_ginHasError;
  return ncclSuccess;
}

void ResetGinFakes() {
  g_ncclGinInitResult = ncclSuccess;
  g_ginHasError = false;
}
